/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#include <sstream>


#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "obj_loader.h"

#include "hello_vulkan.h"
#include "nvh/alignment.hpp"
#include "nvh/cameramanipulator.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvk/buffers_vk.hpp"
#include <random>

extern std::vector<std::string> defaultSearchPaths;


//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const VkInstance& instance, const VkDevice& device, const VkPhysicalDevice& physicalDevice, uint32_t queueFamily)
{
  AppBaseVk::setup(instance, device, physicalDevice, queueFamily);
  m_alloc.init(instance, device, physicalDevice);
  m_debug.setup(m_device);
  m_offscreenDepthFormat = nvvk::findDepthFormat(physicalDevice);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer(const VkCommandBuffer& cmdBuf)
{
  // Prepare new UBO contents on host.
  const float    aspectRatio = m_size.width / static_cast<float>(m_size.height);
  GlobalUniforms hostUBO     = {};
  const auto&    view        = CameraManip.getMatrix();
  const auto&    proj        = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // proj[1][1] *= -1;  // Inverting Y for Vulkan (not needed with perspectiveVK).

  hostUBO.viewProj    = proj * view;
  hostUBO.viewInverse = nvmath::invert(view);
  hostUBO.projInverse = nvmath::invert(proj);

  // UBO on the device, and what stages access it.
  VkBuffer deviceUBO      = m_bGlobals.buffer;
  auto     uboUsageStages = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;

  // Ensure that the modified UBO is not visible to previous frames.
  VkBufferMemoryBarrier beforeBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  beforeBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
  beforeBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  beforeBarrier.buffer        = deviceUBO;
  beforeBarrier.offset        = 0;
  beforeBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, uboUsageStages, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &beforeBarrier, 0, nullptr);


  // Schedule the host-to-device upload. (hostUBO is copied into the cmd
  // buffer so it is okay to deallocate when the function returns).
  vkCmdUpdateBuffer(cmdBuf, m_bGlobals.buffer, 0, sizeof(GlobalUniforms), &hostUBO);

  // Making sure the updated UBO will be visible.
  VkBufferMemoryBarrier afterBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
  afterBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  afterBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  afterBarrier.buffer        = deviceUBO;
  afterBarrier.offset        = 0;
  afterBarrier.size          = sizeof(hostUBO);
  vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, uboUsageStages, VK_DEPENDENCY_DEVICE_GROUP_BIT, 0,
                       nullptr, 1, &afterBarrier, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  auto nbTxt = static_cast<uint32_t>(m_textures.size());

  // Camera matrices
  m_descSetLayoutBind.addBinding(SceneBindings::eGlobals, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_RAYGEN_BIT_KHR);
  // Obj descriptions
  m_descSetLayoutBind.addBinding(SceneBindings::eObjDescs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);
  // Textures
  m_descSetLayoutBind.addBinding(SceneBindings::eTextures, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, nbTxt,
                                 VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR);

  // DispObj descriptions ( -> custom intersection)
  m_descSetLayoutBind.addBinding(SceneBindings::eDispObjDescs, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR);

  m_descSetLayout = m_descSetLayoutBind.createLayout(m_device);
  m_descPool      = m_descSetLayoutBind.createPool(m_device, 1);
  m_descSet       = nvvk::allocateDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<VkWriteDescriptorSet> writes;

  // Camera matrices and scene description
  VkDescriptorBufferInfo dbiUnif{m_bGlobals.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eGlobals, &dbiUnif));

  VkDescriptorBufferInfo dbiSceneDesc{m_bObjDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eObjDescs, &dbiSceneDesc));

  // All texture samplers
  std::vector<VkDescriptorImageInfo> diit;
  for(auto& texture : m_textures)
  {
    diit.emplace_back(texture.descriptor);
  }
  writes.emplace_back(m_descSetLayoutBind.makeWriteArray(m_descSet, SceneBindings::eTextures, diit.data()));

  // @author Josias

  // Models constaining displacement textures
  VkDescriptorBufferInfo dbiSceneDispDesc{m_bDispObjDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(m_descSetLayoutBind.makeWrite(m_descSet, SceneBindings::eDispObjDescs, &dbiSceneDispDesc));
  // \@author Josias

  // Writing the information
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(PushConstantRaster)};

  // Creating the Pipeline Layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_descSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_pipelineLayout);


  // Creating the Pipeline
  std::vector<std::string>                paths = defaultSearchPaths;
  nvvk::GraphicsPipelineGeneratorCombined gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState.depthTestEnable = true;
  gpb.addShader(nvh::loadFile("spv/vert_shader.vert.spv", true, paths, true), VK_SHADER_STAGE_VERTEX_BIT);
  gpb.addShader(nvh::loadFile("spv/frag_shader.frag.spv", true, paths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  gpb.addBindingDescription({0, sizeof(VertexObj)});
  gpb.addAttributeDescriptions({
      {0, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, pos))},
      {1, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, nrm))},
      {2, 0, VK_FORMAT_R32G32B32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, color))},
      {3, 0, VK_FORMAT_R32G32_SFLOAT, static_cast<uint32_t>(offsetof(VertexObj, texCoord))},
  });

  m_graphicsPipeline = gpb.createPipeline();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadModel(const std::string& filename, nvmath::mat4f transform)
{
  LOGI("Loading File:  %s \n", filename.c_str());
  ObjLoader loader;
  loader.loadModel(filename);
  bool displacement = false;

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = nvmath::pow(m.ambient, 2.2f);
    m.diffuse  = nvmath::pow(m.diffuse, 2.2f);
    m.specular = nvmath::pow(m.specular, 2.2f);

    if(m.dispTextureID >= 0)
      displacement = true;
  }

  if(displacement)
  {
    loadDisplacementModel(loader, transform);
  }
  else
  {
    loadNonDisplacementModel(loader, transform);
  }
}

//--------------------------------------------------------------------------------------------------
// Setting all the buffers for a loaded OBJ file that does not contain displacement textures
//
void HelloVulkan::loadNonDisplacementModel(ObjLoader loader, nvmath::mat4f transform)
{
  ObjModel model;
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool  cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer    cmdBuf = cmdBufGet.createCommandBuffer();
  VkBufferUsageFlags flag   = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  // #VKRay
  VkBufferUsageFlags rayTracingFlags =
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  model.vertexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags);
  model.indexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  // Creates all textures found and find the offset for this model
  auto txtOffset = static_cast<uint32_t>(m_textures.size());

  createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  std::string objNb = std::to_string(m_objModel.size());
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb)));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb)));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb)));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb)));

  // Keeping transformation matrix of the instance
  ObjInstance instance;
  instance.transform = transform;
  instance.objIndex  = static_cast<uint32_t>(m_objModel.size());
  m_instances.push_back(instance);

  // Creating information for device access
  ObjDesc desc;
  desc.txtOffset            = txtOffset;
  desc.vertexAddress        = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  desc.indexAddress         = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);
  desc.materialAddress      = nvvk::getBufferDeviceAddress(m_device, model.matColorBuffer.buffer);
  desc.materialIndexAddress = nvvk::getBufferDeviceAddress(m_device, model.matIndexBuffer.buffer);

  // Keeping the obj host model and device description
  m_objModel.emplace_back(model);
  m_objDesc.emplace_back(desc);
}

//--------------------------------------------------------------------------------------------------
// Setting all the buffers for a loaded OBJ file that contains displacement textures
//
void HelloVulkan::loadDisplacementModel(ObjLoader loader, nvmath::mat4f transform)
{
  // m_matIndx contains a material index for every triangles, so it's an easy way to grab the total triangle count
  int               nbTriangles =  loader.m_indices.size() / 3;
  std::vector<Aabb> aabbs;
  aabbs.reserve(nbTriangles);

  for(int i = 0; i < nbTriangles; i++)
  {
    VertexObj v0 = loader.m_vertices[loader.m_indices[i * 3]];
    VertexObj v1 = loader.m_vertices[loader.m_indices[i * 3 + 1]];
    VertexObj v2 = loader.m_vertices[loader.m_indices[i * 3 + 2]];

    TriangleObj t{v0, v1, v2, 0};
    Aabb        aabb = createAabbFromTriangle(t);
    aabbs.emplace_back(aabb);
  }

  DispObjModel model;
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  // Create the buffers on Device and copy vertices, indices and materials
  nvvk::CommandPool  cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer    cmdBuf = cmdBufGet.createCommandBuffer();
  VkBufferUsageFlags flag   = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

  // #VKRay
  VkBufferUsageFlags rayTracingFlags =
      flag | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  model.vertexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | rayTracingFlags);
  model.indexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | rayTracingFlags);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | flag);
  model.aabbBuffer     = m_alloc.createBuffer(cmdBuf, aabbs,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                  | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | flag);
  // Creates all textures found and find the offset for this model
  auto txtOffset = static_cast<uint32_t>(m_textures.size());

  createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();

  std::string objNb = std::to_string(m_dispObjModel.size());
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb)));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb)));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb)));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb)));
  m_debug.setObjectName(model.aabbBuffer.buffer, (std::string("aabb_" + objNb)));

  // Keeping transformation matrix of the instance
  ObjInstance instance;
  instance.transform = transform;
  instance.objIndex  = static_cast<uint32_t>(m_dispObjModel.size());
  m_dispInstances.push_back(instance);

  // Creating information for device access
  DispObjDesc desc;
  desc.txtOffset            = txtOffset;
  desc.vertexAddress        = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  desc.indexAddress         = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);
  desc.materialAddress      = nvvk::getBufferDeviceAddress(m_device, model.matColorBuffer.buffer);
  desc.materialIndexAddress = nvvk::getBufferDeviceAddress(m_device, model.matIndexBuffer.buffer);
  desc.aabbAddress          = nvvk::getBufferDeviceAddress(m_device, model.aabbBuffer.buffer);

  // Keeping the obj host model and device description
  m_dispObjModel.emplace_back(model);
  m_dispObjDesc.emplace_back(desc);
}


//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  m_bGlobals = m_alloc.createBuffer(sizeof(GlobalUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  m_debug.setObjectName(m_bGlobals.buffer, "Globals");
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void HelloVulkan::createObjDescriptionBuffer()
{
  nvvk::CommandPool cmdGen(m_device, m_graphicsQueueIndex);

  // @author Josias
  // if the list of ObjDesc (m_objDesc) is empty, add a dummy ObjDesc so the pipeline doesn't crash
  if(m_objDesc.size() < 1)
  {
    ObjDesc dummy;
    m_objDesc.emplace_back(dummy);
  }
  // \@author Josias

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_bObjDesc  = m_alloc.createBuffer(cmdBuf, m_objDesc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

  // @author Josias
  // if the list of DispObjDesc (m_dispObjDesc) is empty, add a dummy DispObjDesc so the pipeline doesn't crash
  if(m_dispObjDesc.size() < 1)
  {
    DispObjDesc dummy;
    m_dispObjDesc.emplace_back(dummy);
  }
  m_bDispObjDesc = m_alloc.createBuffer(cmdBuf, m_dispObjDesc, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  // \@author Josias
  cmdGen.submitAndWait(cmdBuf);
  m_alloc.finalizeAndReleaseStaging();
  m_debug.setObjectName(m_bObjDesc.buffer, "ObjDescs");
  // @author Josias
  m_debug.setObjectName(m_bDispObjDesc.buffer, "DispObjDescs");
  // \@author Josias
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const VkCommandBuffer& cmdBuf, const std::vector<std::string>& textures)
{
  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;

  VkFormat format = VK_FORMAT_R8G8B8A8_SRGB;

  // If no textures are present, create a dummy one to accommodate the pipeline layout
  if(textures.empty() && m_textures.empty())
  {
    nvvk::Texture texture;

    std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
    VkDeviceSize           bufferSize      = sizeof(color);
    auto                   imgSize         = VkExtent2D{1, 1};
    auto                   imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format);

    // Creating the dummy texture
    nvvk::Image           image  = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
    texture                      = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

    // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    nvvk::cmdBarrierImageLayout(cmdBuf, texture.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_textures.push_back(texture);
  }
  else
  {
    // Uploading all images
    for(const auto& texture : textures)
    {
      std::stringstream o;
      int               texWidth, texHeight, texChannels;
      o << "media/textures/" << texture;
      std::string txtFile = nvh::findFile(o.str(), defaultSearchPaths, true);

      stbi_uc* stbi_pixels = stbi_load(txtFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

      std::array<stbi_uc, 4> color{255u, 0u, 255u, 255u};

      stbi_uc* pixels = stbi_pixels;
      // Handle failure
      if(!stbi_pixels)
      {
        texWidth = texHeight = 1;
        texChannels          = 4;
        pixels               = reinterpret_cast<stbi_uc*>(color.data());
      }

      VkDeviceSize bufferSize      = static_cast<uint64_t>(texWidth) * texHeight * sizeof(uint8_t) * 4;
      auto         imgSize         = VkExtent2D{(uint32_t)texWidth, (uint32_t)texHeight};
      auto         imageCreateInfo = nvvk::makeImage2DCreateInfo(imgSize, format, VK_IMAGE_USAGE_SAMPLED_BIT, true);

      {
        nvvk::Image image = m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);
        nvvk::cmdGenerateMipmaps(cmdBuf, image.image, format, imgSize, imageCreateInfo.mipLevels);
        VkImageViewCreateInfo ivInfo  = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);
        nvvk::Texture         texture = m_alloc.createTexture(image, ivInfo, samplerCreateInfo);

        m_textures.push_back(texture);
      }

      stbi_image_free(stbi_pixels);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);

  m_alloc.destroy(m_bGlobals);
  m_alloc.destroy(m_bObjDesc);

  // @author Josias
  m_alloc.destroy(m_bDispObjDesc);
  // \@author Josias

  for(auto& m : m_objModel)
  {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
    m_alloc.destroy(m.matIndexBuffer);
  }

  // @author Josias
  for(auto& m : m_dispObjModel)
  {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
    m_alloc.destroy(m.matIndexBuffer);
    m_alloc.destroy(m.aabbBuffer);
  }
  // TODO: find out why sampler destruction throws an exception
  //vkDestroySampler(m_device, m_textureSampler, nullptr);
  //vkDestroyImageView(m_device, m_textureImageView, nullptr);
  //vkDestroyImage(m_device, m_textureImage, nullptr);
  vkFreeMemory(m_device, m_textureImageMemory, nullptr);
  // \@author Josias

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  // #Post
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  vkDestroyPipeline(m_device, m_postPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_postPipelineLayout, nullptr);
  vkDestroyDescriptorPool(m_device, m_postDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_postDescSetLayout, nullptr);
  vkDestroyRenderPass(m_device, m_offscreenRenderPass, nullptr);
  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);

  // #VKRay
  m_rtBuilder.destroy();
  vkDestroyDescriptorPool(m_device, m_rtDescPool, nullptr);
  vkDestroyDescriptorSetLayout(m_device, m_rtDescSetLayout, nullptr);
  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
  m_alloc.destroy(m_rtSBTBuffer);

  m_alloc.deinit();
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const VkCommandBuffer& cmdBuf)
{
  VkDeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  setViewport(cmdBuf);

  // Drawing all triangles
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1, &m_descSet, 0, nullptr);


  for(const HelloVulkan::ObjInstance& inst : m_instances)
  {
    auto& model            = m_objModel[inst.objIndex];
    m_pcRaster.objIndex    = inst.objIndex;  // Telling which object is drawn
    m_pcRaster.modelMatrix = inst.transform;

    vkCmdPushConstants(cmdBuf, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(PushConstantRaster), &m_pcRaster);
    vkCmdBindVertexBuffers(cmdBuf, 0, 1, &model.vertexBuffer.buffer, &offset);
    vkCmdBindIndexBuffer(cmdBuf, model.indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cmdBuf, model.nbIndices, 1, 0, 0, 0);
  }
  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  createOffscreenRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
}


//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////


//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);

  // Creating the color image
  {
    auto colorCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenColorFormat,
                                                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                                           | VK_IMAGE_USAGE_STORAGE_BIT);


    nvvk::Image           image  = m_alloc.createImage(colorCreateInfo);
    VkImageViewCreateInfo ivInfo = nvvk::makeImageViewCreateInfo(image.image, colorCreateInfo);
    VkSamplerCreateInfo   sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    m_offscreenColor                        = m_alloc.createTexture(image, ivInfo, sampler);
    m_offscreenColor.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  }

  // Creating the depth buffer
  auto depthCreateInfo = nvvk::makeImage2DCreateInfo(m_size, m_offscreenDepthFormat, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
  {
    nvvk::Image image = m_alloc.createImage(depthCreateInfo);


    VkImageViewCreateInfo depthStencilView{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    depthStencilView.viewType         = VK_IMAGE_VIEW_TYPE_2D;
    depthStencilView.format           = m_offscreenDepthFormat;
    depthStencilView.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    depthStencilView.image            = image.image;

    m_offscreenDepth = m_alloc.createTexture(image, depthStencilView);
  }

  // Setting the image layout for both color and depth
  {
    nvvk::CommandPool genCmdBuf(m_device, m_graphicsQueueIndex);
    auto              cmdBuf = genCmdBuf.createCommandBuffer();
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenColor.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    nvvk::cmdBarrierImageLayout(cmdBuf, m_offscreenDepth.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);

    genCmdBuf.submitAndWait(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass = nvvk::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat, 1, true,
                                                   true, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
  }


  // Creating the frame buffer for offscreen
  std::vector<VkImageView> attachments = {m_offscreenColor.descriptor.imageView, m_offscreenDepth.descriptor.imageView};

  vkDestroyFramebuffer(m_device, m_offscreenFramebuffer, nullptr);
  VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
  info.renderPass      = m_offscreenRenderPass;
  info.attachmentCount = 2;
  info.pAttachments    = attachments.data();
  info.width           = m_size.width;
  info.height          = m_size.height;
  info.layers          = 1;
  vkCreateFramebuffer(m_device, &info, nullptr, &m_offscreenFramebuffer);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline()
{
  // Push constants in the fragment shader
  VkPushConstantRange pushConstantRanges = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float)};

  // Creating the pipeline layout
  VkPipelineLayoutCreateInfo createInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  createInfo.setLayoutCount         = 1;
  createInfo.pSetLayouts            = &m_postDescSetLayout;
  createInfo.pushConstantRangeCount = 1;
  createInfo.pPushConstantRanges    = &pushConstantRanges;
  vkCreatePipelineLayout(m_device, &createInfo, nullptr, &m_postPipelineLayout);


  // Pipeline: completely generic, no vertices
  nvvk::GraphicsPipelineGeneratorCombined pipelineGenerator(m_device, m_postPipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("spv/passthrough.vert.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_VERTEX_BIT);
  pipelineGenerator.addShader(nvh::loadFile("spv/post.frag.spv", true, defaultSearchPaths, true), VK_SHADER_STAGE_FRAGMENT_BIT);
  pipelineGenerator.rasterizationState.cullMode = VK_CULL_MODE_NONE;
  m_postPipeline                                = pipelineGenerator.createPipeline();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
  m_postDescSetLayoutBind.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
  m_postDescSetLayout = m_postDescSetLayoutBind.createLayout(m_device);
  m_postDescPool      = m_postDescSetLayoutBind.createPool(m_device);
  m_postDescSet       = nvvk::allocateDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}


//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  VkWriteDescriptorSet writeDescriptorSets = m_postDescSetLayoutBind.makeWrite(m_postDescSet, 0, &m_offscreenColor.descriptor);
  vkUpdateDescriptorSets(m_device, 1, &writeDescriptorSets, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(VkCommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  setViewport(cmdBuf);

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  vkCmdPushConstants(cmdBuf, m_postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float), &aspectRatio);
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipelineLayout, 0, 1, &m_postDescSet, 0, nullptr);
  vkCmdDraw(cmdBuf, 3, 1, 0, 0);


  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
void HelloVulkan::initRayTracing()
{
  // Requesting ray tracing properties
  VkPhysicalDeviceProperties2 prop2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
  prop2.pNext = &m_rtProperties;
  vkGetPhysicalDeviceProperties2(m_physicalDevice, &prop2);

  m_rtBuilder.setup(m_device, &m_alloc, m_graphicsQueueIndex);
}

//--------------------------------------------------------------------------------------------------
// Convert an OBJ model into the ray tracing geometry used to build the BLAS
//
auto HelloVulkan::objectToVkGeometryKHR(const ObjModel& model)
{
  // Fetch raw device addresses
  VkDeviceAddress vertexAddress = nvvk::getBufferDeviceAddress(m_device, model.vertexBuffer.buffer);
  VkDeviceAddress indexAddress  = nvvk::getBufferDeviceAddress(m_device, model.indexBuffer.buffer);

  uint32_t maxPrimitiveCount = model.nbIndices / 3;

  // describe buffer as array of VertexObj
  VkAccelerationStructureGeometryTrianglesDataKHR triangles{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
  triangles.vertexFormat             = VK_FORMAT_R32G32B32_SFLOAT;  // vec3 vertex position data
  triangles.vertexData.deviceAddress = vertexAddress;
  triangles.vertexStride             = sizeof(VertexObj);
  // describe index data (32bit unsigned int)
  triangles.indexType               = VK_INDEX_TYPE_UINT32;
  triangles.indexData.deviceAddress = indexAddress;
  // identity transform by setting to null pointer
  triangles.transformData = {};
  triangles.maxVertex     = model.nbVertices;

  // identify the above data as containing opaque triangles
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType       = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  asGeom.flags              = VK_GEOMETRY_OPAQUE_BIT_KHR;
  asGeom.geometry.triangles = triangles;

  // entire array will be used to build the BLAS
  VkAccelerationStructureBuildRangeInfoKHR offset;
  offset.firstVertex     = 0;
  offset.primitiveCount  = maxPrimitiveCount;
  offset.primitiveOffset = 0;
  offset.transformOffset = 0;

  // our BLAS is only made from one geometry (could be made of many geometries)
  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);

  return input;
}

//--------------------------------------------------------------------------------------------------
// Convert an OBJ model containing displacement textures into the ray tracing geometry used to build the BLAS.
// For each triangle of the OBJ model a precalculated AABB is added to the BLAS builder.
//
auto HelloVulkan::displacementObjectToVkGeometryKHR(const DispObjModel& model)
{
  VkDeviceAddress dataAddress = nvvk::getBufferDeviceAddress(m_device, model.aabbBuffer.buffer);

  VkAccelerationStructureGeometryAabbsDataKHR aabbs{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR};
  aabbs.data.deviceAddress = dataAddress;
  aabbs.stride             = sizeof(Aabb);

  // Setting up the build info of the acceleration (C version, c++ gives wrong type)
  VkAccelerationStructureGeometryKHR asGeom{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  asGeom.geometryType   = VK_GEOMETRY_TYPE_AABBS_KHR;
  asGeom.flags          = VK_GEOMETRY_OPAQUE_BIT_KHR;
  asGeom.geometry.aabbs = aabbs;

  // TODO: SET PRIMITIVE COUNT!
  VkAccelerationStructureBuildRangeInfoKHR offset{};
  offset.firstVertex     = 0;
  offset.primitiveCount  = (uint32_t)model.nbIndices / 3;  // Nb aabb (remember: one AABB for each triangle
  offset.primitiveOffset = 0;
  offset.transformOffset = 0;

  nvvk::RaytracingBuilderKHR::BlasInput input;
  input.asGeometry.emplace_back(asGeom);
  input.asBuildOffsetInfo.emplace_back(offset);
  return input;
}

//--------------------------------------------------------------------------------------------------
// create Bottom Level Acceleration Structure
void HelloVulkan::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<nvvk::RaytracingBuilderKHR::BlasInput> allBlas;
  allBlas.reserve(m_objModel.size() + m_dispObjModel.size());
  for(const auto& obj : m_objModel)
  {
    auto blas = objectToVkGeometryKHR(obj);

    // We could add more geometry in each BLAS, but we add only one for now
    allBlas.emplace_back(blas);
  }

  // @author Josias
  for(const auto& obj : m_dispObjModel)
  {
    auto blas = displacementObjectToVkGeometryKHR(obj);

    allBlas.emplace_back(blas);
  }

  // \@author Josias
  m_rtBuilder.buildBlas(allBlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// create Top Level Acceleration Structure
void HelloVulkan::createTopLevelAS()
{
  std::vector<VkAccelerationStructureInstanceKHR> tlas;
  auto                                            nbObj     = static_cast<uint32_t>(m_instances.size());
  auto                                            nbDispObj = static_cast<uint32_t>(m_dispInstances.size());
  tlas.reserve(nbObj + nbDispObj - 1);
  // last instance in m_instances is dummy instance for Custom Triangles
  for(uint32_t i = 0; i < nbObj; i++)
  {
    const auto& inst = m_instances[i];

    VkAccelerationStructureInstanceKHR rayInst{};
    rayInst.transform                      = nvvk::toTransformMatrixKHR(inst.transform);  // Position of the instance
    rayInst.instanceCustomIndex            = inst.objIndex;                               // gl_InstanceCustomIndexEXT
    rayInst.accelerationStructureReference = m_rtBuilder.getBlasDeviceAddress(inst.objIndex);
    rayInst.flags                          = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.mask                           = 0xFF;       //  Only be hit if rayMask & instance.mask != 0
    rayInst.instanceShaderBindingTableRecordOffset = 0;  // Hit Group 0 uses default intersection and then raytrace.rchit
    tlas.emplace_back(rayInst);
  }

  for(uint32_t i = 0; i < nbDispObj; i++)
  {
    const auto& inst = m_dispInstances[i];

    VkAccelerationStructureInstanceKHR rayInst{};
    rayInst.transform = nvvk::toTransformMatrixKHR(inst.transform);  // Position of the instance
    // will be added after the non displacement models, so their number has to get added to the index
    rayInst.instanceCustomIndex                    = inst.objIndex;  // gl_InstanceCustomIndexEXT
    rayInst.accelerationStructureReference         = m_rtBuilder.getBlasDeviceAddress(inst.objIndex + nbObj);
    rayInst.flags                                  = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    rayInst.mask                                   = 0xFF;  //  Only be hit if rayMask & instance.mask != 0
    rayInst.instanceShaderBindingTableRecordOffset = 1;  // Hit Group 1 is for custom intersection and then raytrace2.rchit
    tlas.emplace_back(rayInst);
  }

  m_rtBuilder.buildTlas(tlas, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
}

//--------------------------------------------------------------------------------------------------
// create descriptor set for ray tracing by adding the TLAS and the output image to it
void HelloVulkan::createRtDescriptorSet()
{
  // Top-level acceleration structure, usable by both the ray generation and the closest hit (to
  // shoot shadow rays)
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR);  // TLAS
  m_rtDescSetLayoutBind.addBinding(RtxBindings::eOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
                                   VK_SHADER_STAGE_RAYGEN_BIT_KHR);  // Output image

  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);

  VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  allocateInfo.descriptorPool     = m_rtDescPool;
  allocateInfo.descriptorSetCount = 1;
  allocateInfo.pSetLayouts        = &m_rtDescSetLayout;
  vkAllocateDescriptorSets(m_device, &allocateInfo, &m_rtDescSet);


  VkAccelerationStructureKHR                   tlas = m_rtBuilder.getAccelerationStructure();
  VkWriteDescriptorSetAccelerationStructureKHR descASInfo{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  descASInfo.accelerationStructureCount = 1;
  descASInfo.pAccelerationStructures    = &tlas;
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};

  std::vector<VkWriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eTlas, &descASInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo));
  vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// writes the output image to the descriptor set (required when changing resolution)
void HelloVulkan::updateRtDescriptorSet()
{
  // (1) Output buffer
  VkDescriptorImageInfo imageInfo{{}, m_offscreenColor.descriptor.imageView, VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet  wds = m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, RtxBindings::eOutImage, &imageInfo);
  vkUpdateDescriptorSets(m_device, 1, &wds, 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Ray Tracer Pipeline: all shaders, ray generation (raygen), closest hit (chit), miss
void HelloVulkan::createRtPipeline()
{
  enum StageIndices
  {
    eRaygen,
    eMiss,
    eMiss2,
    eClosestHit,
    eClosestHit2,
    eIntersection,
    eShaderGroupCount
  };

  // All stages
  std::array<VkPipelineShaderStageCreateInfo, eShaderGroupCount> stages{};
  VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
  stage.pName = "main";  // All the same entry point
  // Raygen
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rgen.spv", true, defaultSearchPaths, true));
  stage.stage     = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
  stages[eRaygen] = stage;
  // Miss
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rmiss.spv", true, defaultSearchPaths, true));
  stage.stage   = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss] = stage;
  // The second miss shader is invoked when a shadow ray misses the geometry. It simply indicates that no occlusion has been found
  stage.module =
      nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytraceShadow.rmiss.spv", true, defaultSearchPaths, true));
  stage.stage    = VK_SHADER_STAGE_MISS_BIT_KHR;
  stages[eMiss2] = stage;
  // Hit Group - Closest Hit
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rchit.spv", true, defaultSearchPaths, true));
  stage.stage         = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit] = stage;
  // Closest hit
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace2.rchit.spv", true, defaultSearchPaths, true));
  stage.stage          = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
  stages[eClosestHit2] = stage;
  // Intersection
  stage.module = nvvk::createShaderModule(m_device, nvh::loadFile("spv/raytrace.rint.spv", true, defaultSearchPaths, true));
  stage.stage           = VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
  stages[eIntersection] = stage;

  // Shader groups
  VkRayTracingShaderGroupCreateInfoKHR group{VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
  group.anyHitShader       = VK_SHADER_UNUSED_KHR;
  group.closestHitShader   = VK_SHADER_UNUSED_KHR;
  group.generalShader      = VK_SHADER_UNUSED_KHR;
  group.intersectionShader = VK_SHADER_UNUSED_KHR;

  // Raygen
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eRaygen;
  m_rtShaderGroups.push_back(group);

  // Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss;
  m_rtShaderGroups.push_back(group);

  // Shadow Miss
  group.type          = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
  group.generalShader = eMiss2;
  m_rtShaderGroups.push_back(group);

  // closest hit shader
  group.type             = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
  group.generalShader    = VK_SHADER_UNUSED_KHR;
  group.closestHitShader = eClosestHit;
  m_rtShaderGroups.push_back(group);

  // closest hit shader + Intersection
  group.type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_PROCEDURAL_HIT_GROUP_KHR;
  group.closestHitShader   = eClosestHit2;
  group.intersectionShader = eIntersection;
  m_rtShaderGroups.push_back(group);


  // Push constant: we want to be able to update constants used by the shaders
  VkPushConstantRange pushConstant{VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
                                       | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                                   0, sizeof(PushConstantRay)};


  VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
  pipelineLayoutCreateInfo.pPushConstantRanges    = &pushConstant;

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<VkDescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
  pipelineLayoutCreateInfo.setLayoutCount             = static_cast<uint32_t>(rtDescSetLayouts.size());
  pipelineLayoutCreateInfo.pSetLayouts                = rtDescSetLayouts.data();

  vkCreatePipelineLayout(m_device, &pipelineLayoutCreateInfo, nullptr, &m_rtPipelineLayout);


  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  VkRayTracingPipelineCreateInfoKHR rayPipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
  rayPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());  // Stages are shaders
  rayPipelineInfo.pStages    = stages.data();

  // In this case, m_rtShaderGroups.size() == 4: we have one raygen group,
  // two miss shader groups, and one hit group.
  rayPipelineInfo.groupCount = static_cast<uint32_t>(m_rtShaderGroups.size());
  rayPipelineInfo.pGroups    = m_rtShaderGroups.data();

  // The ray tracing process can shoot rays from the camera, and a shadow ray can be shot from the
  // hit points of the camera rays, hence a recursion level of 2. This number should be kept as low
  // as possible for performance reasons. Even recursive ray tracing should be flattened into a loop
  // in the ray generation to avoid deep recursion.
  rayPipelineInfo.maxPipelineRayRecursionDepth = 2;  // Ray depth
  rayPipelineInfo.layout                       = m_rtPipelineLayout;

  vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &rayPipelineInfo, nullptr, &m_rtPipeline);


  for(auto& s : stages)
    vkDestroyShaderModule(m_device, s.module, nullptr);
}

//--------------------------------------------------------------------------------------------------
// creates the shader binding table (SBT)
// get all shader handles and write them into a SBT buffer
void HelloVulkan::createShaderBindingTable()
{
  uint32_t missCount{2};
  uint32_t hitCount{2};
  // there is always only one raygen, so therefore the 1 here
  auto     handleCount = 1 + missCount + hitCount;
  uint32_t handleSize  = m_rtProperties.shaderGroupHandleSize;

  // the SBT buffer needs to have starting groups to be aligned and handles in the group to be aligned
  uint32_t handleSizeAligned = nvh::align_up(handleSize, m_rtProperties.shaderGroupHandleAlignment);

  m_rgenRegion.stride = nvh::align_up(handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);
  m_rgenRegion.size = m_rgenRegion.stride;  // The size member of pRayGenShaderBindingTable must be equal to its stride member (KHR spec)
  m_missRegion.stride = handleSizeAligned;
  m_missRegion.size   = nvh::align_up(missCount * handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);
  m_hitRegion.stride  = handleSizeAligned;
  m_hitRegion.size    = nvh::align_up(hitCount * handleSizeAligned, m_rtProperties.shaderGroupBaseAlignment);

  // fetch shader group handles
  uint32_t             dataSize = handleCount * handleSize;
  std::vector<uint8_t> handles(dataSize);
  auto result = vkGetRayTracingShaderGroupHandlesKHR(m_device, m_rtPipeline, 0, handleCount, dataSize, handles.data());
  assert(result == VK_SUCCESS);

  // allocate a buffer for storing the SBT
  VkDeviceSize sbtSize = m_rgenRegion.size + m_missRegion.size + m_hitRegion.size + m_callRegion.size;  // we don't use callables, so size is still 0
  m_rtSBTBuffer        = m_alloc.createBuffer(sbtSize,
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                                                  | VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  m_debug.setObjectName(m_rtSBTBuffer.buffer, std::string("SBT"));  // set a name for debugging with NSight

  // find the SBT addresses of each group
  VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, m_rtSBTBuffer.buffer};
  VkDeviceAddress           sbtAddress = vkGetBufferDeviceAddress(m_device, &info);
  m_rgenRegion.deviceAddress           = sbtAddress;
  m_missRegion.deviceAddress           = sbtAddress + m_rgenRegion.size;
  m_hitRegion.deviceAddress            = sbtAddress + m_rgenRegion.size + m_missRegion.size;

  // helper lambda function to retrieve the handle data
  auto getHandle = [&](int i) { return handles.data() + i * handleSize; };

  // map SBT buffer and write in the handles
  auto*    pSBTBuffer = reinterpret_cast<uint8_t*>(m_alloc.map(m_rtSBTBuffer));
  uint8_t* pData{nullptr};
  uint32_t handleIdx{0};

  // Raygen
  pData = pSBTBuffer;
  memcpy(pData, getHandle(handleIdx++), handleSize);

  // Miss
  pData = pSBTBuffer + m_rgenRegion.size;
  for(uint32_t c = 0; c < missCount; c++)
  {
    memcpy(pData, getHandle(handleIdx++), handleSize);
    pData += m_missRegion.stride;
  }

  // Hit
  pData = pSBTBuffer + m_rgenRegion.size + m_missRegion.size;
  for(uint32_t c = 0; c < hitCount; c++)
  {
    memcpy(pData, getHandle(handleIdx++), handleSize);
    pData += m_hitRegion.stride;
  }

  m_alloc.unmap(m_rtSBTBuffer);
  m_alloc.finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// ray tracing the scene
void HelloVulkan::raytrace(const VkCommandBuffer& cmdBuf, const nvmath::vec4f& clearColor)
{
  m_debug.beginLabel(cmdBuf, "Ray trace");
  // init push constant values
  m_pcRay.clearColor     = clearColor;

  std::vector<VkDescriptorSet> descSets{m_rtDescSet, m_descSet};
  vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipeline);
  vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipelineLayout, 0,
                          (uint32_t)descSets.size(), descSets.data(), 0, nullptr);
  vkCmdPushConstants(cmdBuf, m_rtPipelineLayout,
                     VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR
                         | VK_SHADER_STAGE_INTERSECTION_BIT_KHR,
                     0, sizeof(PushConstantRay), &m_pcRay);

  vkCmdTraceRaysKHR(cmdBuf, &m_rgenRegion, &m_missRegion, &m_hitRegion, &m_callRegion, m_size.width, m_size.height, 1);
  m_debug.endLabel(cmdBuf);
}

// stores last write time for various files used as #includes for the shaders
void HelloVulkan::initShaderUpdater()
{
  for(const auto& entry : std::filesystem::directory_iterator(m_shaderSourcePathPrefix))
  {
    std::string path = entry.path().string();
    if(path.substr(path.size() - 5) == ".glsl" || path.substr(path.size() - 2) == ".h")
    {
      m_shaderWriteTimes.insert(std::make_pair(path.substr(m_shaderSourcePathPrefix.size()), entry.last_write_time()));
    }
  }
}

// recompiles a shader via systemcall, for better performance and debugging should be compiled with shaderc library
// TODO: MIGHT NOT WORK ON UNIX SYSTEMS, ONLY TESTED ON WINDOWS
void HelloVulkan::compileShader(std::string path) {
  std::string shaderName = path.substr(m_shaderSourcePathPrefix.size());
  LOGI("Updating shader:  %s \n", shaderName.c_str());

  // this will get the path to the VulkanSDK via the environment variables
  //   -> if the variable "VULKAN_SDK" is not defined, shader recompiling will not work!
  std::string vulkanPath = std::string(getenv("VULKAN_SDK"));

  std::string winInPath = path;

  std::string winOutPath = m_shaderCompilePathPrefix + shaderName + ".spv";

  std::string command = "\"\"" + vulkanPath + "\\bin\\glslangValidator.exe\" -g --target-env vulkan1.2 -o \""
                        + winOutPath + "\" \"" + winInPath + "\"\"";

  // this suppresses the console output from the command (command differs on windows and unix)
  #if defined(_WIN32) || defined(_WIN64)
    command += " > nul";
  #else
    command += " > /dev/null";
  #endif

  system(command.c_str());
}

// Checks if a shader file or an #include file for a shader got updated and recompiles affected shaders
// note: although this method checks for changed shader files first and afterwards for changed #include
//       files, this should in praxis never lead to recompilation of the same shader, as the user would
//       have to change a shader and a #include file in the same frame -> so no noticable overhead
void HelloVulkan::compileChangedShaders() {
  // recompiles every changed shader file
  for(const auto& entry : std::filesystem::directory_iterator(m_shaderCompilePathPrefix))
  {
    // get name of source shader
    std::string shaderSourcePath = entry.path().string().substr(m_shaderCompilePathPrefix.size());
    shaderSourcePath             = m_shaderSourcePathPrefix + shaderSourcePath.erase(shaderSourcePath.size() - 4);

    // check if source shader exists and get its last write time
    std::filesystem::file_time_type lastSourceTime;
    try
    {
      lastSourceTime = std::filesystem::last_write_time(shaderSourcePath);
    }
    catch(...)
    {
      LOGI("Shader source file not found:  %s \n", (shaderSourcePath).c_str());
      continue;
    }

    // recompile shader if it is not up to date with the source file
    if(entry.last_write_time() < lastSourceTime)
    {
      compileShader(shaderSourcePath);
    }
  }

  // recompile certain shaders, when their #include files get updated
  if(compareLastWriteTime("host_device.h"))
  {
    // recompile all affected shaders
   compileShader(m_shaderSourcePathPrefix + "raytrace.rgen");
   compileShader(m_shaderSourcePathPrefix + "raytrace.rmiss");
   compileShader(m_shaderSourcePathPrefix + "raytrace.rchit");
   compileShader(m_shaderSourcePathPrefix + "raytrace2.rchit");
   compileShader(m_shaderSourcePathPrefix + "raytrace.rint");

    // update existing write times
    updateLastWriteTime("host_device.h");
    updateLastWriteTime("raycommon.glsl");
    updateLastWriteTime("blending.glsl");
    updateLastWriteTime("wavefront.glsl");
  }

  if(compareLastWriteTime("raycommon.glsl"))
  {
    // recompile all affected shaders
    compileShader(m_shaderSourcePathPrefix + "raytrace.rgen");
    compileShader(m_shaderSourcePathPrefix + "raytrace.rmiss");
    compileShader(m_shaderSourcePathPrefix + "raytrace.rchit");
    compileShader(m_shaderSourcePathPrefix + "raytrace2.rchit");
    compileShader(m_shaderSourcePathPrefix + "raytrace.rint");

    // update existing write times
    updateLastWriteTime("raycommon.glsl");
    updateLastWriteTime("blending.glsl");
    updateLastWriteTime("wavefront.glsl");
  }

  if(compareLastWriteTime("blending.glsl"))
  {
    // recompile all affected shaders
    compileShader(m_shaderSourcePathPrefix + "raytrace2.rchit");
    compileShader(m_shaderSourcePathPrefix + "raytrace.rint");

    // update existing write time
    updateLastWriteTime("blending.glsl");
  }

  if(compareLastWriteTime("wavefront.glsl"))
  {
    // recompile all affected shaders
    compileShader(m_shaderSourcePathPrefix + "raytrace.rchit");
    compileShader(m_shaderSourcePathPrefix + "raytrace2.rchit");

    // update existing write time
    updateLastWriteTime("wavefront.glsl");
  }
}

// reloads precompiled SPIR-V by deleting the old pipeline and creating a new one
void HelloVulkan::reloadShaders() {
  // wait for every vulkan command to end
  vkDeviceWaitIdle(m_device);
  // destroying old pipeline and SBT
  vkDestroyPipeline(m_device, m_rtPipeline, nullptr);
  vkDestroyPipelineLayout(m_device, m_rtPipelineLayout, nullptr);
  m_alloc.destroy(m_rtSBTBuffer);

  // creating new pipeline and SBT
  createRtPipeline();
  createShaderBindingTable();
}

//--------------------------------------------------------------------------------------------------
// @author Josias
// Creates an axis-aligned bounding box for a triangle that can have a certain amount of displacement
//
Aabb HelloVulkan::createAabbFromTriangle(TriangleObj t)
{
  Aabb aabb;
  // displace in direction of normal
  nvmath::vec3f extA = t.v0.pos + displacementAmount * t.v0.nrm;
  nvmath::vec3f extB = t.v1.pos + displacementAmount * t.v1.nrm;
  nvmath::vec3f extC = t.v2.pos + displacementAmount * t.v2.nrm;

  for(uint32_t i = 0; i < 3; i++)
  {
    aabb.minimum[i] =
        std::min(std::min(std::min(t.v0.pos[i], t.v1.pos[i]), std::min(t.v2.pos[i], extA[i])), std::min(extB[i], extC[i]));
    aabb.maximum[i] =
        std::max(std::max(std::max(t.v0.pos[i], t.v1.pos[i]), std::max(t.v2.pos[i], extA[i])), std::max(extB[i], extC[i]));
  }
  return aabb;
}

bool HelloVulkan::compareLastWriteTime(std::string shaderName)
{
  return std::filesystem::last_write_time(m_shaderSourcePathPrefix + shaderName) > m_shaderWriteTimes.find(shaderName)->second;
}

void HelloVulkan::updateLastWriteTime(std::string shaderName)
{
  m_shaderWriteTimes.find(shaderName)->second = std::filesystem::last_write_time(m_shaderSourcePathPrefix + shaderName);
}


// Creates one mip level above the input texture. Only works for power of 2 quadratic textures. The program WILL crash, if the input
// texture does not exist or if the texture doesn't have the required format. VERY FRAGILE FUNCTION AT THE MOMENT!
void HelloVulkan::createMips(const std::string inputTexture) {
  // 1.  load base texture from file
  std::stringstream o;
  int               texWidth, texHeight, texChannels;
  o << "media/textures/" << inputTexture;
  std::string txtFile = nvh::findFile(o.str(), defaultSearchPaths, true);
  stbi_uc* pixels = stbi_load(txtFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

  // texWidth == texHeight through contract -> only power of 2 textures allowed
  uint32_t mipLevels = nvvk::mipLevels(VkExtent2D{(uint32_t)texWidth, (uint32_t)texWidth});

  // all mip levels added together are at maximum one third of the base texture
  size_t totalSize = ((static_cast<unsigned long long>(texWidth) * texWidth * texChannels) * (1 + 1.0 / 3)) * sizeof(stbi_uc);
  // allocate memory for the whole texture data
  stbi_uc* data = static_cast<stbi_uc*>(malloc(totalSize));

  // copy pixels into data array
  for(int i = 0; i < texWidth * texWidth * 4; i++)
  {
    *(data + i) = *(pixels + i);
  }

  // free the stbi image data as it is no longer needed
  stbi_image_free(pixels);

  // these values are used throughout the iteration
  int      inputSize     = texWidth;
  stbi_uc* inputPointer  = data;
  stbi_uc* outputPointer = data + texWidth * texWidth * texChannels;


  // 2.  calculate mip levels on CPU and store inside a data array
  for(int i = 1; i < mipLevels; i++)
  {
    createSingleMip(inputPointer, outputPointer, inputSize, texChannels);
    //createSingleDebugMip(outputPointer, inputSize, texChannels, std::array<stbi_uc, 4>{0u, 0u, 255u, 255u});
    // recalculate iteration values
    inputSize /= 2;
    inputPointer = outputPointer;
    outputPointer += inputSize * inputSize * texChannels;  // inputSize is now the size of the old mipmap

    // debug output of the mip levels
    std::string path = "X:\\Bibliotheken\\Downloads\\debug\\mip" + std::to_string(i) + ".png";
    stbi_write_png(path.c_str(), inputSize, inputSize, texChannels, inputPointer, inputSize * texChannels);
  }

  //  4. create VkDeviceMemory to store the data array
  VkBuffer stagingBuffer;
  VkDeviceMemory stagingBufferMemory;

  createBuffer(totalSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

  // copy texture data array from CPU to GPU
  void* bufferData;
  vkMapMemory(m_device, stagingBufferMemory, 0, totalSize, 0, &bufferData);
  memcpy(bufferData, data, static_cast<size_t>(totalSize));
  vkUnmapMemory(m_device, stagingBufferMemory);

  free(data);


  // 5.  create VkImageCreateInfo
  VkImageCreateInfo imageInfo{};
  imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  imageInfo.imageType     = VK_IMAGE_TYPE_2D;
  imageInfo.extent.width  = texWidth;
  imageInfo.extent.height = texWidth;
  imageInfo.extent.depth  = 1;
  imageInfo.mipLevels     = mipLevels;
  imageInfo.arrayLayers   = 1;
  imageInfo.format        = VK_FORMAT_R8G8B8A8_SRGB;
  imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
  imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  imageInfo.usage         = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
  imageInfo.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;



  // 6.  create a VkImage via vkCreateImage
  if(vkCreateImage(m_device, &imageInfo, nullptr, &m_textureImage) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create image!");
  }

  VkMemoryRequirements memRequirements;
  vkGetImageMemoryRequirements(m_device, m_textureImage, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize  = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  if(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_textureImageMemory) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to allocate image memory!");
  }
  vkBindImageMemory(m_device, m_textureImage, m_textureImageMemory, 0);

  // copy buffer data into the image (this is internally done for each mip map)
  nvvk::CommandPool cmdBufGet(m_device, m_graphicsQueueIndex);
  VkCommandBuffer   cmdBuf = cmdBufGet.createCommandBuffer();

  transitionImageLayout(cmdBuf, m_textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, mipLevels);
  copyBufferToImage(cmdBuf, stagingBuffer, m_textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texWidth));
  transitionImageLayout(cmdBuf, m_textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, mipLevels);

  cmdBufGet.submitAndWait(cmdBuf);

  VkSamplerCreateInfo samplerCreateInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  samplerCreateInfo.minFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.magFilter  = VK_FILTER_LINEAR;
  samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  samplerCreateInfo.maxLod     = FLT_MAX;

  if(vkCreateSampler(m_device, &samplerCreateInfo, nullptr, &m_textureSampler) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create texture sampler!");
  }

  // creating image view
  m_textureImageView = createImageView(m_textureImage, VK_FORMAT_R8G8B8A8_SRGB, mipLevels);

  // adding image to descriptor set
  m_imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  m_imageInfo.imageView   = m_textureImageView;
  m_imageInfo.sampler     = m_textureSampler;

  nvvk::Texture texture{};
  texture.image = m_textureImage;
  texture.descriptor = m_imageInfo;

  //VkImageViewCreateInfo ivInfo  = nvvk::makeImageViewCreateInfo(image.image, imageCreateInfo);

  // TODO: add correct memHandle to the nvvk::Texture -> only makes problems when destroying the image
  m_textures.push_back(texture);

  // 16. destroy the initial buffer and free its memory
  vkDestroyBuffer(m_device, stagingBuffer, nullptr);
  vkFreeMemory(m_device, stagingBufferMemory, nullptr);
}

// creates a single mip level from input texture
// memory for "output" has to be pre-allocated outside of this function
// inputSize also has to be a positive even number
void HelloVulkan::createSingleMip(stbi_uc* input, stbi_uc* output, int inputSize, int texChannels)
{
  // as only power of 2 and quadratic textures are supported at the moment,
  // it is sufficient to store only the size and not width AND height
  int    mipSize  = inputSize / 2;

  // iteration over the texels from the mip level 1 -> left to right and top to bottom
  for(int i = 0; i < mipSize * mipSize; i++)
  {
    // row and column of mip level 1
    int mipRow = i / mipSize;
    int mipCol = i % mipSize;
    // row and column of mip level 0 -> top left pixel of the desired block of 4
    int row = mipRow * 2;
    int col = mipCol * 2;

    // fetch the 2x2 pixel block to be reduced (index scaled by number of color channels)
    stbi_uc* topLeft     = input + (row * inputSize + col) * texChannels;
    stbi_uc* topRight    = input + (row * inputSize + col + 1) * texChannels;
    stbi_uc* bottomLeft  = input + ((row + 1) * inputSize + col) * texChannels;
    stbi_uc* bottomRight = input + ((row + 1) * inputSize + col + 1) * texChannels;

    // accessing red values for linear blending
    stbi_uc linear = 0.25 * ((*topLeft) + (*topRight) + (*bottomLeft) + (*bottomRight));
    // accessing green values for minimum
    stbi_uc min = std::min(std::min(*(topLeft + 1), *(topRight + 1)), std::min(*(bottomLeft + 1), *(bottomRight + 1)));
    // accessing blue values for maximum
    stbi_uc max = std::max(std::max(*(topLeft + 2), *(topRight + 2)), std::max(*(bottomLeft + 2), *(bottomRight + 2)));

    // for final texture layout, see notes -> (linear, min, max, 255u)
    *(output + (mipRow * mipSize + mipCol) * texChannels)  = linear;
    *(output + (mipRow * mipSize + mipCol) * texChannels + 1) = min;
    *(output + (mipRow * mipSize + mipCol) * texChannels + 2) = max;
    *(output + (mipRow * mipSize + mipCol) * texChannels + 3) = 255u;
  }
}

// creates a single mip level which is only one color
// memory for "output" has to be pre-allocated outside of this function
void HelloVulkan::createSingleDebugMip(stbi_uc* output, int inputSize, int texChannels, std::array<stbi_uc, 4> color)
{
  // as only power of 2 and quadratic textures are supported at the moment,
  // it is sufficient to store only the size and not width AND height
  int mipSize = inputSize / 2;

  // iteration over the texels from the mip level 1 -> left to right and top to bottom
  for(int i = 0; i < mipSize * mipSize; i++)
  {
    // THE FOLLOWING INDICES DO NOT TAKE THE NUMBER OF TEXCHANNELS INTO ACCOUNT!!
    // row and column of mip level 1
    int mipRow = i / mipSize;
    int mipCol = i % mipSize;

    // set the color into every pixel
    //std::cout << "ROW: " << mipRow << std::endl;
    //std::cout << "COL: " << mipCol << std::endl;
    *(output + (mipRow * mipSize + mipCol) * texChannels)             = color[0];
    *(output + (mipRow * mipSize + mipCol) * texChannels + 1)         = color[1];
    *(output + (mipRow * mipSize + mipCol) * texChannels + 2)         = color[2];
    *(output + (mipRow * mipSize + mipCol) * texChannels + 3)         = color[3];
  }
}

void HelloVulkan::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
{
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size        = size;
  bufferInfo.usage       = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  if(vkCreateBuffer(m_device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create buffer!");
  }

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(m_device, buffer, &memRequirements);

  VkMemoryAllocateInfo allocInfo{};
  allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocInfo.allocationSize  = memRequirements.size;
  allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

  if(vkAllocateMemory(m_device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to allocate buffer memory!");
  }

  vkBindBufferMemory(m_device, buffer, bufferMemory, 0);
}

uint32_t HelloVulkan::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
  VkPhysicalDeviceMemoryProperties memProperties;
  vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);

  for(uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
  {
    if((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
    {
      return i;
    }
  }

  throw std::runtime_error("failed to find suitable memory type!");
}

void HelloVulkan::transitionImageLayout(VkCommandBuffer cmdBuf, VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout, int miplevels)
{
  VkImageMemoryBarrier barrier{};
  barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout                       = oldLayout;
  barrier.newLayout                       = newLayout;
  barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.image                           = image;
  barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel   = 0;
  barrier.subresourceRange.levelCount     = miplevels;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount     = 1;

  VkPipelineStageFlags sourceStage;
  VkPipelineStageFlags destinationStage;

  if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
  {
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

    sourceStage      = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
  else if(oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
  {
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    sourceStage      = VK_PIPELINE_STAGE_TRANSFER_BIT;
    destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  }
  else
  {
    throw std::invalid_argument("unsupported layout transition!");
  }

  vkCmdPipelineBarrier(cmdBuf, sourceStage, destinationStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void HelloVulkan::copyBufferToImage(VkCommandBuffer cmdBuf, VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
{
  uint32_t mipLevels = nvvk::mipLevels(VkExtent2D{(uint32_t)width, (uint32_t)height});
  uint32_t mipSize   = width;
  uint64_t offset    = 0;

  for(int i = 0; i < mipLevels; i++)
  {
    VkBufferImageCopy region{};
    region.bufferOffset                    = offset;
    region.bufferRowLength                 = 0;
    region.bufferImageHeight               = 0;
    region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel       = i;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount     = 1;
    region.imageOffset                     = {0, 0, 0};
    region.imageExtent                     = {mipSize, mipSize, 1};

    vkCmdCopyBufferToImage(cmdBuf, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    offset += static_cast<uint64_t>(mipSize) * mipSize * 4;
    mipSize /= 2;
  }
}

VkImageView HelloVulkan::createImageView(VkImage image, VkFormat format, int miplevels)
{
  VkImageViewCreateInfo viewInfo{};
  viewInfo.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  viewInfo.image                           = image;
  viewInfo.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  viewInfo.format                          = format;
  viewInfo.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  viewInfo.subresourceRange.baseMipLevel   = 0;
  viewInfo.subresourceRange.levelCount     = miplevels;
  viewInfo.subresourceRange.baseArrayLayer = 0;
  viewInfo.subresourceRange.layerCount     = 1;

  VkImageView imageView;
  if(vkCreateImageView(m_device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
  {
    throw std::runtime_error("failed to create texture image view!");
  }

  return imageView;
}