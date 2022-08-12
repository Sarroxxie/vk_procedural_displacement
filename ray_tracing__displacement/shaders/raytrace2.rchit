#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "raycommon.glsl"
#include "wavefront.glsl"
#include "blending.glsl"

hitAttributeEXT intersectionPayload intPayload;

// clang-format off
layout(location = 0) rayPayloadInEXT hitPayload prd;
layout(location = 1) rayPayloadEXT bool isShadowed;

layout(buffer_reference, scalar) buffer Vertices {Vertex v[]; }; // Positions of an object
layout(buffer_reference, scalar) buffer Indices {uint i[]; }; // Triangle indices
layout(buffer_reference, scalar) buffer Materials {WaveFrontMaterial m[]; }; // Array of all materials on an object
layout(buffer_reference, scalar) buffer MatIndices {int i[]; }; // Material ID for each triangle

layout(set = 0, binding = eTlas) uniform accelerationStructureEXT topLevelAS;
layout(set = 1, binding = eDispObjDescs, scalar) buffer DispObjDesc_ { DispObjDesc i[]; } dispObjDesc;
layout(set = 1, binding = eTextures) uniform sampler2D textureSamplers[];

layout(push_constant) uniform _PushConstantRay { PushConstantRay pcRay; };
// clang-format on

void main()
{
  // Object data
  DispObjDesc    objResource = dispObjDesc.i[gl_InstanceCustomIndexEXT];
  MatIndices matIndices  = MatIndices(objResource.materialIndexAddress);
  Materials  materials   = Materials(objResource.materialAddress);

  vec3 worldPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
  vec3 worldNrm = intPayload.worldNrm;

  // Vector toward the light
  vec3  L;
  float lightIntensity = pcRay.lightIntensity;
  float lightDistance  = 100000.0;
  // Point light
  if(pcRay.lightType == 0)
  {
    vec3 lDir      = pcRay.lightPosition - worldPos;
    lightDistance  = length(lDir);
    lightIntensity = pcRay.lightIntensity / (lightDistance * lightDistance);
    L              = normalize(lDir);
  }
  else  // Directional light
  {
    L = normalize(pcRay.lightPosition);
  }

  // Material of the object
  int               matIdx = matIndices.i[gl_PrimitiveID];
  WaveFrontMaterial mat    = materials.m[matIdx];

  // Diffuse
  vec3  diffuse     = computeDiffuse(mat, L, worldNrm);
  if(mat.diffTextureID >= 0)
  {
    // TODO: grabbing mat.dispTextureID for debugging purposes (needs mat.diffTextureID later)
    uint txtId = mat.dispTextureID + dispObjDesc.i[gl_InstanceCustomIndexEXT].txtOffset;
    vec2 texCoord = intPayload.texCoord;
    diffuse = texture(textureSamplers[nonuniformEXT(txtId)], texCoord).xyz;
    diffuse = proceduralTilingAndBlending(texCoord, textureSamplers[nonuniformEXT(txtId)], pcRay.blendingOffset);
  }
  vec3  specular    = vec3(0);
  float attenuation = 0.3;

  // TODO: when shadowray casting is fixed for the displaced surface, uncomment this code and remove the 2 lines below it
  /*
  // Tracing shadow ray only if the light is visible from the surface
  if(dot(worldNrm, L) > 0)
  {
    float tMin   = 0.001;
    float tMax   = lightDistance;
    vec3  origin = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    vec3  rayDir = L;
    uint  flags  = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT;
    isShadowed   = true;
    traceRayEXT(topLevelAS,  // acceleration structure
                flags,       // rayFlags
                0xFF,        // cullMask
                0,           // sbtRecordOffset
                0,           // sbtRecordStride
                1,           // missIndex
                origin,      // ray origin
                tMin,        // ray min range
                rayDir,      // ray direction
                tMax,        // ray max range
                1            // payload (location = 1)
    );

    if(isShadowed)
    {
      attenuation = 0.3;
    }
    else
    {
      attenuation = 1;
      // Specular
      specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, worldNrm);
    }
  }*/

  attenuation = 1;
  specular = computeSpecular(mat, gl_WorldRayDirectionEXT, L, worldNrm);

  prd.hitValue = vec3(lightIntensity * attenuation * (diffuse + specular));
}