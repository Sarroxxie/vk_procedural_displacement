# Procedural Displacement (Vulkan)
 The goal is to create tesselation free procedural displacement. This implementation was created alongside a bachelor thesis (see the .pdf file). Forgive me for sloppy code, the program had to be ready in time and the coding style and performance suffered from it.

## Dependencies
You will need several dependencies to compile and run this program.

### Ray Tracing Compatible Graphics Card
Not all graphics cards support vulkan raytracing. You will have to see for yourself if your card support the necessary extensions. When running the program, it will throw an error if an extension is missing and also show which one in the console output. An unofficial website so check if your GPU supports the `VK_KHR_acceleration_structure` extension can be found [here](https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_KHR_acceleration_structure&platform=all).

### Graphics Card Driver
While your graphics card driver only has to support at least Vulkan 1.3, it is recommended to just update to the latest version. NVIDIA drivers can be found on [NVIDIAs official website](https://www.nvidia.com/download/index.aspx) and AMD drivers can be found on [AMDs official website](https://www.amd.com/en/support).

### CMake
A version of [CMake](https://cmake.org/download/) is required to build the project.

### nvpro_core
NVIDIAs [nvpro_core](https://github.com/nvpro-samples/nvpro_core) is the framework on which this program is built. You will need to clone it from Github. The `nvpro_core` folder should be on the same level as the `vk_procedural_displacement` folder.

### VulkanSDK
This program runs on the graphics card using the Vulkan API, so a version of the [VulkanSDK](https://vulkan.lunarg.com/sdk/home) is required. Which specific version is needed depends on the version of `nvpro_core`, but a version higher than 1.3.221.0 is recommended.

-------

## Setup

* install all dependencies listed above
* create a `build` directory inside of the `vk_procedural_displacement` folder
* open `cmd` -> navigate to `build` directory -> type `cmake..`
* open the Visual Studio solution named `vk_procedural_displacement` that can be found inside the `build` directory
* in Visual Studio, right click on `vk_ray_tracing__displacement_KHR` -> *Set as Startup Project*
* now you can start running the program via the built in compiler
