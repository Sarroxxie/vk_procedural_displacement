# Procedural Displacement (Vulkan)
 The goal is to create tesselation free procedural displacement. This implementation was created alongside a bachelor thesis (see the .pdf file). Forgive me for sloppy code, the program had to be ready in time and the coding style and performance suffered from it in the final stages.

## Dependencies
You will need several dependencies to compile and run this program.

### Ray Tracing Compatible Graphics Card
Not all graphics cards support vulkan raytracing. You will have to see for yourself if your card supports the necessary extensions. When running the program, it will throw an error if an extension is missing and also show which one in the console output. An unofficial website to check if your GPU supports the `VK_KHR_acceleration_structure` extension can be found [here](https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_KHR_acceleration_structure&platform=all).

### Graphics Card Driver
While your graphics card driver only has to support at least Vulkan 1.3, it is recommended to just update to the latest version. NVIDIA drivers can be found on [NVIDIAs official website](https://www.nvidia.com/download/index.aspx) and AMD drivers can be found on [AMDs official website](https://www.amd.com/en/support).

### CMake
A version of [CMake](https://cmake.org/download/) is required to build the project.

### nvpro_core
NVIDIAs [nvpro_core](https://github.com/nvpro-samples/nvpro_core) is the framework on which this program is built. However, this program was developed on a [previous version](https://github.com/Sarroxxie/nvpro_core), which is included as a submodule. There is no need to clone it manually as it is included as a submodule.

### VulkanSDK
This program runs on the graphics card using the Vulkan API, so a version of the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) is required. Most recently tested with version 1.4.335 of the Vulkan SDK.

-------

## Setup

* install all dependencies listed above
* open `cmd` -> navigate to the repository directory and call `git submodule init`, followed by `git submodule update` after the first command is finished
* open `cmd` -> navigate to the `libs\nvpro_core` directory and call `git submodule init`, followed by `git submodule update` after the first command is finished
* create a `build` directory inside of the repository directory
* open `cmd` -> navigate to `build` directory (using `cd build`) -> type `cmake ..`
* open the Visual Studio solution named `vk_procedural_displacement` that can be found inside the `build` directory
* in Visual Studio, right click on `vk_ray_tracing__displacement_KHR` -> *Set as Startup Project*
* now you can start running the program via the built-in compiler
