# Vulkan Renderer

A rudimentary Vulkan renderer implemented in Rust using Vulkano and Slang.<br>
It features a visibility buffer using Device Generated Commands for a near unlimited amount of pipelines.

## Build Instructions

You can build the project with cargo. Clone the repository and run cargo build or cargo run. For convenience, a helper script ``run.bat`` is provided which will build and run the project in release mode.<br>
You need to install the Vulkan SDK and make sure that its path is added to the ``VULKAN_SDK`` environment variable (this should happen automatically during install of the Vulkan SDK).<br>
It is tested using Vulkan SDK version 1.4.341.1 but anything above 1.4 that has Slang bundled into should work.<br>
Additionally, you need to have <a href=https://github.com/llvm/llvm-project/releases/tag/llvmorg-18.1.8>clang</a> installed and the environment variable ``LIBCLANG_PATH`` must point to the ``bin`` directory of clang.

## Hardware/Software support

Currently only supported on Windows 11. <br>
Requires support for the ``VK_NV_device_generated_commands_compute`` extension which is likely only supported on Nvidia hardware. It is tested on an Nvidia RTX 3080. 