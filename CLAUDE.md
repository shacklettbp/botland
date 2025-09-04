# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

This project uses CMake with Ninja as the build system. The main build commands are:

```bash
# Configure and build (from project root)
cmake -S. -B build -G Ninja
ninja -C build
```

## Architecture Overview

This is a C++ game/simulation engine with the following key components:

### Core Modules

- **rt/**: Runtime system providing core functionality including memory management, math utilities, CUDA support, and cross-platform abstractions
- **game/**: Game simulation logic including physics (BVH, GJK collision), geometry, import systems (glTF, OBJ, STL), and grid-based simulation
- **viz/**: Visualization and rendering using Slang shaders with support for multiple graphics APIs
- **gas/**: Graphics abstraction system supporting Vulkan, Metal, and WebGPU through Dawn
- **net/**: Basic networking functionality with socket support
- **deps/**: External dependencies including Slang shader compiler, Dawn WebGPU implementation, SDL3, and imgui

### Key Technologies

- **Language**: C++20 with some C11 components
- **Graphics**: Slang shading language with multi-API support (Vulkan/Metal/WebGPU)
- **Compute**: Optional CUDA support for GPU acceleration
- **Build**: CMake 3.24+ with custom toolchain support
- **Platform**: Cross-platform (Windows, macOS, Linux, Web via Emscripten)

### Important Constants

- Grid-based simulation with `GRID_SIZE = 8` and `TEAM_SIZE = 4`
- Default unit stats: `DEFAULT_HP = 10`, `DEFAULT_SPEED = 10`
- Memory management with 64KB allocation blocks
- Support for up to 8 color attachments and 4 bind groups per shader

## Development Notes

- The project can be built for web using Emscripten (`BOT_WEB` option)
- CUDA support is optional and controlled by `BOT_CUDA_SUPPORT`
- Uses a custom memory allocation system with arenas and stack allocators
- Shader compilation happens at build time, with generated metadata stored in `build/shaders_metadata/`
- No formal test suite - testing is primarily through the executable targets