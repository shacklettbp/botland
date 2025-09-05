# Shader Architecture Documentation

## Overview
The project uses Slang as the shading language with support for multiple graphics APIs (Vulkan, Metal, WebGPU) through the GAS (Graphics Abstraction System).

## Shader Structure

### Data Flow
1. **Host Data Structures** (`shader_host.slang`)
   - Shared between C++ and shaders
   - Uses `#ifdef BOT_SHADER_HOST_CPP_INCLUDE` for C++ compatibility
   - Key structures:
     - `RenderVertex`: Vertex data (position, normal, UV)
     - `ViewData`: Camera transformation and projection data
     - `GlobalPassData`: Contains view and lighting data
     - `UnitsPerDraw`: Per-instance data for unit rendering
     - `BoardDrawData`: Grid size for board rendering

2. **Parameter Blocks**
   - `ParameterBlock<T>`: GPU-side constant buffers
   - Bound via `setParamBlock()` in render passes
   - Examples:
     - `ParameterBlock<GlobalPassData> global`: Camera/view data
     - `ParameterBlock<UnitsPerDraw> perDraw`: Per-draw instance data

3. **Vertex/Fragment Pipeline**
   - Vertex shader marked with `[shader("vertex")]`
   - Fragment shader marked with `[shader("fragment")]`
   - Data passed via interpolated structures (e.g., `V2F`)

## Shader Registration

### CMakeLists.txt
```cmake
gas_add_shaders(TARGET shader-target-name
  SHADER_ENUM ShaderIDEnum
  SHADER_CLASS ShaderClassName
  CPP_NAMESPACE bot
  SHADERS
    ShaderName shader_file.slang
)
```

### C++ Integration
1. Shaders compiled at build time to bytecode
2. Loaded via generated header files (e.g., `bot-viz-material-shaders.hpp`)
3. Created using `gpu->createRasterShader()` with:
   - Bytecode reference
   - Entry points (vertMain/fragMain)
   - Render pass interface
   - Parameter block types
   - Vertex format
   - Raster configuration

## Utility Functions (`utils.slang`)

### Coordinate Transformations
- `worldToClip()`: Transform world space to clip space
- `clipToWorld()`: Inverse transformation
- `objectPositionToWorld()`: Object to world transformation
- `objectNormalToWorld()`: Normal transformation

### Procedural Geometry
- `cubeVert()`: Generate cube vertices procedurally
- `fullscreenVS()`: Generate fullscreen triangle

### Camera Utilities
- `cameraPosition()`: Extract camera position from ViewData
- `viewDirection()`: Get camera forward vector

## Rendering Pipeline

### Pass Structure
1. **HDR Pass** (`hdrPassInterface`)
   - Renders to floating-point texture
   - Depth testing enabled
   - Used for main scene geometry

2. **Final Pass** (`finalPassInterface`)
   - Renders to swapchain
   - Tonemap from HDR to SDR
   - UI overlay (ImGui)

### Draw Flow
1. Update global data (camera, lights)
2. Begin render pass
3. Set shader and parameter blocks
4. Issue draw calls with per-draw data
5. End render pass

## Billboard/Camera-Aligned Rendering

For camera-aligned elements (like health bars):
1. Use camera right/up vectors from `ViewData.camTxfm`
2. Construct billboard transformation in vertex shader
3. Position in world space, orient to face camera

## Performance Considerations
- Per-draw data passed via `drawData()` for efficiency
- Instancing via vertex/index offsets
- Minimal state changes between draws
- Procedural geometry to reduce buffer uploads