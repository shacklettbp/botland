#pragma once

#include <gas/gas.hpp>
#include <gas/gas_ui.hpp>
#include <gas/gas_imgui.hpp>
#include <scene/scene.hpp>
#include <scene/import.hpp>

#include <sim/render.hpp>
#include <sim/backend.hpp>

#include <vector>

namespace bot {
using namespace gas;

struct UIControl {
  enum class Flag : u32 {
    None         = 0,
    RawMouseMode = 1 << 0,
    EnableIME    = 1 << 1,
    DisableIME   = 1 << 2,
  };
  using enum Flag;

  Flag flags = None;
  Vector2 imePos = {};
  float imeLineHeight = 0.f;
};

struct RenderMesh {
  u32 vertexOffset = 0;
  u32 indexOffset;
  u32 numTriangles;
  u32 materialIndex;
};

struct RenderObject {
  u32 meshOffset = 0;
  u32 numMeshes = 0;
};

struct Scene {
  Buffer geoBuffer;
  std::vector<RenderMesh> renderMeshes;
  std::vector<RenderObject> renderObjects;
};

struct OrbitCam {
  Vector3 target = Vector3::zero();
  float heading = 0.f;
  float azimuth = 0.25f * PI;
  float zoom = 10.f;

  float fov = 60.f;

  // These are calculated based on target, heading, azimuth & zoom each frame
  Vector3 position = {};
  Vector3 right = {};
  Vector3 up = {};
  Vector3 fwd = {};
};

struct FrameInput {
  Buffer globalDataBuffer = {};
  ParamBlock globalDataPB = {};
};

// Resolution dependent
struct RenderFrame {
  Texture depth = {};
  Texture hdr = {};
  RasterPass hdrPass = {};
  RasterPass finalPass = {};
  ParamBlock tonemapPB = {};
};

struct FrameState {
  FrameInput input = {};
  RenderFrame render = {};
};

struct Backend;

struct Viz {
  static constexpr inline i32 NUM_FRAMES_IN_FLIGHT = 2;

  GPULib * gpuLib = nullptr;
  GPUDevice * gpu = nullptr;
  GPUQueue mainQueue = {};

  Swapchain swapchain = {};
  TextureFormat swapchainFormat = TextureFormat::None;

  u32 windowWidth = 0;
  u32 windowHeight = 0;

  TextureFormat hdrRenderFormat = {};

  Sampler bilinearRepeatSampler = {};

  ParamBlockType globalDataPBType = {};
  ParamBlockType tonemapPBType = {};

  RasterPassInterface hdrPassInterface = {};
  RasterPassInterface finalPassInterface = {};

  std::array<FrameState, NUM_FRAMES_IN_FLIGHT> frames = {};
  i32 curFrameIdx = 0;

  ParamBlockType globalParamBlockType = {};
  Buffer globalPassDataBuffer = {};
  ParamBlock globalParamBlock = {};

  RasterShader objectShader = {};
  RasterShader tonemapShader = {};

  CommandEncoder frameEnc = {};

  OrbitCam cam = {};

  Scene scene = {};

  BridgeData<RenderBridge> *renderBridge = nullptr;
  Backend *backend = nullptr;

  void init(GPULib *gpu_lib, GPUDevice *gpu_in, Surface surface,
            Backend *backend, ImportedRenderAssets *assets,
            BridgeData<RenderBridge> *render_bridge);
  void shutdown();

  void resize(Surface surface);

  UIControl updateUI(UserInput &input, UserInputEvents &events,
                     const char *text_input, float ui_scale, float delta_t);
  void render();

private:
  inline void initSwapchain(Surface surface);
  inline void cleanupSwapchain();

  inline void initSamplers();
  inline void cleanupSamplers();

  inline void initParamBlockTypes();
  inline void cleanupParamBlockTypes();
  inline void initPassInterfaces();
  inline void cleanupPassInterfaces();

  inline void initFrameInputs();
  inline void cleanupFrameInputs();

  inline void initRenderFrames();
  inline void cleanupRenderFrames();

  inline void loadShaders();
  inline void cleanupShaders();

  inline void buildImguiWidgets();

  inline void renderGeo(FrameState &frame, RasterPassEncoder &enc);
};

inline UIControl::Flag & operator|=(UIControl::Flag &a, UIControl::Flag b);
inline UIControl::Flag operator|(UIControl::Flag a, UIControl::Flag b);
inline UIControl::Flag & operator&=(UIControl::Flag &a, UIControl::Flag b);
inline UIControl::Flag operator&(UIControl::Flag a, UIControl::Flag b);

}

#include "viz.inl"
