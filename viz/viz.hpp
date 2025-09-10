#pragma once

#include <gas/gas.hpp>
#include <gas/gas_ui.hpp>
#include <gas/gas_imgui.hpp>

#include "game/sim.hpp"
#include "font.hpp"

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
  Vector3 target = { i32(GRID_SIZE / 2) - 0.5f, i32(GRID_SIZE / 2) - 0.5f, 0.f };
  float heading = 0.5f * PI;
  float azimuth = 0.4f * PI;
  float zoom = 8.f;

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

struct Materials {
  // Main grid / board material data
  Texture boardTexture = {};
  ParamBlockType boardPBType = {};
  ParamBlock boardPB = {};
  RasterShader boardShader = {};

  // Units
  RasterShader unitsShader = {};
  
  // Generic location effects
  RasterShader genericLocationEffectShader = {};
  
  // Health bars
  RasterShader healthBarShader = {};
  
  // Text rendering
  RasterShader nameShader = {};
  ParamBlockType namePBType = {};
  ParamBlock namePB = {};
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

  RTStateHandle rtStateHdl = {};
  Sim *sim = nullptr;

  i32 curVizActiveWorld = 0;

  GPULib * gpuLib = nullptr;
  GPUDevice * gpu = nullptr;
  GPUQueue mainQueue = {};

  Swapchain swapchain = {};
  TextureFormat swapchainFormat = TextureFormat::None;

  u32 windowWidth = 0;
  u32 windowHeight = 0;

  TextureFormat hdrRenderFormat = {};

  Sampler bilinearRepeatSampler = {};
  Sampler nearestRepeatSampler = {};

  ParamBlockType globalDataPBType = {};
  ParamBlockType tonemapPBType = {};

  RasterPassInterface hdrPassInterface = {};
  RasterPassInterface finalPassInterface = {};

  RasterShader tonemapShader = {};

  Materials materials = {};
  FontAtlas fontAtlas = {};

  std::array<FrameState, NUM_FRAMES_IN_FLIGHT> frames = {};
  i32 curFrameIdx = 0;

  ParamBlock globalParamBlock = {};

  CommandEncoder frameEnc = {};

  OrbitCam cam = {};

  Scene scene = {};
  
  int numWorldResets = 0;

  // Unit selection state
  GridPos selectedGridPos = { -1, -1 };
  UnitID selectedUnit = UnitID::none();

  void init(RTStateHandle rt_state_hdl, Sim *sim_state,
            GPULib *gpu_lib, GPUDevice *gpu_in, Surface surface);
  void shutdown();

  void resize(SimRT &rt, Surface surface);

  UIControl runUI(SimRT &rt, UserInput &input, UserInputEvents &events,
                  const char *text_input, float ui_scale, float delta_t);
  void render(SimRT &rt);

private:
  inline void initSwapchain(Surface surface);
  inline void cleanupSwapchain();

  inline void initSamplers();
  inline void cleanupSamplers();

  inline void initParamBlockTypes();
  inline void cleanupParamBlockTypes();
  inline void initPassInterfaces();
  inline void cleanupPassInterfaces();

  inline void initMaterials(Runtime &rt);
  inline void cleanupMaterials();

  inline void initFrameInputs();
  inline void cleanupFrameInputs();

  inline void initRenderFrames();
  inline void cleanupRenderFrames();

  inline void loadGlobalShaders();
  inline void cleanupGlobalShaders();

  inline void buildImguiWidgets(float ui_scale);

  inline void renderBoard(
    SimRT &rt, FrameState &frame, RasterPassEncoder &enc);
  
  inline void renderGenericLocationEffects(
    SimRT &rt, FrameState &frame, RasterPassEncoder &enc);

  inline void renderUnits(
    SimRT &rt, FrameState &frame, RasterPassEncoder &enc);
  
  inline void renderWalls(
    SimRT &rt, FrameState &frame, RasterPassEncoder &enc);
  
  inline GridPos screenToGridPos(Vector2 screenPos);
};

inline UIControl::Flag & operator|=(UIControl::Flag &a, UIControl::Flag b);
inline UIControl::Flag operator|(UIControl::Flag a, UIControl::Flag b);
inline UIControl::Flag & operator&=(UIControl::Flag &a, UIControl::Flag b);
inline UIControl::Flag operator&(UIControl::Flag a, UIControl::Flag b);

}

#include "viz.inl"
