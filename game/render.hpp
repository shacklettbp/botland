#pragma once

#include <rt/rt.hpp>
#include <rt/store.hpp>
#include <scene/render.hpp>

#include "bridge.hpp"

namespace bot {
  
struct RenderBridge {
  BridgeRef<PerspectiveCameraData> cameras;
  BridgeRef<InstanceData> instances;
  BridgeRef<LightData> lights;

  u32 numCameras;
  u32 numInstances;
  u32 numLights;

  u32 worldID;
};

#define RENDER_INSTANCE_FIELDS(F) \
  F(InstanceData, instanceData)
BOT_PERSISTENT_STORE(RenderInstance, 128, RENDER_INSTANCE_FIELDS);
#undef RENDER_INSTANCE_FIELDS

#define RENDER_CAMERA_FIELDS(F) \
  F(PerspectiveCameraData, cameraData)
BOT_PERSISTENT_STORE(RenderCamera, 128, RENDER_CAMERA_FIELDS);
#undef RENDER_CAMERA_FIELDS

#define RENDER_LIGHT_FIELDS(F) \
  F(LightData, lightData)
BOT_PERSISTENT_STORE(RenderLight, 32, RENDER_LIGHT_FIELDS);
#undef RENDER_LIGHT_FIELDS

class SimRT;

struct RenderWorld {
  RenderInstanceStore instanceStore;
  RenderCameraStore cameraStore;
  RenderLightStore lightStore;

  RenderBridge *bridge;

  AtomicI32 numInstances = 0;
  AtomicI32 numCameras = 0;
  AtomicI32 numLights = 0;

  void init(Runtime &rt,
            MemArena &persistent_arena,
            MemArena &tmp_arena,
            RenderBridge *render_bridge);

  RenderInstanceID makeInstance();
  RenderCameraID makeCamera();
  RenderLightID makeLight();
};

void renderTasks(SimRT &rt, TaskExec &exec);

}
