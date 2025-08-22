#include <rt/rt.hpp>
#include <rt/types.hpp>
#include <sim/backend.hpp>
#include <scene/import.hpp>
#include <filesystem>

int main(int argc, const char *argv[])
{
  using namespace bot;
  namespace fs = std::filesystem;

  using namespace bot;

  int gpu_id = -1;
  int num_worlds = 1;

  if (argc != 3) {
    printf("./headless [cpu|gpu] [# worlds]\n");
    return -1;
  } else {
    const char *backend_type = argv[1];
    const char *num_worlds_str = argv[2];

    if (!strcmp(backend_type, "cpu")) {
      gpu_id = -1;
    } else if (!strcmp(backend_type, "gpu")) {
      gpu_id = 0;
    } else {
      printf("./headless [cpu|gpu] [# worlds]\n");
      return -1;
    }

    num_worlds = std::stoi(num_worlds_str);
  }

  (void)argc;
  (void)argv;

  Backend *backend = backendInit({
    .rt = {},
    .cuda = CUDAConfig {
      .gpuID = gpu_id,
      .memPoolSizeMB = 128,
    },
  });
  Runtime rt = Runtime(backendRTStateHandle(backend), 0);

  PhysicsAssetImporter physics_importer;
  { // Load physics assets
    physics_importer.importAsset(
      (fs::path(BOT_DATA_DIR) / "cube_collision.obj").string());
  }

  PhysicsAssetProcessor physics_processor(
      physics_importer.getImportedAssets());

  BridgeData<ObjectManager> physics_bridge = 
    physics_processor.process(backend);

  auto render_bridge = backendBridgeData<RenderBridge>(backend);

  constexpr u32 MAX_NUM_CAMERAS = 4;
  constexpr u32 MAX_NUM_INSTANCES = 1024;
  constexpr u32 MAX_NUM_LIGHTS = 1;

  { // Setup simulation parameters / shared state between host and sim
    auto cameras = backendBridgeBuffer<PerspectiveCameraData>(
        backend, MAX_NUM_CAMERAS);
    auto instances = backendBridgeBuffer<InstanceData>(
        backend, MAX_NUM_INSTANCES);
    auto lights = backendBridgeBuffer<LightData>(
        backend, MAX_NUM_LIGHTS);

    RenderBridge bridge = {
      .cameras = cameras.detach(),
      .instances = instances.detach(),
      .lights = lights.detach(),
      .numCameras = 0,
      .numInstances = 0,
      .numLights = 0,
      .worldID = 0,
    };

    render_bridge.commit(bridge);
  }

  backendStart(backend, SimConfig {
    .numActiveWorlds = num_worlds,
    .numActionsPerAgent = 4,
    .numDOFSPerAgent = 6,
    .maxNumAgentsPerWorld = 1,
    .renderBridge = render_bridge.get(),
    .objManager = physics_bridge.get(),
  });

  for (int i = 0; i < 1000; i++) {
    backendSyncStepWorlds(backend);
  }
  
  backendShutdown(backend);

  return 0;
}
