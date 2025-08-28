#include <rt/rt.hpp>
#include <rt/types.hpp>
#include <filesystem>

#include "backend.hpp"
#include "import.hpp"

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
    .sim = SimConfig {
      .numActiveWorlds = num_worlds,
      .maxNumAgentsPerWorld = 1,
    },
  });
  Runtime rt = Runtime(backendRTStateHandle(backend), 0);

  for (int i = 0; i < 1000; i++) {
    backendSyncStepWorlds(backend);
  }
  
  backendShutdown(backend);

  return 0;
}
