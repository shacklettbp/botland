#include <memory>
#include "backend.hpp"

#include "rt/os.hpp"
#include <filesystem>


#ifdef BOT_CUDA_SUPPORT
#include "rt/cuda_utils.hpp"
#include "rt/cuda_mgr.hpp"
#include "rt/rt_gpu.hpp"
#include "rt/cuda_comm.hpp"
#endif

#include <rt/host_print.hpp>

namespace bot {

#ifdef BOT_CUDA_SUPPORT
struct CUDABackend {
  CUDAManager mgr = {};
  CUgraphExec stepGraphExec;
};
#endif

struct Backend {
  RTStateHandle rtStateHandle = {};
  MemArena globalArena = {};

  BackendConfig cfg = {};

#ifdef BOT_CUDA_SUPPORT
  CUDABackend cuda;
#endif

  Sim *sim = nullptr;
};

#ifdef BOT_CUDA_SUPPORT

static CUDABackend initCUDABackend(Runtime &rt,
                                   const BackendConfig &cfg,
                                   const SimConfig &sim_cfg,
                                   MemArena &global_arena,
                                   Sim **sim_out)
{
  CUDABackend cuda;

  rt.resultArena() = rt.tmpArena();
  rt.tmpArena() = {};

  u64 code_path_len = 0;
  const char *code_path_str = getCodePath(
      rt, []() {}, &code_path_len);

  rt.tmpArena() = rt.resultArena();
  rt.resultArena() = {};

  if (!code_path_str) {
    FATAL("Could not find bot cubin files");
  }

  std::filesystem::path code_path(code_path_str);
  std::filesystem::path cubin_path = code_path.parent_path() / "bot-sim-gpu.cubin";

  cuda.mgr.init(rt, cfg.cuda, cubin_path.c_str(), global_arena);

  SimConfig *gpu_sim_cfg = (SimConfig *)allocGPU(sizeof(sim_cfg));
  cpyCPUToGPU(0, gpu_sim_cfg, (void *)&sim_cfg, sizeof(sim_cfg));

  Sim **gpu_sim = (Sim **)allocGPU(sizeof(Sim));

  Sim *sim = nullptr;
  cuda.mgr.launchOneOff(rt, "botInitSim", { .numBlocksX = 1, .blockSizeX = 1 },
                        gpu_sim_cfg, 
                        gpu_sim);
  cuda.mgr.sync();

  cpyGPUToCPU(0, &sim, gpu_sim, sizeof(Sim *));

  cuda.mgr.launchOneOff(rt, "botCreateWorlds", cuda.mgr.defaultPersistentCfg,
                        sim, gpu_sim_cfg);
  cuda.mgr.sync();

  {
    CUgraph step_graph;
    REQ_CU(cuGraphCreate(&step_graph, 0));

    cuda.mgr.addLaunchGraphNode(rt, step_graph, "botStepWorlds",
      cuda.mgr.defaultPersistentCfg, sim);

    REQ_CU(cuGraphInstantiate(&cuda.stepGraphExec, step_graph, 0));
    REQ_CU(cuGraphDestroy(step_graph));
  }

  *sim_out = sim;
  return cuda;
}

static void shutdownCUDABackend(CUDABackend &cuda)
{
  REQ_CU(cuGraphExecDestroy(cuda.stepGraphExec));
  cuda.mgr.shutdown();
}

#endif

static Sim * initCPUBackend(Runtime &rt,
                           const BackendConfig &cfg,
                           const SimConfig &sim_cfg)
{
  (void)cfg;

  Sim *sim = nullptr;
  botInitSim(rt.stateHandle(), 0, &sim_cfg, &sim);
  botCreateWorlds(rt.stateHandle(), 0, sim, &sim_cfg);

  return sim;
}

Backend * backendInit(BackendConfig cfg)
{
  RTStateHandle rt_state_hdl = createRuntimeState({});
  Runtime rt(rt_state_hdl, 0);

  Backend *be = nullptr;
  {
    MemArena global_arena = {};

    be = rt.arenaAlloc<Backend>(global_arena);
    new (be) Backend {};
    be->rtStateHandle = rt_state_hdl;
    be->globalArena = global_arena;
  }

  be->cfg = cfg;

  if (be->cfg.cuda.gpuID != -1) {
#ifdef BOT_CUDA_SUPPORT
    be->cuda = initCUDABackend(
        rt, be->cfg, cfg.sim, 
        be->globalArena, &be->sim);
#else
    FATAL("Trying to use CUDA be when build wasn't "
          "compiled with CUDA support");
#endif
  } else {
    be->sim = initCPUBackend(rt, be->cfg, cfg.sim);
  }

  return be;
}

void backendSyncStepWorlds(Backend *backend)
{
  if (backend->cfg.cuda.gpuID == -1) {
    botStepWorlds(backend->rtStateHandle, 0, backend->sim);
  } else {
#ifdef BOT_CUDA_SUPPORT
    REQ_CU(cuGraphLaunch(backend->cuda.stepGraphExec, backend->cuda.mgr.strm));
    backend->cuda.mgr.sync();
#else
    FATAL("Build wasn't compiled with CUDA support");
#endif
  }
}

RuntimeState * backendRuntimeState(Backend *backend)
{
  return getRuntimeState(backend->rtStateHandle);
}

RTStateHandle backendRTStateHandle(Backend *backend)
{
  return backend->rtStateHandle;
}

void backendShutdown(Backend *be)
{
  RTStateHandle rt_state_hdl = be->rtStateHandle;

#ifdef BOT_CUDA_SUPPORT
  if (be->cuda.mgr.gpuID != -1) {
    shutdownCUDABackend(be->cuda);
  }
#endif

  MemArena global_arena = be->globalArena;
  arenaRelease(rt_state_hdl, global_arena);
  destroyRuntimeState(rt_state_hdl);
}

int backendGPUID(Backend *be)
{
  return be->cfg.cuda.gpuID;
}

}
