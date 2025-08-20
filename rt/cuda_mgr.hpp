#pragma once

#include "cuda_utils.hpp"
#include "cuda_cfg.hpp"

#include "rt.hpp"

#include "host_print.hpp"

namespace bot {

struct CUDALaunchConfig {
  u32 numBlocksX = 0;
  u32 numBlocksY = 1;
  u32 numBlocksZ = 1;
  u32 blockSizeX = 0;
  u32 blockSizeY = 1;
  u32 blockSizeZ = 1;
  u32 numSMemBytes = 0;
};

struct CUDAManager {
  int gpuID = -1;
  CUcontext ctx = {};
  cudaStream_t strm = {};
  int numSMs = 0;

  CUDALaunchConfig defaultPersistentCfg = {};

  CUmodule mod = {};
  char * gpuBuffer = nullptr;

  CUfunction globalAllocFromCPUFn = {};
  CUfunction globalDeallocFromCPUFn = {};

  HostPrintCPU *hostPrint = nullptr;

  void init(Runtime &rt, const CUDAConfig &cfg,
            const char *cubin_path,
            MemArena &global_arena);
  void shutdown();

  template <typename... Ts>
  void ** packArgs(Runtime &rt, MemArena &arena, Ts &&...args);

  CUfunction findFn(const char *name);

  void launch(CUfunction f, CUDALaunchConfig cfg, void **args);

  template <typename... Ts>
  void launchOneOff(Runtime &rt, const char *func_name, CUDALaunchConfig cfg,
                    Ts &&...args);

  template <typename... Ts>
  CUgraphNode addLaunchGraphNode(Runtime &rt, CUgraph graph,
    const char *func_name, CUDALaunchConfig cfg, Ts &&...args);

  void sync();

  MemHandle allocGPUMemoryFromCPU(Runtime &rt, u64 num_bytes);
  void deallocGPUMemoryFromCPU(Runtime &rt, MemHandle mem);
};

}

#include "cuda_mgr.inl"
