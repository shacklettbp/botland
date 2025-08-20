#include "cuda_mgr.hpp"

#include "os.hpp"
#include "rt_gpu.hpp"

namespace bot {

void CUDAManager::init(Runtime &rt, const CUDAConfig &cfg,
                       const char *cubin_path,
                       MemArena &global_arena)
{
  chk(cfg.gpuID != -1);
  chk(cfg.memPoolSizeMB > 0);

  ArenaRegion tmp_region = rt.beginTmpRegion();
  BOT_DEFER(rt.endTmpRegion(tmp_region));

  gpuID = cfg.gpuID;
  ctx = initCUContext(cfg.gpuID);
  strm = makeCUStream();

  CUdevice cu_gpu;
  REQ_CU(cuCtxGetDevice(&cu_gpu));

  {
    REQ_CU(cuDeviceGetAttribute(
      &numSMs, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cu_gpu));
  }

  defaultPersistentCfg = {
    .numBlocksX = numSMs * GPU_DEFAULT_SM_CFG::NUM_BLOCKS_PER_SM,
    .blockSizeX = GPU_DEFAULT_SM_CFG::NUM_THREADS_PER_BLOCK,
    .numSMemBytes = GPU_DEFAULT_SM_CFG::NUM_SMEM_BYTES_PER_BLOCK,
  };

  {
    ArenaRegion mod_region = rt.beginTmpRegion();
    BOT_DEFER(rt.endTmpRegion(mod_region));

    u64 num_bytes;
    char * data = readFile(rt, rt.tmpArena(), cubin_path, &num_bytes);
        
    if (!data) {
      FATAL("Failed to read bot cubin at %s", cubin_path);
    }

    REQ_CU(cuModuleLoadData(&mod, data));
  }

  u64 mem_pool_num_bytes = (u64)cfg.memPoolSizeMB * 1024 * 1024;
  u32 mem_pool_num_blks = u32(mem_pool_num_bytes / GLOBAL_ALLOC_BLOCK_SIZE);
  chk((u64)mem_pool_num_blks * GLOBAL_ALLOC_BLOCK_SIZE == mem_pool_num_bytes);

  gpuBuffer = (char *)allocGPU(mem_pool_num_bytes);

  hostPrint = (HostPrintCPU *)
    rt.arenaAlloc<HostPrintCPU>(global_arena);
  new (hostPrint) HostPrintCPU(cu_gpu);

  device::GPURuntimeConsts *consts_readback = 
      (device::GPURuntimeConsts *)allocReadback(sizeof(device::GPURuntimeConsts));

  HostPrintGPU *host_print_gpu_addr =
      (HostPrintGPU *)allocGPU(sizeof(HostPrintGPU));

  launchOneOff(rt, "botGPUInitRuntimeState", { .numBlocksX = 1, .blockSizeX = 1 },
               mem_pool_num_blks, 
               gpuBuffer,
               numSMs,
               host_print_gpu_addr,
               hostPrint->getChannelPtr(),
               consts_readback);
  sync();

  CUdeviceptr bot_gpu_consts_addr;
  size_t bot_gpu_consts_size;
  REQ_CU(cuModuleGetGlobal(&bot_gpu_consts_addr, &bot_gpu_consts_size,
                           mod, "botGPUConsts"));
  REQ_CU(cuMemcpyHtoDAsync(
    bot_gpu_consts_addr, consts_readback, bot_gpu_consts_size, strm));

  sync();
  deallocReadback(consts_readback);

  globalAllocFromCPUFn = findFn("botGPUGlobalAllocFromCPU");
  globalDeallocFromCPUFn = findFn("botGPUGlobalDeallocFromCPU");
}

void CUDAManager::shutdown()
{
  hostPrint->terminate();

  deallocGPU(gpuBuffer);
  REQ_CU(cuModuleUnload(mod));
  REQ_CUDA(cudaStreamDestroy(strm));
  releaseCUContext(gpuID, ctx);
}

CUfunction CUDAManager::findFn(const char *name)
{
  CUfunction fn;
  REQ_CU(cuModuleGetFunction(&fn, mod, name));
  return fn;
}

void CUDAManager::launch(
  CUfunction f, CUDALaunchConfig cfg, void **args)
{
  chk(cfg.numBlocksX > 0);
  chk(cfg.numBlocksY > 0);
  chk(cfg.numBlocksZ > 0);

  chk(cfg.blockSizeX > 0);
  chk(cfg.blockSizeY > 0);
  chk(cfg.blockSizeZ > 0);

  REQ_CU(cuLaunchKernel(f, cfg.numBlocksX, cfg.numBlocksY, cfg.numBlocksZ, 
                        cfg.blockSizeX, cfg.blockSizeY, cfg.blockSizeZ,
                        cfg.numSMemBytes, strm, nullptr, args));
}

void CUDAManager::sync()
{
  REQ_CUDA(cudaStreamSynchronize(strm));
}

MemHandle CUDAManager::allocGPUMemoryFromCPU(Runtime &rt, u64 num_bytes)
{
  ArenaRegion tmp_region = rt.beginTmpRegion();
  BOT_DEFER(rt.endTmpRegion(tmp_region));

  u64 num_blocks = divideRoundUp(num_bytes, (u64)GLOBAL_ALLOC_BLOCK_SIZE);

  MemHandle mem_handle;
  launch(globalAllocFromCPUFn, { .numBlocksX = 1, .blockSizeX = 1 },
         packArgs(rt, rt.tmpArena(), num_blocks, &mem_handle));
  sync();

  return mem_handle;
}

void CUDAManager::deallocGPUMemoryFromCPU(Runtime &rt, MemHandle mem)
{
  ArenaRegion tmp_region = rt.beginTmpRegion();
  BOT_DEFER(rt.endTmpRegion(tmp_region));

  launch(globalAllocFromCPUFn, { .numBlocksX = 1, .blockSizeX = 1 },
         packArgs(rt, rt.tmpArena(), mem));
  sync();
}

}
