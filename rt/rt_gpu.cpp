#include "rt.hpp"
#include "rt_gpu.hpp"
#include "rt_gpu_impl.hpp"

extern "C" {
  __constant__ ::bot::GPURuntimeConsts botGPUConsts;
}

namespace bot {

extern "C" __global__ void botGPUInitRuntimeState(u32 pool_num_blocks,
                                                 char *gpu_buffer,
                                                 int num_sms,
                                                 HostPrintGPU *host_print_addr,
                                                 CudaCommChannel<HostPrintPayload> *hp_channel,
                                                 GPURuntimeConsts *consts_readback)
{
  u32 init_used_blocks = 
    (u32)divideRoundUp(sizeof(RuntimeState), (size_t)GLOBAL_ALLOC_BLOCK_SIZE);
  chk(init_used_blocks < pool_num_blocks);

  RuntimeState * state = new (gpu_buffer) RuntimeState {};

  state->globalAlloc.init(
    gpu_buffer, init_used_blocks, pool_num_blocks);

  consts_readback->gpuBuffer = gpu_buffer;
  consts_readback->numSMs = num_sms;
  consts_readback->hostPrint = host_print_addr;
  new (consts_readback->hostPrint) HostPrintGPU(hp_channel);
}

extern "C" __global__ void botGPUGlobalAllocFromCPU(
  u32 num_blocks, MemHandle *mem_out)
{
  MemHandle mem = globalAlloc({botGPUConsts.gpuBuffer}, num_blocks);
  *mem_out = mem;
}

extern "C" __global__ void botGPUGlobalDeallocFromCPU(
  MemHandle mem)
{
  globalDealloc({botGPUConsts.gpuBuffer}, mem);
}

}
