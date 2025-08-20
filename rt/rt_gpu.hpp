#pragma once

#include "types.hpp"
#include <rt/host_print.hpp>

namespace bot {

namespace GPU_DEFAULT_SM_CFG {
  constexpr inline u32 NUM_THREADS_PER_BLOCK = 256;
  constexpr inline u32 NUM_BLOCKS_PER_SM = 3;
  constexpr inline u32 NUM_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / 32;
  constexpr inline u32 NUM_SMEM_BYTES_PER_BLOCK = 32768;

  constexpr inline u32 NUM_SMEM_BYTES_PER_WARP =
    NUM_SMEM_BYTES_PER_BLOCK / NUM_WARPS_PER_BLOCK; 
}

#ifndef BOT_GPU
namespace device {
#endif

struct alignas(128) SMemAligned {};

struct RuntimeState;
struct GPURuntimeConsts {
  void *gpuBuffer;
  i32 numSMs;
  HostPrintGPU *hostPrint;
};

struct GPUThreadInfo {
  i32 warpID;
  i32 laneID;
};

#ifndef BOT_GPU
}
#endif
}

#ifdef BOT_GPU

extern "C" {
  extern __constant__ ::bot::GPURuntimeConsts botGPUConsts;
  extern __shared__ ::bot::SMemAligned botGPUSMemBuffer[];
}

namespace bot {

constexpr inline u32 NUM_RT_RESERVED_SMEM_BYTES = 8;

inline GPURuntimeConsts & gpuConsts() { return botGPUConsts; }
inline char * gpuSMem() { return (char *)botGPUSMemBuffer; }
inline char * gpuSMemUser() { return (char *)botGPUSMemBuffer + 
                                     NUM_RT_RESERVED_SMEM_BYTES; }

inline GPUThreadInfo gpuThreadInfo();

}
#endif

#include "rt_gpu.inl"
