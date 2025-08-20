#pragma once

#include <utility>

namespace bot::gpu_prims {

inline bool isLeader()
{
  return threadIdx.x % 32 == 0;
}

template <typename Fn, int granularity = 1>
void warpLoop(u32 total_num_iters, Fn &&fn)
{
  uint32_t iter = granularity * (threadIdx.x % 32);
  while (iter < total_num_iters) {
#pragma unroll
    for (int i = 0; i < granularity; ++i) {
      fn(iter + i);
    }

    iter += 32 * granularity;
  }
}

template <typename Fn>
void warpLoopSync(uint32_t total_num_iters, Fn &&fn)
{
  uint32_t iter = threadIdx.x % 32;
  bool run = (iter < total_num_iters);

  while (__any_sync(0xFFFF'FFFF, run)) {
    fn(run ? iter : 0xFFFF'FFFF);
    iter += 32;

    run = (iter < total_num_iters);
  }
}

template <typename Fn>
auto leaderExec(Fn &&fn)
{
  if constexpr (std::is_void_v<decltype(fn())>) {
    if (threadIdx.x % 32 == 0) {
      fn();
    }

    __syncwarp();
  } else {
    decltype(fn()) ret;

    if (threadIdx.x % 32 == 0) {
      ret = fn();
    }

    ret = __shfl_sync(0xFFFF'FFFF, ret, 0);

    return ret;
  }
}

}
