#include "rt.hpp"
#include "rt_gpu.hpp"
#include "rt_gpu_impl.hpp"

namespace bot {

static constexpr inline bool FATAL_ERROR_ON_OOM = true;

MemHandle globalAlloc(RTStateHandle state_hdl, u32 num_blocks)
{
  RuntimeState *state = getRuntimeState(state_hdl);

  spinLock(&state->allocLock);
  BOT_DEFER(spinUnlock(&state->allocLock));

  u32 blk = state->globalAlloc.alloc((char *)state, num_blocks);
  if (blk == GLOBAL_ALLOC_OOM) [[unlikely]] {
    if constexpr (FATAL_ERROR_ON_OOM) {
      FATAL("GPU OOM %lu", num_blocks * GLOBAL_ALLOC_BLOCK_SIZE);
    } else {
      return { GLOBAL_ALLOC_OOM, 0 };
    }
  }

  return { blk, num_blocks };
}

void globalDealloc(RTStateHandle state_hdl, MemHandle mem)
{
  RuntimeState *state = getRuntimeState(state_hdl);

  if (mem.hdl == GLOBAL_ALLOC_OOM) [[unlikely]] {
    return;
  }

  spinLock(&state->allocLock);
  BOT_DEFER(spinUnlock(&state->allocLock));

  GlobalAlloc::DeallocStatus dealloc_status = 
      state->globalAlloc.dealloc((char *)state, mem.hdl, mem.numBlks);
  (void)dealloc_status;
}

}
