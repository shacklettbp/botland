#pragma once

#include "rt.hpp"
#include "macros.hpp"
#include "global_alloc.hpp"
#include "sync.hpp"

namespace bot {

struct RuntimeState {
  alignas(BOT_CACHE_LINE) u32 allocLock = 0;
  GlobalAlloc globalAlloc = {};

  u32 memPoolReservedBlocks;
  u32 memPoolCommittedBlocks;
  u32 memPoolMinCommittedBlocks;
};

}
