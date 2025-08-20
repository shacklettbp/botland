#pragma once

#include "rt.hpp"
#include "global_alloc.hpp"

namespace bot {
#ifndef BOT_GPU
namespace device {
#endif

struct RuntimeState {
  alignas(BOT_CACHE_LINE) u32 allocLock = 0;
  GlobalAlloc globalAlloc = {};
};

#ifndef BOT_GPU
}
#endif
}
