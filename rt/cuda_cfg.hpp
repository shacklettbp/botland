#pragma once

#include <rt/types.hpp>

namespace bot {

struct CUDAConfig {
  int gpuID = -1;
  u32 memPoolSizeMB = 0;
};

}
