#pragma once

#ifdef BOT_CUDA_SUPPORT
#include <rt/rt_gpu.hpp>
#endif

#ifndef BOT_GPU
#include <iostream>
#endif

#include "host_print.hpp"

namespace bot {

// This invokes host print for guaranteed logging
struct Log {
  template <typename ...ArgsT>
  static void log(const char *str, ArgsT &&...args);
};

}

#include "log.inl"
