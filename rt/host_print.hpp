#pragma once

#if defined(BOT_CUDA_SUPPORT) || defined(BOT_GPU)
#include "cuda_comm.hpp"
#endif

namespace bot {

struct HostPrintPayload {
  static inline constexpr int32_t MAX_ARGS = 64;
  static inline constexpr int32_t MAX_BYTES = 4096;

  enum class FmtType : uint32_t {
    I32,
    U32,
    I64,
    U64,
    Float,
    Ptr,
  };

  char buffer[MAX_BYTES];
  FmtType args[MAX_ARGS];
  int32_t numArgs;
};

#if defined(BOT_CUDA_SUPPORT) || defined(BOT_GPU)
class HostPrintGPU : CudaCommGPU<HostPrintPayload> {
public:
  HostPrintGPU();
  HostPrintGPU(CudaCommChannel<HostPrintPayload> *channel);

  void init(CudaCommChannel<HostPrintPayload> *channel);

  template <typename ...ArgsT>
  void log(const char *str, ArgsT &&...args);

private:
  void logSubmit(const char *str, void **ptrs,
                 HostPrintPayload::FmtType *types,
                 int32_t num_args);
};

#ifndef BOT_GPU
class HostPrintCPU : public CudaCommCPU<HostPrintPayload> {
public:
  HostPrintCPU(CUdevice cu_gpu);
};
#endif

#endif

}

#include "host_print.inl"
