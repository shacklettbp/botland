#pragma once

#ifndef BOT_GPU
#include <thread>
#include <chrono>
#include "cuda_utils.hpp"
#endif

#include "sync.hpp"

#include <cuda/atomic>


namespace bot {

template <typename PayloadT>
struct CudaCommChannel {
  PayloadT payload;

  cuda::atomic<int32_t, 
    cuda::thread_scope_system> signal;
};

#ifndef BOT_GPU
template <typename PayloadT>
class CudaCommCPU {
public:
  // ProcessFnT should take as parameter: PayloadT *.
  template <typename ProcessFnT>
  inline CudaCommCPU(CUdevice cu_gpu, const ProcessFnT &fn);
  virtual inline ~CudaCommCPU() = default;

  inline void terminate();
  // virtual inline ~CudaCommCPU();

  CudaCommCPU(CudaCommCPU &&o) = delete;

  inline void * getChannelPtr();

private:
  inline CudaCommChannel<PayloadT> * createChannel(CUdevice cu_gpu);

  CudaCommChannel<PayloadT> *channel_;
  std::thread thread_;
};
#endif

template <typename PayloadT>
class CudaCommGPU {
public:
  CudaCommGPU(CudaCommChannel<PayloadT> *channel);

  void submit(const PayloadT &payload);

  template <typename FillPayloadT>
  void submit(FillPayloadT fn);

private:
  CudaCommChannel<PayloadT> *channel_;
  u32 device_lock_;
};

}

#include "rt/cuda_comm.inl"
