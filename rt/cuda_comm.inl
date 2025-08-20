namespace bot {

#ifndef BOT_GPU
template <typename PayloadT>
template <typename ProcessFnT>
CudaCommCPU<PayloadT>::CudaCommCPU(CUdevice cu_gpu, const ProcessFnT &fn)
  : channel_(createChannel(cu_gpu)),
    thread_(
      [&]() {
        using namespace std::chrono_literals;
        using cuda::std::memory_order_acquire;
        using cuda::std::memory_order_relaxed;
        const auto reset_duration = 1ms;
        const auto max_duration = 1s;

        auto cur_duration = reset_duration;

        while (true) {
          auto signal = channel_->signal.load(memory_order_acquire);
          if (signal == 0) {
            std::this_thread::sleep_for(cur_duration);
            cur_duration *= 2;
            if (cur_duration > max_duration) {
              cur_duration = max_duration;
            }
            continue;
          } else if (signal == -1) {
            break;
          }

          fn(&channel_->payload);

          channel_->signal.store(0, memory_order_relaxed);
          cur_duration = reset_duration;
        }
      })
{
}

template <typename PayloadT>
void CudaCommCPU<PayloadT>::terminate()
{
  channel_->signal.store(-1, cuda::std::memory_order_release);
  thread_.join();

  REQ_CU(cuMemFree((CUdeviceptr)channel_));
}

template <typename PayloadT>
CudaCommChannel<PayloadT> * CudaCommCPU<PayloadT>::createChannel(
    CUdevice cu_gpu)
{
  CUdeviceptr channel_devptr;
  REQ_CU(cuMemAllocManaged(&channel_devptr,
        sizeof(CudaCommChannel<PayloadT>),
        CU_MEM_ATTACH_GLOBAL));

  REQ_CU(cuMemAdvise((CUdeviceptr)channel_devptr, 
        sizeof(CudaCommChannel<PayloadT>),
        CU_MEM_ADVISE_SET_READ_MOSTLY, 0));
  REQ_CU(cuMemAdvise((CUdeviceptr)channel_devptr,
        sizeof(CudaCommChannel<PayloadT>),
        CU_MEM_ADVISE_SET_ACCESSED_BY, CU_DEVICE_CPU));

  REQ_CU(cuMemAdvise(channel_devptr, sizeof(CudaCommChannel<PayloadT>),
        CU_MEM_ADVISE_SET_ACCESSED_BY, cu_gpu));

  auto ptr = (CudaCommChannel<PayloadT> *)channel_devptr;
  ptr->signal.store(0, cuda::std::memory_order_release);

  return ptr;
}

template <typename PayloadT>
void * CudaCommCPU<PayloadT>::getChannelPtr()
{
  return (void *)channel_;
}
#endif

template <typename PayloadT>
CudaCommGPU<PayloadT>::CudaCommGPU(CudaCommChannel<PayloadT> *channel)
  : channel_(channel),
    device_lock_(0)
{
}

template <typename PayloadT>
void CudaCommGPU<PayloadT>::submit(const PayloadT &payload)
{
#ifdef BOT_GPU
  using cuda::std::memory_order_relaxed;
  using cuda::std::memory_order_release;

  spinLock(&device_lock_);

  channel_->payload = payload;
  channel_->signal.store(1, memory_order_release);

  while (channel_->signal.load(memory_order_relaxed) == 1) {
    __nanosleep(0);
  }

  spinUnlock(&device_lock_);
#else
  (void)payload;
#endif
}

template <typename PayloadT>
template <typename FillPayloadT>
void CudaCommGPU<PayloadT>::submit(FillPayloadT fn)
{
#ifdef BOT_GPU
  using cuda::std::memory_order_relaxed;
  using cuda::std::memory_order_release;

  spinLock(&device_lock_);

  // channel_->payload = payload;
  fn(&channel_->payload);

  channel_->signal.store(1, memory_order_release);

  while (channel_->signal.load(memory_order_relaxed) == 1) {
    __nanosleep(0);
  }

  spinUnlock(&device_lock_);
#else
  (void)fn;
#endif
}

}
