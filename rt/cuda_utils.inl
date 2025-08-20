namespace bot {

void *allocGPU(u64 num_bytes)
{
  void *ptr;
  REQ_CUDA(cudaMalloc(&ptr, num_bytes));

  return ptr;
}

void deallocGPU(void *ptr)
{
  REQ_CUDA(cudaFree(ptr));
}

void *allocStaging(u64 num_bytes)
{
  void *ptr;
  REQ_CUDA(cudaHostAlloc(&ptr, num_bytes,
                         cudaHostAllocMapped | cudaHostAllocWriteCombined));

  return ptr;
}

void *allocReadback(u64 num_bytes)
{
  void *ptr;
  REQ_CUDA(cudaHostAlloc(&ptr, num_bytes,
                         cudaHostAllocMapped));

  return ptr;
}

void deallocStaging(void *ptr)
{
  REQ_CUDA(cudaFreeHost(ptr));
}

void deallocReadback(void *ptr)
{
  REQ_CUDA(cudaFreeHost(ptr));
}

void cpyCPUToGPU(cudaStream_t strm, void *gpu, void *cpu, u64 num_bytes)
{
  REQ_CUDA(cudaMemcpyAsync(gpu, cpu, num_bytes, cudaMemcpyHostToDevice,
                           strm));
}

void cpyGPUToCPU(cudaStream_t strm, void *cpu, void *gpu, u64 num_bytes)
{
  REQ_CUDA(cudaMemcpyAsync(cpu, gpu, num_bytes, cudaMemcpyDeviceToHost,
                           strm));
}

cudaStream_t makeCUStream()
{
  cudaStream_t strm;
  REQ_CUDA(cudaStreamCreate(&strm));

  return strm;
}

void checkCuda(cudaError_t res, const char *file,
               int line, const char *funcname) noexcept
{
  if (res != cudaSuccess) {
    cudaRuntimeError(res, file, line, funcname);
  }
}

void checkCuDrv(CUresult res, const char *file,
                int line, const char *funcname) noexcept
{
  if (res != CUDA_SUCCESS) {
    cuDrvError(res, file, line, funcname);
  }
}

}
