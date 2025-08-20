#include "cuda_utils.hpp"
#include "err.hpp"

namespace bot {

[[noreturn]] void cudaRuntimeError(
  cudaError_t err, const char *file,
  int line, const char *funcname) noexcept
{
  fatal(file, line, funcname, "%s", cudaGetErrorString(err));
}

[[noreturn]] void cuDrvError(
  CUresult err, const char *file,
  int line, const char *funcname) noexcept
{
  const char *name, *desc;
  cuGetErrorName(err, &name);
  cuGetErrorString(err, &desc);
  fatal(file, line, funcname, "%s: %s", name, desc);
}

CUcontext initCUContext(int gpu_id)
{
  REQ_CUDA(cudaSetDevice(gpu_id));
  REQ_CUDA(cudaFree(nullptr));
  CUdevice cu_dev;
  REQ_CU(cuDeviceGet(&cu_dev, gpu_id));
  CUcontext cu_ctx;
  REQ_CU(cuDevicePrimaryCtxRetain(&cu_ctx, cu_dev));
  REQ_CU(cuCtxSetCurrent(cu_ctx));
  
  return cu_ctx;
}

void releaseCUContext(int gpu_id, CUcontext ctx)
{
  (void)ctx;

  CUdevice cu_dev;
  REQ_CU(cuDeviceGet(&cu_dev, gpu_id));
  REQ_CU(cuDevicePrimaryCtxRelease(cu_dev));
}


}
