#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include "macros.hpp"
#include "types.hpp"

namespace bot {

inline void *allocGPU(u64 num_bytes);

inline void deallocGPU(void *ptr);

inline void *allocStaging(u64 num_bytes);

inline void *allocReadback(u64 num_bytes);

inline void deallocStaging(void *ptr);
inline void deallocReadback(void *ptr);

inline void cpyCPUToGPU(cudaStream_t strm, void *gpu, void *cpu, u64 num_bytes);

inline void cpyGPUToCPU(cudaStream_t strm, void *cpu, void *gpu, u64 num_bytes);

inline cudaStream_t makeCUStream();

[[noreturn]] void cudaRuntimeError(
    cudaError_t err, const char *file,
    int line, const char *funcname) noexcept;
[[noreturn]] void cuDrvError(
    CUresult err, const char *file,
    int line, const char *funcname) noexcept;

inline void checkCuda(cudaError_t res, const char *file,
                      int line, const char *funcname) noexcept;
inline void checkCuDrv(CUresult res, const char *file,
                       int line, const char *funcname) noexcept;

CUcontext initCUContext(int gpu_id);
void releaseCUContext(int gpu_id, CUcontext ctx);

}

#define ERR_CUDA(err) ::bot::cudaError((err), __FILE__, __LINE__,\
                                      BOT_COMPILER_FUNCTION_NAME)
#define ERR_CU(err) ::bot::cuDrvError((err), __FILE__, __LINE__,\
                                      BOT_COMPILER_FUNCTION_NAME)

#define REQ_CUDA(expr) ::bot::checkCuda((expr), __FILE__, __LINE__,\
                                       BOT_COMPILER_FUNCTION_NAME)
#define REQ_CU(expr) ::bot::checkCuDrv((expr), __FILE__, __LINE__,\
                                      BOT_COMPILER_FUNCTION_NAME)

#include "cuda_utils.inl"
