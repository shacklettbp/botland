#include <iostream>
#include "host_print.hpp"

namespace bot {

#if defined(BOT_CUDA_SUPPORT) || defined(BOT_GPU)

#ifdef BOT_GPU
HostPrintGPU::HostPrintGPU()
  : CudaCommGPU<HostPrintPayload>::CudaCommGPU(nullptr)
{
}

HostPrintGPU::HostPrintGPU(
    CudaCommChannel<HostPrintPayload> *channel)
  : CudaCommGPU<HostPrintPayload>::CudaCommGPU(channel)
{
}

void HostPrintGPU::logSubmit(
    const char *str, void **ptrs, HostPrintPayload::FmtType *types,
    int32_t num_args)
{
  using cuda::std::memory_order_relaxed;
  using cuda::std::memory_order_release;

  submit([&](HostPrintPayload *payload) {
    int32_t cur_offset = 0;
    do {
      payload->buffer[cur_offset] = str[cur_offset];
    } while (str[cur_offset++] != '\0');

    for (int i = 0; i < num_args; i++) {
      HostPrintPayload::FmtType type = types[i];

      int32_t arg_size;
      switch (type) {
        case HostPrintPayload::FmtType::I32: {
          arg_size = sizeof(int32_t);
        }; break;
        case HostPrintPayload::FmtType::U32: {
          arg_size = sizeof(uint32_t);
        }; break;
        case HostPrintPayload::FmtType::I64: {
          arg_size = sizeof(int64_t);
        }; break;
        case HostPrintPayload::FmtType::U64: {
          arg_size = sizeof(uint64_t);
        }; break;
        case HostPrintPayload::FmtType::Float: {
          arg_size = sizeof(float);
        }; break;
        case HostPrintPayload::FmtType::Ptr: {
          arg_size = sizeof(void *);
        }; break;
        default: 
                           __builtin_unreachable();
      }

      memcpy(&payload->buffer[cur_offset],
          ptrs[i], arg_size);
      cur_offset += arg_size;
      assert(cur_offset < HostPrintPayload::MAX_BYTES);

      payload->args[i] = type;
    }
    payload->numArgs = num_args;
  });
}

#else
static void handleHostPrintPayload(HostPrintPayload *channel)
{
  std::string_view print_str = channel->buffer;
  size_t buffer_offset = print_str.length() + 1;
  size_t str_offset = 0;

  CountT cur_arg = 0;

  while (str_offset < print_str.size()) {
    size_t pos = print_str.find("{}", str_offset);
    if (pos == print_str.npos) {
      std::cout << print_str.substr(str_offset);
      break;
    }

    std::cout << print_str.substr(str_offset, pos - str_offset);

    assert(cur_arg < channel->numArgs);
    HostPrintPayload::HostPrintPayload::FmtType type = channel->args[cur_arg];
    switch (type) {
    case HostPrintPayload::FmtType::I32: {
      int32_t v;
      memcpy(&v, &channel->buffer[buffer_offset],
          sizeof(int32_t));
      buffer_offset += sizeof(uint32_t);
      std::cout << v;
    } break;
    case HostPrintPayload::FmtType::U32: {
      uint32_t v;
      memcpy(&v, &channel->buffer[buffer_offset],
          sizeof(uint32_t));
      buffer_offset += sizeof(uint32_t);
      std::cout << v;
    } break;
    case HostPrintPayload::FmtType::I64: {
      int64_t v;
      memcpy(&v, &channel->buffer[buffer_offset],
          sizeof(int64_t));
      buffer_offset += sizeof(int64_t);
      std::cout << v;
    } break;
    case HostPrintPayload::FmtType::U64: {
      uint64_t v;
      memcpy(&v, &channel->buffer[buffer_offset],
          sizeof(uint64_t));
      buffer_offset += sizeof(uint64_t);
      std::cout << v;
    } break;
    case HostPrintPayload::FmtType::Float: {
      float v;
      memcpy(&v, &channel->buffer[buffer_offset],
          sizeof(float));
      buffer_offset += sizeof(float);
      std::cout << v;
    } break;
    case HostPrintPayload::FmtType::Ptr: {
      void *v;
      memcpy(&v, &channel->buffer[buffer_offset],
          sizeof(void *));
      buffer_offset += sizeof(void *);
      std::cout << v;
    } break;
    }

    cur_arg++;
    str_offset = pos + 2;
  }

  std::cout << std::flush;
}

HostPrintCPU::HostPrintCPU(CUdevice cu_gpu)
  : CudaCommCPU<HostPrintPayload>::CudaCommCPU(
      cu_gpu, 
      handleHostPrintPayload)
{
}
#endif
#endif

}
