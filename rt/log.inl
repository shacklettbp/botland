namespace bot {

template <typename ...ArgsT>
void Log::log(const char *str, ArgsT &&...args)
{
#ifdef BOT_GPU
  GPURuntimeConsts &consts = gpuConsts();
  consts.hostPrint->log(str, std::forward<ArgsT>(args)...);
#else
  auto translate_type = [](auto *ptr) {
    using T = std::decay_t<decltype(*ptr)>;

    if constexpr (std::is_same_v<T, int32_t>) {
      return HostPrintPayload::FmtType::I32;
    } else if constexpr (std::is_same_v<T, uint32_t>) {
      return HostPrintPayload::FmtType::U32;
    } else if constexpr (std::is_same_v<T, int64_t>) {
      return HostPrintPayload::FmtType::I64;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
      return HostPrintPayload::FmtType::U64;
    } else if constexpr (std::is_same_v<T, float>) {
      return HostPrintPayload::FmtType::Float;
    } else if constexpr (std::is_pointer_v<T>) {
      return HostPrintPayload::FmtType::Ptr;
    } else {
      static_assert(!std::is_same_v<T, T>);
    }
  };

  std::array<void *, sizeof...(ArgsT)> ptrs {
    (void *)&args
      ...
  };

  std::array<HostPrintPayload::FmtType, ptrs.size()> types {
    translate_type(&args)
      ...
  };

  HostPrintPayload payload = 
    [&](const char *str, void **ptrs, HostPrintPayload::FmtType *types,
        int32_t num_args) {
    HostPrintPayload payload = {};

    int32_t cur_offset = 0;
    do {
      payload.buffer[cur_offset] = str[cur_offset];
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

      memcpy(&payload.buffer[cur_offset],
          ptrs[i], arg_size);
      cur_offset += arg_size;
      assert(cur_offset < HostPrintPayload::MAX_BYTES);

      payload.args[i] = type;
    }
    payload.numArgs = num_args;

    return payload;
  } (str, ptrs.data(), types.data(),
      (int32_t)sizeof...(ArgsT));

  std::string_view print_str = payload.buffer;
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

    assert(cur_arg < payload.numArgs);
    HostPrintPayload::HostPrintPayload::FmtType type = payload.args[cur_arg];
    switch (type) {
    case HostPrintPayload::FmtType::I32: {
      int32_t v;
      memcpy(&v, &payload.buffer[buffer_offset],
          sizeof(int32_t));
      buffer_offset += sizeof(uint32_t);
      std::cout << v;
    } break;
    case HostPrintPayload::FmtType::U32: {
      uint32_t v;
      memcpy(&v, &payload.buffer[buffer_offset],
          sizeof(uint32_t));
      buffer_offset += sizeof(uint32_t);
      std::cout << v;
    } break;
    case HostPrintPayload::FmtType::I64: {
      int64_t v;
      memcpy(&v, &payload.buffer[buffer_offset],
          sizeof(int64_t));
      buffer_offset += sizeof(int64_t);
      std::cout << v;
    } break;
    case HostPrintPayload::FmtType::U64: {
      uint64_t v;
      memcpy(&v, &payload.buffer[buffer_offset],
          sizeof(uint64_t));
      buffer_offset += sizeof(uint64_t);
      std::cout << v;
    } break;
    case HostPrintPayload::FmtType::Float: {
      float v;
      memcpy(&v, &payload.buffer[buffer_offset],
          sizeof(float));
      buffer_offset += sizeof(float);
      std::cout << v;
    } break;
    case HostPrintPayload::FmtType::Ptr: {
      void *v;
      memcpy(&v, &payload.buffer[buffer_offset],
          sizeof(void *));
      buffer_offset += sizeof(void *);
      std::cout << v;
    } break;
    }

    cur_arg++;
    str_offset = pos + 2;
  }

  std::cout << std::flush;
#endif
}

}
