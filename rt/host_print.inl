#include <cassert>

namespace bot {

#ifdef BOT_GPU
template <typename... Args>
void HostPrintGPU::log(const char *str, Args && ...args)
{
  __threadfence_system();

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

  std::array<void *, sizeof...(Args)> ptrs {
    (void *)&args
      ...
  };

  std::array<HostPrintPayload::FmtType, ptrs.size()> types {
    translate_type(&args)
      ...
  };

  logSubmit(str, ptrs.data(), types.data(),
      (int32_t)sizeof...(Args));

  __threadfence_system();
}
#endif

}
