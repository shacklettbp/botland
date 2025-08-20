#include "macros.hpp"
#include "err.hpp"

#include <cstdlib>
#include <cstdio>
#include <new>

namespace bot {
namespace {

inline void * osAlloc(size_t num_bytes, size_t alignment)
{
  void *ptr;
  if (alignment <= alignof(void *)) {
    ptr = std::malloc(num_bytes);
  } else {
#if defined(_LIBCPP_VERSION)
    ptr = std::aligned_alloc(alignment, num_bytes);
#elif defined(BOT_CXX_MSVC)
    ptr = _aligned_malloc(num_bytes, alignment);
#else
    STATIC_UNIMPLEMENTED();
#endif
  }

  if (ptr == nullptr) [[unlikely]] {
    FATAL("OOM: %lu\n", num_bytes);
  }

  return ptr;
}

inline void osDealloc(void *ptr, size_t alignment)
{
#if defined(_LIBCPP_VERSION)
  (void)alignment;
  std::free(ptr);
#elif defined(BOT_CXX_MSVC)
  if (alignment <= alignof(void *)) {
    std::free(ptr);
  } else {
    _aligned_free(ptr);
  }
#else
  STATIC_UNIMPLEMENTED();
#endif
}

}
}

// bot-libcxx is compiled without operator new and delete,
// because libc++'s static hermetic mode marks operator new and delete
// as hidden symbols. Unfortunately, this breaks ASAN's (and our own) ability
// to export operator new and operator delete outside of the shared library
// executable. Therefore we disable operator new and delete in libcxx and
// libcxxabi and must provide them here.

// Unaligned versions

#ifdef BOT_WINDOWS
#define BOT_NEWDEL_VIS
#else
#define BOT_NEWDEL_VIS BOT_EXPORT
#endif

BOT_NEWDEL_VIS void * operator new(size_t num_bytes)
{
  return ::bot::osAlloc(num_bytes, alignof(void *));
}

BOT_NEWDEL_VIS void operator delete(void *ptr) noexcept
{
 ::bot::osDealloc(ptr, alignof(void *));
}

BOT_NEWDEL_VIS void * operator new(
    size_t num_bytes, const std::nothrow_t &) noexcept
{
 return ::bot::osAlloc(num_bytes, alignof(void *));
}

BOT_NEWDEL_VIS void operator delete(
    void *ptr, const std::nothrow_t &) noexcept
{
  ::bot::osDealloc(ptr, alignof(void *));
}

BOT_NEWDEL_VIS void * operator new[](size_t num_bytes)
{
  return ::bot::osAlloc(num_bytes, alignof(void *));
}

BOT_NEWDEL_VIS void operator delete[](void *ptr) noexcept
{
  ::bot::osDealloc(ptr, alignof(void *));
}

BOT_NEWDEL_VIS void * operator new[](
    size_t num_bytes, const std::nothrow_t &) noexcept
{
  return ::bot::osAlloc(num_bytes, alignof(void *));
}

BOT_NEWDEL_VIS void operator delete[](
    void *ptr, const std::nothrow_t &) noexcept
{
  ::bot::osDealloc(ptr, alignof(void *));
}

// Aligned versions

BOT_NEWDEL_VIS void * operator new(size_t num_bytes, std::align_val_t al)
{
  return ::bot::osAlloc(num_bytes, (size_t)al);
}

BOT_NEWDEL_VIS void operator delete(void *ptr, std::align_val_t al) noexcept
{
  ::bot::osDealloc(ptr, (size_t)al);
}

BOT_NEWDEL_VIS void * operator new(
    size_t num_bytes, std::align_val_t al, const std::nothrow_t &) noexcept
{
  return ::bot::osAlloc(num_bytes, (size_t)al);
}

BOT_NEWDEL_VIS void operator delete(
    void *ptr, std::align_val_t al, const std::nothrow_t &) noexcept
{
  ::bot::osDealloc(ptr, (size_t)al);
}

BOT_NEWDEL_VIS void * operator new[](size_t num_bytes, std::align_val_t al)
{
  return ::bot::osAlloc(num_bytes, (size_t)al);
}

BOT_NEWDEL_VIS void operator delete[](void *ptr, std::align_val_t al) noexcept
{
  ::bot::osDealloc(ptr, (size_t)al);
}

BOT_NEWDEL_VIS void * operator new[](
    size_t num_bytes, std::align_val_t al, const std::nothrow_t &) noexcept
{
  return ::bot::osAlloc(num_bytes, (size_t)al);
}

BOT_NEWDEL_VIS void operator delete[](
    void *ptr, std::align_val_t al, const std::nothrow_t &) noexcept
{
  ::bot::osDealloc(ptr, (size_t)al);
}
