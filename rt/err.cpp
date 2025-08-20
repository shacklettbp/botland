#include "err.hpp"

#ifdef BOT_GPU
extern "C" {
  int vprintf(const char *, va_list);
}
#else
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#endif

#if defined(BOT_OS_LINUX) || defined(BOT_OS_MACOS)
#include "signal.h"
#endif

namespace bot {

void failAssert(const char *str, const char *file, int line, const char *funcname)
{
#ifdef BOT_GPU
  __assertfail(str, file, line, funcname, 1);
  __trap();
#else
  fprintf(stderr, "Assert '%s' failed in %s at %s line %d", str, funcname, file, line);
  fflush(stderr);
  abort();
#endif
}

#ifdef BOT_GPU
__noinline__
#endif
void fatal(const char *file, int line, const char *funcname,
           const char *fmt, ...)
{
#ifdef BOT_GPU
  printf("Fatal error in %s at %s:%d\n", funcname, file, line);

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  printf("\n");

  __trap();
#else
  // Use a fixed size buffer for the error message. This sets an upper
  // bound on total memory size, and wastes 4kb on memory, but is very
  // robust to things going horribly wrong elsewhere.
  static std::array<char, 4096> buffer;

  va_list args;
  va_start(args, fmt);
  vsnprintf(buffer.data(), buffer.size(), fmt, args);

  fprintf(stderr, "Fatal error at %s:%d in %s\n", file, line,
          funcname);
  fprintf(stderr, "%s\n", buffer.data());

  fflush(stderr);
  abort();
#endif
}

void debuggerBreakPoint()
{
#if defined(BOT_GPU)
  __trap();
#elif defined(BOT_OS_LINUX) || defined(BOT_OS_MACOS)
  signal(SIGTRAP, SIG_IGN);
  raise(SIGTRAP);
  signal(SIGTRAP, SIG_DFL);
#elif defined(BOT_OS_WINDOWS)
  if (IsDebuggerPresent()) {
    DebugBreak();
  }
#endif
}

}
