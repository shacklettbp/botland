#pragma once

#include "macros.hpp"

namespace bot {

[[noreturn]] void fatal(const char *file, int line,
    const char *funcname, const char *fmt, ...);

BOT_ALWAYS_INLINE inline void checkAssert(bool cond, const char *str,
                                         const char *file, int line,
                                         const char *funcname);

void debuggerBreakPoint();

}

#define chk(x) ::bot::checkAssert(x, #x, __FILE__, __LINE__, BOT_COMPILER_FUNCTION_NAME)

#if __cplusplus >= 202002L
#define FATAL(fmt, ...) ::bot::fatal(__FILE__, __LINE__,\
    BOT_COMPILER_FUNCTION_NAME, fmt __VA_OPT__(,) __VA_ARGS__ )
#else
#define FATAL(fmt, ...) ::bot::fatal(__FILE__, __LINE__,\
    BOT_COMPILER_FUNCTION_NAME, fmt ##__VA_ARGS__ )
#endif

#include "err.inl"
