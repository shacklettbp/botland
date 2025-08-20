/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#define BOT_STRINGIFY_HELPER(m) #m
#define BOT_STRINGIFY(m) BOT_STRINGIFY_HELPER(m)
#define BOT_MACRO_CONCAT_HELPER(x, y) x##y
#define BOT_MACRO_CONCAT(x, y) BOT_MACRO_CONCAT_HELPER(x, y)

#define BOT_LOC_APPEND(m) m ": " __FILE__ " @ " BOT_STRINGIFY(__LINE__)

#if defined(BOT_CXX_CLANG) or defined(BOT_CXX_GCC) or defined(BOT_GPU)
#define BOT_COMPILER_FUNCTION_NAME __PRETTY_FUNCTION__
#elif defined(BOT_CXX_MSVC)
#define BOT_COMPILER_FUNCTION_NAME __FUNCSIG__
#endif

#ifdef BOT_ARCH_X64
#define BOT_CACHE_LINE (64)
#elif defined(BOT_ARCH_ARM) && defined(BOT_OS_MACOS)
#define BOT_CACHE_LINE (128)
#else
#define BOT_CACHE_LINE (64)
#endif

#if defined(BOT_CXX_MSVC)

#define BOT_NO_INLINE __declspec(noinline)
#if defined(BOT_CXX_CLANG_CL)
#define BOT_ALWAYS_INLINE __attribute__((always_inline))
#else
#define BOT_ALWAYS_INLINE [[msvc::forceinline]]
#endif

#elif defined(BOT_CXX_CLANG) || defined(BOT_CXX_GCC) || defined(BOT_GPU)

#define BOT_ALWAYS_INLINE __attribute__((always_inline))
#define BOT_NO_INLINE __attribute__((noinline))

#endif

#if defined(BOT_OS_WINDOWS)
#define BOT_IMPORT __declspec(dllimport)
#define BOT_EXPORT __declspec(dllexport)
#else
#define BOT_IMPORT __attribute__ ((visibility ("default")))
#define BOT_EXPORT __attribute__ ((visibility ("default")))
#endif

#if defined(BOT_CXX_MSVC)
#define BOT_UNREACHABLE() __assume(0)
#else
#define BOT_UNREACHABLE() __builtin_unreachable()
#endif

#if defined(BOT_CXX_CLANG) || defined(BOT_CXX_CLANG_CL)
#define BOT_LFBOUND [[clang::lifetimebound]]
#elif defined(BOT_CXX_MSVC)
#define BOT_LFBOUND [[msvc::lifetimebound]]
#else
#define BOT_LFBOUND
#endif

#define BOT_UNIMPLEMENTED() \
    static_assert(false, "Unimplemented")

#if defined(BOT_GPU) || defined(BOT_CXX_CLANG)
#define BOT_UNROLL _Pragma("unroll")
#else
#define BOT_UNROLL
#endif


#ifdef BOT_GPU
#define BOT_GPU_COND(...) __VA_ARGS__
#else
#define BOT_GPU_COND(...)
#endif

#ifdef BOT_GPU
#define BOT_GPU_SELECT(a, b) (a)
#else
#define BOT_GPU_SELECT(a, b) (b)
#endif
