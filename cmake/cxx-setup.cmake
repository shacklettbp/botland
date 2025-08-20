set(BOT_CXX_CLANG FALSE)
set(BOT_CXX_GCC FALSE)
set(BOT_CXX_MSVC FALSE)

if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(BOT_CXX_CLANG TRUE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(BOT_CXX_GCC TRUE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(BOT_CXX_MSVC TRUE)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  if (CMAKE_CXX_COMPILER_BOT_CXX_VARIANT STREQUAL "MSVC")
    set(BOT_CXX_MSVC TRUE)
    set(BOT_CXX_CLANG_CL TRUE)
  else ()
    set(BOT_CXX_CLANG TRUE)
  endif ()
endif ()

if (BOT_CXX_MSVC)
  string(REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
else()
  string(REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
  string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -ggdb3")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb3")
endif()

add_library(bot-sys-defns INTERFACE)

if (BOT_CXX_CLANG)
  list(APPEND BOT_SYS_DEFNS "BOT_CXX_CLANG=1")
endif()

if (BOT_CXX_GCC)
  list(APPEND BOT_SYS_DEFNS "BOT_CXX_GCC=1")
endif()

if (BOT_CXX_MSVC)
  list(APPEND BOT_SYS_DEFNS "BOT_CXX_MSVC=1")
endif()

if (BOT_CXX_CLANG_CL)
  list(APPEND BOT_SYS_DEFNS "BOT_CXX_CLANG_CL=1")
endif()

if (BOT_OS_LINUX)
  list(APPEND BOT_SYS_DEFNS "BOT_OS_LINUX=1")
endif()

if (BOT_OS_MACOS)
  list(APPEND BOT_SYS_DEFNS "BOT_OS_MACOS=1")
endif()

if (BOT_OS_WINDOWS)
  list(APPEND BOT_SYS_DEFNS "BOT_OS_WINDOWS=1")
endif()

if (BOT_ARCH_X64)
  list(APPEND BOT_SYS_DEFNS "BOT_OS_X64=1")
endif()

if (BOT_ARCH_ARM)
  list(APPEND BOT_SYS_DEFNS "BOT_ARCH_ARM=1")
endif()

target_compile_definitions(bot-sys-defns INTERFACE ${BOT_SYS_DEFNS})

add_library(bot-cxx-flags INTERFACE)
target_link_libraries(bot-cxx-flags INTERFACE
    bot-sys-defns
)

if (BOT_CXX_CLANG OR BOT_CXX_GCC)
  target_compile_options(bot-cxx-flags INTERFACE
    -pedantic -Wall -Wextra
  )

  if (BOT_ARCH_X64 AND BOT_OS_LINUX)
    target_compile_options(bot-cxx-flags INTERFACE
      -march=x86-64-v3
    )
  elseif (BOT_ARCH_ARM AND BOT_OS_MACOS)
    target_compile_options(bot-cxx-flags INTERFACE
      -mcpu=apple-m1
    )
  endif()
elseif (BOT_CXX_MSVC)
  # FIXME: some of these options (/permissive-, /Zc:__cplusplus,
  # /Zc:preprocessor) should just be applied globally to the toolchain
  target_compile_options(bot-cxx-flags INTERFACE
    /Zc:__cplusplus
    /permissive-
    /W4
    /wd4324 # Struct padded for alignas ... yeah that's the point
    /wd4701 # Potentially uninitialized variable. MSVC analysis really sucks on this
    /wd4244 /wd4267 # Should reenable these
  )

  if (NOT BOT_CXX_CLANG_CL)
    target_compile_options(bot-cxx-flags INTERFACE
      /Zc:preprocessor
    )
  endif()
endif()

if (BOT_CXX_GCC)
  target_compile_options(bot-cxx-flags INTERFACE
    -fdiagnostics-color=always  
  )
elseif (BOT_CXX_CLANG)
  target_compile_options(bot-cxx-flags INTERFACE
    -fcolor-diagnostics
  )
endif ()

if (BOT_CXX_CLANG)
  target_compile_options(bot-cxx-flags INTERFACE
    -Wshadow
  )
endif()

add_library(bot-cxx-noexceptrtti INTERFACE)
if (BOT_CXX_GCC OR BOT_CXX_CLANG)
  target_compile_options(bot-cxx-noexceptrtti INTERFACE
    -fno-exceptions -fno-rtti
  )
elseif (BOT_CXX_MSVC)
  target_compile_options(bot-cxx-noexceptrtti INTERFACE
    /GR-
  )
else()
  message(FATAL_ERROR "Unsupported compiler frontend")
endif()

add_library(bot-libcxx INTERFACE)
if (BOT_USE_BUNDLED_TOOLCHAIN)
  target_link_libraries(bot-libcxx INTERFACE madrona_libcxx)
endif ()

install(
  TARGETS bot-sys-defns bot-cxx-noexceptrtti bot-cxx-flags bot-libcxx
  EXPORT bot-cxx)
install(EXPORT bot-cxx DESTINATION "${CMAKE_INSTALL_PREFIX}")
