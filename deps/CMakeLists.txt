if (BOT_USE_BUNDLED_TOOLCHAIN)
  add_subdirectory(toolchain)
endif()

if (NOT BOT_OS_WINDOWS)
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
endif()
find_package(Threads REQUIRED GLOBAL)

if (BOT_DISABLE_CUDA)
  set(CUDAToolkit_FOUND FALSE)
  set(BOT_CUDA FALSE)
else ()
  find_package(CUDAToolkit GLOBAL)

  if (CUDAToolkit_FOUND)
    find_library(CUDA_NVJITLINK_LIBRARY nvJitLink
      PATHS
          ${CUDAToolkit_LIBRARY_DIR}
      REQUIRED
    )

    set(CUDA_NVJITLINK_LIBRARY ${CUDA_NVJITLINK_LIBRARY} PARENT_SCOPE)
    set(CUDAToolkit_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} PARENT_SCOPE)
    set(CUDAToolkit_VERSION_MAJOR ${CUDAToolkit_VERSION_MAJOR} PARENT_SCOPE)
    set(CUDAToolkit_VERSION_MINOR ${CUDAToolkit_VERSION_MINOR} PARENT_SCOPE)

    set(BOT_CUDA TRUE)
  else ()
    set(BOT_CUDA FALSE)
  endif ()
endif ()

set(BOT_CUDA ${BOT_CUDA} PARENT_SCOPE)

set(BOT_BUNDLE_TMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/bundle-tmp")
file(MAKE_DIRECTORY "${BOT_BUNDLE_TMP_DIR}")

include(git-deps.cmake)

fetch_git_dependency(gtest
  URL https://github.com/google/googletest.git
  TAG e4ece4881d1fefc1e67d21c7493835815cd13085
  SUBMODULES OFF
)

add_subdirectory(gtest EXCLUDE_FROM_ALL)
target_link_libraries(gtest_main PUBLIC
  bot-cxx-flags bot-cxx-noexceptrtti bot-libcxx)
target_link_libraries(gtest PUBLIC
  bot-cxx-flags bot-cxx-noexceptrtti bot-libcxx)

fetch_git_dependency(tinyxml2
  URL https://github.com/leethomason/tinyxml2.git
  TAG 9148bdf719e997d1f474be6bcc7943881046dba1
)

add_subdirectory(tinyxml2 EXCLUDE_FROM_ALL)
set_property(TARGET tinyxml2 PROPERTY
  POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(tinyxml2 PRIVATE
  bot-cxx-flags bot-cxx-noexceptrtti bot-libcxx )

fetch_git_dependency(meshoptimizer
  URL https://github.com/zeux/meshoptimizer.git
  TAG 4e35e1fd6a787fe09f10abcb70159f138e75c6e9
)

add_subdirectory(meshoptimizer EXCLUDE_FROM_ALL)
set_property(TARGET meshoptimizer PROPERTY
    POSITION_INDEPENDENT_CODE TRUE)

target_link_libraries(meshoptimizer PRIVATE
  bot-cxx-flags bot-cxx-noexceptrtti bot-libcxx)

fetch_git_dependency(stb
  URL https://github.com/nothings/stb.git
  TAG f58f558c120e9b32c217290b80bad1a0729fbb2c
)

add_library(stb INTERFACE)
target_include_directories(stb SYSTEM INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/stb
)

fetch_git_dependency(fast_float
  URL https://github.com/fastfloat/fast_float.git
  TAG b8085ba3634018ad6e5618cf87dd5a04a0e3f146
)

add_subdirectory(fast_float EXCLUDE_FROM_ALL)
