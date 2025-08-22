cmake_minimum_required(VERSION 3.24 FATAL_ERROR)
cmake_policy(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

IF (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
ENDIF()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

option(BOT_WEB "Web compatible build" OFF)

include(toolchain)

list(APPEND LANGS C CXX)
if (APPLE AND NOT BOT_WEB)
  list(APPEND LANGS OBJC)
endif()

project(botland LANGUAGES ${LANGS})

if (BOT_WEB AND NOT EMSCRIPTEN)
  message(FATAL_ERROR "BOT_WEB option must only be true when building with emscripten")
elseif (EMSCRIPTEN AND BOT_USE_BUNDLED_TOOLCHAIN)
  message(FATAL_ERROR "Web support should use emscripten toolchain, not bundled toolchain")
endif()

include(sys-detect)
include(cxx-setup)

add_subdirectory(deps EXCLUDE_FROM_ALL)

add_library(bot-hdrs INTERFACE)
target_include_directories(bot-hdrs INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(bot-common-flags INTERFACE)
target_link_libraries(bot-common-flags INTERFACE
  bot-hdrs
  bot-cxx-flags
  bot-cxx-noexceptrtti
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_PDB_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pdb)

add_subdirectory(rt)

add_subdirectory(net)

add_subdirectory(game)

option(BOT_BUILD_VIZ "Build visualizer" ON)

if (BOT_BUILD_VIZ)
  add_subdirectory(gas EXCLUDE_FROM_ALL)
  add_subdirectory(viz)
endif()
