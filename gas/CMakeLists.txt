list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(deps)

include(gas-find-deps)

list(APPEND GAS_CORE_SOURCES
  gas.hpp gas.inl gas.cpp
  gas_input.hpp gas_input.inl gas_input.cpp
  backend_common.hpp 
  mem.hpp mem.cpp
  linux.hpp windows.hpp
)

if (GAS_USE_DAWN OR EMSCRIPTEN)
  list(APPEND GAS_CORE_SOURCES
    wgpu.hpp wgpu.cpp wgpu_init.hpp
  )
endif()

add_library(gas_core STATIC
  ${GAS_CORE_SOURCES}
)

target_link_libraries(gas_core PUBLIC bot-rt)

if (GAS_USE_DAWN)
  target_link_libraries(gas_core PRIVATE gas_dawn)
endif()

include(gas-shader-compile)

add_library(gas_ui STATIC
  gas_ui.hpp gas_ui.inl gas_ui.cpp
)

target_link_libraries(gas_ui PRIVATE
  gas_core bot-rt
)

if (GAS_USE_SDL)
  target_link_libraries(gas_ui PRIVATE gas_sdl)
endif()

add_library(gas_imgui STATIC
  gas_imgui.hpp gas_imgui.cpp
)

target_compile_definitions(gas_imgui PRIVATE 
  GAS_IMGUI_SHADER_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\"
)

target_link_libraries(gas_imgui
  PUBLIC
    gas_imgui_hdrs
  PRIVATE
    gas_core gas_imgui_impl bot-rt
)

gas_add_shaders(
  TARGET
    gas_imgui_shaders
  SHADER_ENUM
    ImGuiShaderID 
  SHADER_CLASS
    ImGuiShaders
  CPP_NAMESPACE
    bot::gas
  SHADERS
    Render imgui.slang
)

target_link_libraries(gas_imgui PRIVATE gas_imgui_shaders)

add_subdirectory(test)
