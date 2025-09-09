gas_add_shaders(TARGET bot-viz-global-shaders
  SHADER_ENUM GlobalShaderID
  SHADER_CLASS VizGlobalShaders
  CPP_NAMESPACE bot
  SHADERS
    Tonemap tonemap.slang
)

gas_add_shaders(TARGET bot-viz-material-shaders
  SHADER_ENUM MaterialShaderID
  SHADER_CLASS VizMaterialShaders
  CPP_NAMESPACE bot
  SHADERS
    Board board.slang
    Units units.slang
    HealthBar healthbar.slang
    Name name.slang
    GenericLocationEffect generic_location_effect.slang
)

add_library(bot-viz STATIC
  viz.hpp viz.cpp
  font.hpp font.cpp
)

target_link_libraries(bot-viz PUBLIC
  gas_core gas_imgui gas_ui
  bot-viz-global-shaders bot-viz-material-shaders
  stb)

target_compile_definitions(bot-viz PRIVATE
  -DBOT_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
  -DBOT_SHADERS_OUT_DIR="${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/"
)

if (BOT_WEB)
  add_executable(bot-viz-web web.cpp)
  target_link_options(bot-viz-web PRIVATE
    # Must be 1 page short of 4GB limit. Note that >2GB incurs a performance cost
    -sINITIAL_MEMORY=4294901760
    -sALLOW_MEMORY_GROWTH=0
    -sABORTING_MALLOC=0
  )

  target_link_libraries(bot-viz-web bot-viz)
else ()
  add_executable(native_frontend native_frontend.cpp)
  target_link_libraries(native_frontend bot-viz gas_shader_compiler bot-backend)

  target_compile_definitions(native_frontend PRIVATE
    -DBOT_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
  )
endif()
