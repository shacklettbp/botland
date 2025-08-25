add_library(bot-asset-import STATIC
  import.cpp import.hpp
  import-img.cpp
  import-obj.cpp import-obj.hpp
  import-stl.cpp import-stl.hpp
  stb_read.cpp
)

target_link_libraries(bot-asset-import PRIVATE
  bot-rt
  tinyxml2
  meshoptimizer
  fast_float
  stb
)

add_library(bot-sim STATIC
  sim.hpp sim.inl sim.cpp
  geo.hpp geo.cpp
)

target_link_libraries(bot-sim PUBLIC
  bot-rt
)

if (BOT_CUDA)
  bot_cuda_target(bot-sim-gpu bot-sim)
endif()

add_library(bot-backend STATIC
  backend.hpp backend.cpp
)

target_link_libraries(bot-backend 
  PUBLIC
    bot-rt
  PRIVATE 
    bot-sim
    bot-asset-import
)

if (BOT_CUDA)
  add_dependencies(bot-backend bot-sim-gpu)
endif()

add_executable(backend_headless
  headless.cpp
)
target_link_libraries(backend_headless
  PRIVATE bot-backend)

target_compile_definitions(backend_headless PRIVATE
  -DBOT_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)
