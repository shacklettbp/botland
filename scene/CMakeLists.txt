add_library(bot-scene STATIC
  scene.cpp scene.hpp scene.inl
  img.cpp
  import.cpp import.hpp
  obj.cpp obj.hpp
  stb_read.cpp
  stl.cpp stl.hpp
  urdf.cpp
)

target_link_libraries(bot-scene PUBLIC
  bot-rt
  #simbotson
  tinyxml2
  meshoptimizer
  fast_float
  stb
)
