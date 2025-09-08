#pragma once

#include <rt/rt.hpp>
#include <gas/gas.hpp>

namespace bot {
using namespace gas;

struct CharInfo {
  float x0, y0, x1, y1;  // UV coordinates in texture atlas
  float xoff, yoff;       // Offset when rendering
  float xadvance;         // How much to advance for next character
};

struct FontAtlas {
  static constexpr u32 ATLAS_WIDTH = 1024;
  static constexpr u32 ATLAS_HEIGHT = 1024;
  static constexpr u32 FIRST_CHAR = 32;   // Space character
  static constexpr u32 LAST_CHAR = 126;   // Tilde character
  static constexpr u32 NUM_CHARS = LAST_CHAR - FIRST_CHAR + 1;
  
  CharInfo charInfo[NUM_CHARS] = {};
  float fontSize = 0;
  float lineHeight = 0;
  
  Texture texture = {};
  
  void init(Runtime &rt, GPUDevice *gpu, GPUQueue queue, float fontSize);
  void destroy(GPUDevice *gpu);
  
  CharInfo getCharInfo(char c) const {
    if ((u32)c < FIRST_CHAR || (u32)c > LAST_CHAR) {
      return charInfo[0]; // Return space for invalid chars
    }
    return charInfo[c - FIRST_CHAR];
  }
};

} // namespace bot