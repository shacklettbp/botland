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
  static constexpr u32 ATLAS_WIDTH = 512;
  static constexpr u32 ATLAS_HEIGHT = 512;
  static constexpr u32 FIRST_CHAR = 32;   // Space character
  static constexpr u32 LAST_CHAR = 126;   // Tilde character
  static constexpr u32 NUM_CHARS = LAST_CHAR - FIRST_CHAR + 1;
  
  u8* bitmap = nullptr;
  CharInfo charInfo[NUM_CHARS];
  float fontSize;
  float lineHeight;
  
  Texture texture;
  
  void init(MemArena& arena, float fontSize);
  void createGPUTexture(GPUDevice* gpu, GPUQueue queue);
  void destroy(GPUDevice* gpu);
  
  CharInfo getCharInfo(char c) const {
    if ((u32)c < FIRST_CHAR || (u32)c > LAST_CHAR) {
      return charInfo[0]; // Return space for invalid chars
    }
    return charInfo[c - FIRST_CHAR];
  }
};

} // namespace bot