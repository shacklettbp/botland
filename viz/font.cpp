#include "font.hpp"

#include <rt/os.hpp>

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>
#include <cstring>

namespace bot {
using namespace gas;

void FontAtlas::init(Runtime &rt, GPUDevice* gpu, GPUQueue queue, float requestedFontSize)
{
  ArenaRegion tmp_region = rt.beginTmpRegion();
  BOT_DEFER(rt.endTmpRegion(tmp_region));

  fontSize = requestedFontSize;
  
  const char *font_path = BOT_DATA_DIR "imgui_font.ttf";
  
  u64 fileSize = 0;
  u8 *font_buffer = (u8 *)readFile(rt, rt.tmpArena(), font_path, &fileSize);
  
  if (!font_buffer) {
    FATAL("Failed to read font file at %s", font_path);
  }

  stbtt_fontinfo font;
  int faceOffset = stbtt_GetFontOffsetForIndex(font_buffer, 0);
  if (faceOffset < 0) faceOffset = 0;
  if (!stbtt_InitFont(&font, font_buffer, faceOffset)) {
    // Failed to init font, fall back to white square
    FATAL("Failed to initialize UI font");
    return;
  }
  
  // Calculate scale for requested font size
  float scale = stbtt_ScaleForPixelHeight(&font, fontSize);
  
  // Get font metrics
  int ascent, descent, lineGap;
  stbtt_GetFontVMetrics(&font, &ascent, &descent, &lineGap);
  lineHeight = (ascent - descent + lineGap) * scale;
  
  // Pack characters into atlas
  int x = 0, y = 0;
  int rowHeight = 0;
  
  u8 *bitmap = rt.arenaAllocN<u8>(rt.tmpArena(), ATLAS_WIDTH * ATLAS_HEIGHT);
  for (u32 i = 0; i < NUM_CHARS; i++) {
    int c = FIRST_CHAR + i;
    
    // Get character bitmap
    int width, height, xoff, yoff;
    u8* charBitmap = stbtt_GetCodepointBitmap(&font, 0, scale, c, &width, &height, &xoff, &yoff);
    
    if (charBitmap) {
      // Check if we need to move to next row
      if (x + width >= ATLAS_WIDTH) {
        x = 0;
        y += rowHeight + 1;
        rowHeight = 0;
      }
      
      // Copy character bitmap to atlas
      for (int cy = 0; cy < height; cy++) {
        for (int cx = 0; cx < width; cx++) {
          if (y + cy < ATLAS_HEIGHT && x + cx < ATLAS_WIDTH) {
            bitmap[(y + cy) * ATLAS_WIDTH + (x + cx)] = charBitmap[cy * width + cx];
          }
        }
      }
      
      // Get advance width
      int advanceWidth, leftSideBearing;
      stbtt_GetCodepointHMetrics(&font, c, &advanceWidth, &leftSideBearing);

      // Store character info
      charInfo[i] = {
        .x0 = (float)x / ATLAS_WIDTH,
        .y0 = (float)y / ATLAS_HEIGHT,
        .x1 = (float)(x + width) / ATLAS_WIDTH,
        .y1 = (float)(y + height) / ATLAS_HEIGHT,
        .xoff = (float)xoff,
        .yoff = (float)yoff + ascent * scale,
        .xadvance = (float)advanceWidth * scale
      };
      
      
      // Update position for next character
      x += width + 1;
      rowHeight = (height > rowHeight) ? height : rowHeight;
      
      // Free character bitmap
      stbtt_FreeBitmap(charBitmap, nullptr);
    } else {
      // Character not available, use space metrics
      charInfo[i] = charInfo[0];
    }
  }
  
  texture = gpu->createTexture({
    .format = TextureFormat::R8_UNorm,
    .width = ATLAS_WIDTH,
    .height = ATLAS_HEIGHT,
    .initData = { .ptr = bitmap }
  }, queue);
}

void FontAtlas::destroy(GPUDevice* gpu) {
  gpu->destroyTexture(texture);
  *this = {};
}

} // namespace bot
