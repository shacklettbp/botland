#include "font.hpp"

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>
#include <cstring>

namespace bot {
using namespace gas;

// Default font data (using a simple bitmap font or embedded font)
// For now, we'll load from a system font file
static const char* getDefaultFontPath() {
  #ifdef __APPLE__
    return "/System/Library/Fonts/Helvetica.ttc";
  #elif _WIN32
    return "C:/Windows/Fonts/arial.ttf";
  #else
    return "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf";
  #endif
}

void FontAtlas::init(MemArena& arena, float requestedFontSize) {
  fontSize = requestedFontSize;
  
  // Allocate bitmap
  // Use static allocation for simplicity
  static u8 staticBitmap[ATLAS_WIDTH * ATLAS_HEIGHT];
  bitmap = staticBitmap;
  memset(bitmap, 0, ATLAS_WIDTH * ATLAS_HEIGHT);
  
  // Load font file
  FILE* fontFile = fopen(getDefaultFontPath(), "rb");
  if (!fontFile) {
    // Fall back to a very simple font if we can't load a system font
    // For now, just fill with white so text is visible
    memset(bitmap, 255, ATLAS_WIDTH * ATLAS_HEIGHT);
    lineHeight = fontSize;
    for (u32 i = 0; i < NUM_CHARS; i++) {
      charInfo[i] = {
        .x0 = 0, .y0 = 0,
        .x1 = 1, .y1 = 1,
        .xoff = 0, .yoff = 0,
        .xadvance = fontSize * 0.6f
      };
    }
    return;
  }
  
  // Get font file size
  fseek(fontFile, 0, SEEK_END);
  long fileSize = ftell(fontFile);
  fseek(fontFile, 0, SEEK_SET);
  
  // Read font data
  u8* fontBuffer = new u8[fileSize];
  fread(fontBuffer, 1, fileSize, fontFile);
  fclose(fontFile);
  
  // Initialize font
  stbtt_fontinfo font;
  if (!stbtt_InitFont(&font, fontBuffer, 0)) {
    // Failed to init font, fall back to white square
    memset(bitmap, 255, ATLAS_WIDTH * ATLAS_HEIGHT);
    lineHeight = fontSize;
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
      
      // Store character info
      charInfo[i] = {
        .x0 = (float)x / ATLAS_WIDTH,
        .y0 = (float)y / ATLAS_HEIGHT,
        .x1 = (float)(x + width) / ATLAS_WIDTH,
        .y1 = (float)(y + height) / ATLAS_HEIGHT,
        .xoff = (float)xoff,
        .yoff = (float)yoff + ascent * scale,
        .xadvance = 0
      };
      
      // Get advance width
      int advanceWidth, leftSideBearing;
      stbtt_GetCodepointHMetrics(&font, c, &advanceWidth, &leftSideBearing);
      charInfo[i].xadvance = advanceWidth * scale;
      
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
  
  delete[] fontBuffer;
}

void FontAtlas::createGPUTexture(GPUDevice* gpu, GPUQueue queue) {
  // Convert single-channel bitmap to RGBA
  u32 dataSize = ATLAS_WIDTH * ATLAS_HEIGHT * 4;
  u8* rgbaData = new u8[dataSize];
  
  for (u32 i = 0; i < ATLAS_WIDTH * ATLAS_HEIGHT; i++) {
    rgbaData[i * 4 + 0] = 255;  // R
    rgbaData[i * 4 + 1] = 255;  // G
    rgbaData[i * 4 + 2] = 255;  // B
    rgbaData[i * 4 + 3] = bitmap[i];  // A
  }
  
  texture = gpu->createTexture({
    .format = TextureFormat::RGBA8_UNorm,
    .width = ATLAS_WIDTH,
    .height = ATLAS_HEIGHT,
    .initData = { .ptr = rgbaData }
  }, queue);
  
  delete[] rgbaData;
}

void FontAtlas::destroy(GPUDevice* gpu) {
  if (!texture.null()) {
    gpu->destroyTexture(texture);
    texture = {};
  }
}

} // namespace bot