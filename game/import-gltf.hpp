#pragma once

#include <bot/scene/import.hpp>

#include <memory>

namespace bot {

struct GLTFLoader {
  struct Impl;

  GLTFLoader(ImageImporter &img_importer, brt::Span<char> err_buf);
  GLTFLoader(GLTFLoader &&) = default;
  ~GLTFLoader();

  std::unique_ptr<Impl> impl_;

  bool load(const char *path,
    ImportedGeometryAssets &imported_assets,
    bool merge_and_flatten,
    ImageImporter &img_importer);
};

}
