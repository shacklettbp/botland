#pragma once

#include <rt/span.hpp>
#include <scene/import.hpp>

namespace bot {
    
struct STLLoader {
  struct Impl;

  STLLoader(Span<char> err_buf);
  STLLoader(STLLoader &&) = default;
  ~STLLoader();

  std::unique_ptr<Impl> impl_;

  bool load(const char *path, ImportedGeometryAssets &imported_assets);
};

}
