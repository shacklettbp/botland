#pragma once

#include "import.hpp"

namespace bot {

struct OBJLoader {
  struct Impl;

  OBJLoader(Span<char> err_buf);
  OBJLoader(OBJLoader &&) = default;
  ~OBJLoader();

  std::unique_ptr<Impl> impl_;

  bool load(const char *path,
            ImportedGeometryAssets &imported_assets);
};

}
