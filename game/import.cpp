#include "import.hpp"

#include <utility>
#include <algorithm>

#include <string_view>
#include <filesystem>
#include <string>

#include <meshoptimizer.h>

#include "import-obj.hpp"
#include "import-stl.hpp"

#ifdef BOT_GLTF_SUPPORT
#include "import-gltf.hpp"
#endif

#ifdef BOT_USD_SUPPORT
#include "import-usd.hpp"
#endif

#include "rt/rt.hpp"

namespace bot {

struct AssetLoaders {
  std::optional<OBJLoader> objLoader;
  std::optional<STLLoader> stlLoader;

#ifdef BOT_GLTF_SUPPORT
  std::optional<GLTFLoader> gltfLoader;
#endif

#ifdef BOT_USD_SUPPORT
  std::optional<USDLoader> usdLoader;
#endif
};

static uint32_t importGeometry(
    ImportedAssets &imported,
    const char * const path,
    Span<char> err_buf,
    bool one_object_per_asset,
    AssetLoaders &loaders)
{
  (void)one_object_per_asset;

  bool load_success = false;

  uint32_t old_size = imported.objects.size();

  std::string_view path_view(path);

  auto extension_pos = path_view.rfind('.');
  if (extension_pos == path_view.npos) {
    FATAL("File has no extension\n");
  }
  auto extension = path_view.substr(extension_pos + 1);

  if (extension == "obj") {
    if (!loaders.objLoader.has_value()) {
      loaders.objLoader.emplace(err_buf);
    }

    load_success = loaders.objLoader->load(path, imported);
  } else if (extension == "stl") {
    if (!loaders.stlLoader.has_value()) {
      loaders.stlLoader.emplace(err_buf);
    }

    load_success = loaders.stlLoader->load(path, imported);
  } else if (extension == "gltf" || extension == "glb") {
#ifdef BOT_GLTF_SUPPORT
    if (!loaders.gltfLoader.has_value()) {
      loaders.gltfLoader.emplace(imgImporter, err_buf);
    }

    load_success = loaders.gltfLoader->load(
        path, imported, one_object_per_asset, imgImporter);
#else
    load_success = false;
    snprintf(err_buf.data(), err_buf.size(),
        "Deja not compiled with glTF support");
#endif
  } else if (extension == "usd" ||
      extension == "usda" ||
      extension == "usdc" ||
      extension == "usdz") {
#ifdef BOT_USD_SUPPORT
    if (!loaders.usdLoader.has_value()) {
      loaders.usdLoader.emplace(imgImporter, err_buf);
    }

    load_success = loaders.usdLoader->load(
        path, imported, one_object_per_asset, imgImporter);
#else
    load_success = false;
    snprintf(err_buf.data(), err_buf.size(),
        "Deja not compiled with USD support");
#endif
  }

  if (!load_success) {
    printf("Load failed: %s\n", err_buf.data());
  }

  uint32_t new_size = imported.objects.size();

  // For now, we enforce that the number of objects imported by an asset file
  // is just 1.
  assert(new_size - old_size == 1);

  // The ID is the old size
  return old_size;
}

struct AssetImporter::Impl {
  ImageImporter imgImporter;

  ImportedAssets imported;

  AssetLoaders loaders;

  static inline Impl * make(ImageImporter &&img_importer);

  inline uint32_t importAsset(
      const char * const asset_path,
      Span<char> err_buf = { nullptr, 0 },
      bool one_object = false);
};

AssetImporter::Impl * AssetImporter::Impl::make(
    ImageImporter &&img_importer)
{
  return new Impl {
    .imgImporter = std::move(img_importer),
    .imported = {
      .geoData = ImportedAssets::GeometryData {
        .positionArrays { 0 },
        .normalArrays { 0 },
        .tangentAndSignArrays { 0 },
        .uvArrays { 0 },
        .indexArrays { 0 },
        .faceCountArrays { 0 },
        .meshArrays { 0 },
      },
      .objects { 0 },
      .materials { 0 },
      .instances { 0 },
      .textures { 0 },
    },
    .loaders = {
      .objLoader = std::optional<OBJLoader>{},
      .stlLoader = std::optional<STLLoader>{},
#ifdef BOT_GLTF_SUPPORT
      .gltfLoader = std::optional<GLTFLoader>{},
#endif
#ifdef BOT_USD_SUPPORT
      .usdLoader = std::optional<USDLoader>{},
#endif
    },
  };
}

ImageImporter & AssetImporter::imageImporter()
{
  return impl_->imgImporter;
}

ImportedAssets & AssetImporter::getImportedAssets()
{
  return impl_->imported;
}

uint32_t AssetImporter::Impl::importAsset(
    const char * const path,
    Span<char> err_buf,
    bool one_object_per_asset)
{
  return importGeometry(
      imported,
      path,
      err_buf,
      one_object_per_asset,
      loaders);
}

uint32_t AssetImporter::importAsset(
    const std::string &asset_path,
    Span<char> err_buf,
    bool one_object)
{
  return importAsset(asset_path.c_str(), err_buf, one_object);
}

AssetImporter::AssetImporter()
    : AssetImporter(ImageImporter())
{}

AssetImporter::AssetImporter(ImageImporter &&img_importer)
    : impl_(Impl::make(std::move(img_importer)))
{}

AssetImporter::AssetImporter(AssetImporter &&) = default;
AssetImporter::~AssetImporter() = default;

uint32_t AssetImporter::importAsset(
    const char * const path,
    Span<char> err_buf,
    bool one_object_per_asset)
{
  return impl_->importAsset(path, err_buf, one_object_per_asset);
}

}
