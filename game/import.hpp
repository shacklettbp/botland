#pragma once

#include <rt/rt.hpp>
#include <rt/math.hpp>
#include <rt/span.hpp>
#include <rt/stack_alloc.hpp>

#include <vector>
#include <optional>

#include "geo.hpp"

#include "physics.hpp"

namespace bot {

struct SourceCollisionPrimitive {
  struct HullInput {
    uint32_t hullIDX;

    // For debugging
    const char *name;
  };

  CollisionPrimitive::Type type;
  union {
    CollisionPrimitive::Sphere sphere;
    CollisionPrimitive::Box box;
    CollisionPrimitive::Plane plane;
    CollisionPrimitive::Capsule capsule;
    HullInput hullInput;
  };
};

struct SourceCollisionObject {
  Span<const SourceCollisionPrimitive> prims;
};

struct SourceMesh {
  Vector3 *positions;
  Vector3 *normals;
  Vector4 *tangentAndSigns;
  Vector2 *uvs;

  uint32_t *indices;
  uint32_t *faceCounts;
  uint32_t *faceMaterials;
  
  uint32_t numVertices;
  uint32_t numFaces;
  uint32_t materialIdx;

  const char *name;
};

struct SourceObject {
  Span<SourceMesh> meshes;
};

enum class SourceTextureFormat {
  R8G8B8A8,
  BC7,
};

struct SourceTexture {
  void *data;
  SourceTextureFormat format;
  uint32_t width;
  uint32_t height;
  size_t numBytes;
};

struct SourceMaterial {
  Vector4 color;

  // If this is -1, no texture will be applied. Otherwise,
  // the color gets multipled by color of the texture read in
  // at the UVs of the pixel.
  int32_t textureIdx;

  float roughness;
  float metalness;
};

struct SourceInstance {
  Vector3 translation;
  Quat rotation;
  Diag3x3 scale;
  uint32_t objIDX;
};
  
class ImageImporter {
public:
  ImageImporter();
  ImageImporter(ImageImporter &&);
  ~ImageImporter();

  using ImportHandler =
    std::optional<SourceTexture> (*)(void *data, size_t num_bytes);

  int32_t addHandler(const char *extension, ImportHandler fn);

  int32_t getPNGTypeCode();
  int32_t getJPGTypeCode();
  int32_t getExtensionTypeCode(const char *extension);

  std::optional<SourceTexture> importImage(
      void *data, size_t num_bytes, int32_t type_code);

  std::optional<SourceTexture> importImage(const char *path);

  Span<SourceTexture> importImages(
      StackAlloc &tmp_alloc, Span<const char * const> paths);

  void deallocImportedImages(Span<SourceTexture> textures);


private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct ImportedGeometryAssets {
  struct GeometryData {
    std::vector<std::vector<Vector3>> positionArrays;
    std::vector<std::vector<Vector3>> normalArrays;
    std::vector<std::vector<Vector4>> tangentAndSignArrays;
    std::vector<std::vector<Vector2>> uvArrays;

    std::vector<std::vector<uint32_t>> indexArrays;
    std::vector<std::vector<uint32_t>> faceCountArrays;

    std::vector<std::vector<SourceMesh>> meshArrays;
  } geoData;

  std::vector<SourceObject> objects;
  std::vector<SourceMaterial> materials;
  std::vector<SourceInstance> instances;
  std::vector<SourceTexture> textures;
};

using ImportedRenderAssets = ImportedGeometryAssets;

// This is purely for importing render assets
class RenderAssetImporter {
public:
  RenderAssetImporter();
  RenderAssetImporter(ImageImporter &&img_importer);
  RenderAssetImporter(RenderAssetImporter &&);
  ~RenderAssetImporter();

  ImageImporter & imageImporter();

  ImportedRenderAssets & getImportedAssets();
  
  // Returns index of the asset. We can also support other forms of importing.
  // Like importing directly from a buffer of vertices or stuff which
  // are directly in memory already (TODO)
  uint32_t importAsset(
      const char * const asset_path,
      Span<char> err_buf = { nullptr, 0 },
      bool one_object = false);

  uint32_t importAsset(
      const std::string &asset_path,
      Span<char> err_buf = { nullptr, 0 },
      bool one_object = false);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct ConvexHullData {
  HalfEdge *halfEdges;
  uint32_t *faceBaseHalfEdges;
  Plane *facePlanes;
  Vector3 *vertices;

  uint32_t numHalfEdges;
  uint32_t numFaces;
  uint32_t numVerts;
};

struct ProcessedPhysicsAssets {
  void *buffer;
  uint64_t bufferSize;

  ConvexHullData hullData;
  CollisionPrimitive *primitives;
  AABB *primitiveAABBs;

  AABB *objAABBs;
  uint32_t *primOffsets;
  uint32_t *primCounts;

  uint32_t numConvexHulls;
  uint32_t totalNumPrimitives;
  uint32_t numObjs;
};

struct ImportedPhysicsAssets {
  std::vector<SourceMesh> srcHulls;

  std::vector<std::vector<SourceCollisionPrimitive>> primArrays;
  std::vector<SourceCollisionObject> objs;
};

// This is purely for importing physics primitives
class PhysicsAssetImporter {
public:
  PhysicsAssetImporter();
  PhysicsAssetImporter(PhysicsAssetImporter &&);
  ~PhysicsAssetImporter();

  ImportedPhysicsAssets & getImportedAssets();

  uint32_t importAsset(
      const char * const asset_path,
      Span<char> err_buf = { nullptr, 0 });

  uint32_t importAsset(
      const std::string &asset_path,
      Span<char> err_buf = { nullptr, 0 });

  // Directly import a primitive
  uint32_t importAsset(
      SourceCollisionPrimitive prim);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class PhysicsAssetProcessor {
public:
  PhysicsAssetProcessor(ImportedPhysicsAssets &imported);
  ~PhysicsAssetProcessor();

  BridgeData<ObjectManager> process(
      Backend *backend,
      bool build_convex_hulls = false);

  // ProcessedPhysicsAssets process(Runtime &rt, bool build_convex_hulls);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}
