#include <scene/import.hpp>
#include <utility>
#include <algorithm>

#include <string_view>
#include <filesystem>
#include <string>

#include <meshoptimizer.h>

#include "obj.hpp"
#include "stl.hpp"

#ifdef BOT_GLTF_SUPPORT
#include "gltf.hpp"
#endif

#ifdef BOT_USD_SUPPORT
#include "usd.hpp"
#endif

#include "rt/rt.hpp"

#include <sim/physics.hpp>

#include <sim/backend.hpp>

namespace bot {

struct FileLoaders {
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
    ImportedGeometryAssets &imported,
    const char * const path,
    Span<char> err_buf,
    bool one_object_per_asset,
    FileLoaders &loaders)
{
  (void)one_object_per_asset;

  bool load_success = false;

  uint32_t old_size = imported.objects.size();

  printf("Loading asset from %s\n", path);

  std::string_view path_view(path);

  auto extension_pos = path_view.rfind('.');
  if (extension_pos == path_view.npos) {
    printf("File has no extension\n");
    assert(false);
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

struct RenderAssetImporter::Impl {
  ImageImporter imgImporter;

  ImportedRenderAssets imported;

  FileLoaders loaders;

  static inline Impl * make(ImageImporter &&img_importer);

  inline uint32_t importAsset(
      const char * const asset_path,
      Span<char> err_buf = { nullptr, 0 },
      bool one_object = false);
};

RenderAssetImporter::Impl * RenderAssetImporter::Impl::make(
    ImageImporter &&img_importer)
{
  return new Impl {
    .imgImporter = std::move(img_importer),
    .imported = {
      .geoData = ImportedGeometryAssets::GeometryData {
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

ImageImporter & RenderAssetImporter::imageImporter()
{
  return impl_->imgImporter;
}

ImportedRenderAssets & RenderAssetImporter::getImportedAssets()
{
  return impl_->imported;
}

uint32_t RenderAssetImporter::Impl::importAsset(
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

uint32_t RenderAssetImporter::importAsset(
    const std::string &asset_path,
    Span<char> err_buf,
    bool one_object)
{
  return importAsset(asset_path.c_str(), err_buf, one_object);
}

RenderAssetImporter::RenderAssetImporter()
    : RenderAssetImporter(ImageImporter())
{}

RenderAssetImporter::RenderAssetImporter(ImageImporter &&img_importer)
    : impl_(Impl::make(std::move(img_importer)))
{}

RenderAssetImporter::RenderAssetImporter(RenderAssetImporter &&) = default;
RenderAssetImporter::~RenderAssetImporter() = default;

uint32_t RenderAssetImporter::importAsset(
    const char * const path,
    Span<char> err_buf,
    bool one_object_per_asset)
{
  return impl_->importAsset(path, err_buf, one_object_per_asset);
}



struct PhysicsAssetImporter::Impl {
  // The importedAssets uses the importedGeometry
  ImportedGeometryAssets importedGeometry;
  ImportedPhysicsAssets importedAssets;

  FileLoaders loaders;

  static inline Impl * make();
};

PhysicsAssetImporter::Impl * PhysicsAssetImporter::Impl::make()
{
  return new Impl {
    .importedGeometry = {
      .geoData = ImportedGeometryAssets::GeometryData {
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
    .importedAssets = {
      .srcHulls = {},
      .primArrays = {},
      .objs = {},
#if 0
      .hullData = {
        .halfEdges = {},
        .faceBaseHalfEdges = {},
        .facePlanes = {},
        .vertices = {},
        .numHalfEdges = 0,
        .numFaces = 0,
        .numVerts = 0,
      },
      .primitives = {},
      .primitiveAABBs = {},
      .objAABBs = {},
      .primOffsets = {},
      .primCounts = {},
      .numConvexHulls = 0,
      .totalNumPrimitives = 0,
      .numObjs = 0,
#endif
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

PhysicsAssetImporter::PhysicsAssetImporter()
  : impl_(Impl::make())
{
}

PhysicsAssetImporter::PhysicsAssetImporter(PhysicsAssetImporter &&o) = default;
PhysicsAssetImporter::~PhysicsAssetImporter() = default;

ImportedPhysicsAssets & PhysicsAssetImporter::getImportedAssets()
{
  return impl_->importedAssets;
}

uint32_t PhysicsAssetImporter::importAsset(
    const char * const asset_path,
    Span<char> err_buf)
{
  uint32_t geometry_asset_idx = importGeometry(
      impl_->importedGeometry,
      asset_path,
      err_buf,
      false,
      impl_->loaders);

  // Now, prepare the hull info
  auto meshes = impl_->importedGeometry.objects[geometry_asset_idx].meshes;

  std::vector<SourceCollisionPrimitive> prims;
  prims.reserve(meshes.size());

  for (const SourceMesh &mesh : meshes) {
    impl_->importedAssets.srcHulls.push_back(mesh);
    prims.push_back({
      .type = CollisionPrimitive::Type::Hull,
      .hullInput = {
        .hullIDX = (uint32_t)(impl_->importedAssets.srcHulls.size() - 1),
        .name = ""
      },
    });
  }

  impl_->importedAssets.primArrays.push_back(std::move(prims));

  impl_->importedAssets.objs.push_back(SourceCollisionObject {
    .prims = Span<const SourceCollisionPrimitive>(
      impl_->importedAssets.primArrays.back()
    )
  });

  return (uint32_t)(impl_->importedAssets.objs.size() - 1);
}

uint32_t PhysicsAssetImporter::importAsset(
    const std::string &asset_path,
    Span<char> err_buf)
{
  return importAsset(asset_path.c_str(), err_buf);
}

uint32_t PhysicsAssetImporter::importAsset(
    SourceCollisionPrimitive prim)
{
  std::vector<SourceCollisionPrimitive> prims;
  prims.push_back(prim);

  impl_->importedAssets.primArrays.push_back(std::move(prims));

  impl_->importedAssets.objs.push_back(SourceCollisionObject {
    .prims = Span<const SourceCollisionPrimitive>(
        impl_->importedAssets.primArrays.back()
    )
  });

  return (uint32_t)(impl_->importedAssets.objs.size() - 1);
}

struct EditMesh {
  struct HEdge {
    uint32_t next;
    uint32_t prev;
    uint32_t twin;

    uint32_t vert;
    uint32_t face;
  };

  struct Face {
    uint32_t hedge;
    uint32_t next;
    uint32_t prev;

    Plane plane;
  };

  struct Vert {
    Vector3 pos;
    uint32_t next;
    uint32_t prev;
  };

  HEdge *hedges;
  Face *faces;
  Vert *verts;

  uint32_t numHedges;
  uint32_t numFaces;
  uint32_t numVerts;

  uint32_t hedgeFreeHead;
  uint32_t faceFreeHead;
  uint32_t vertFreeHead;
};

struct HullBuildData {
  EditMesh mesh;
  uint32_t *faceConflictLists;
};

struct MassProperties {
  Diag3x3 inertiaTensor;
  Vector3 centerOfMass;
  Quat toDiagonal;
};

struct PhysicsBridgeData {
  CollisionPrimitive *primitives;
  AABB *primAABBs;

  AABB *objAABBs;
  uint32_t *rigidBodyPrimitiveOffsets;
  uint32_t *rigidBodyPrimitiveCounts;

  u64 curPrimOffset;
  u64 curObjOffset;

  ObjectManager *mgr;
  u64 maxPrims;
  u64 maxObjs;

  int gpuID;
};

struct PhysicsAssetProcessor::Impl {
  ImportedPhysicsAssets *imported;

  static inline Impl * make(ImportedPhysicsAssets &imported)
  {
    return new Impl {
      &imported
    };
  }

  uint32_t allocMeshHedge(EditMesh &mesh)
  {
    uint32_t hedge = mesh.hedgeFreeHead;
    assert(hedge != 0);
    mesh.hedgeFreeHead = mesh.hedges[hedge].next;

    return hedge;
  }

  void freeMeshHedge(EditMesh &mesh, uint32_t hedge)
  {
    uint32_t old_head = mesh.hedgeFreeHead;
    mesh.hedgeFreeHead = hedge;
    mesh.hedges[hedge].next = old_head;
  }

  uint32_t createMeshFace(EditMesh &mesh)
  {
    uint32_t face = mesh.faceFreeHead;
    assert(face != 0);
    mesh.faceFreeHead = mesh.faces[face].next;

    uint32_t prev_prev = mesh.faces[0].prev;
    mesh.faces[0].prev = face;
    mesh.faces[prev_prev].next = face;

    mesh.faces[face].next = 0;
    mesh.faces[face].prev = prev_prev;

    mesh.numFaces += -1;

    return face;
  }

  void deleteMeshFace(EditMesh &mesh, uint32_t face)
  {
    uint32_t next = mesh.faces[face].next;
    uint32_t prev = mesh.faces[face].prev;

    mesh.faces[prev].next = next;
    mesh.faces[next].prev = prev;

    uint32_t old_head = mesh.faceFreeHead;
    mesh.faceFreeHead = face;
    mesh.faces[face].next = old_head;
  }

  uint32_t allocMeshVert(EditMesh &mesh)
  {
    uint32_t vert = mesh.vertFreeHead;
    assert(vert != 0);
    mesh.vertFreeHead = mesh.verts[vert].next;

    return vert;
  }

  void freeMeshVert(EditMesh &mesh, uint32_t vert)
  {
    uint32_t old_head = mesh.vertFreeHead;
    mesh.vertFreeHead = vert;
    mesh.verts[vert].next = old_head;
  }

  uint32_t addVertToMesh(EditMesh &mesh, uint32_t vert)
  {
    uint32_t prev_prev = mesh.verts[0].prev;
    mesh.verts[0].prev = vert;
    mesh.verts[prev_prev].next = vert;

    mesh.verts[vert].next = 0;
    mesh.verts[vert].prev = prev_prev;

    mesh.numVerts += 1;

    return vert;
  }

  void removeVertFromMesh(EditMesh &mesh, uint32_t vert)
  {
    uint32_t next = mesh.verts[vert].next;
    uint32_t prev = mesh.verts[vert].prev;

    mesh.verts[prev].next = next;
    mesh.verts[next].prev = prev;

    mesh.numVerts -= 1;
  }

  uint32_t addConflictVert(HullBuildData &hull_data,
      uint32_t face,
      Vector3 pos)
  {
    auto &mesh = hull_data.mesh;
    uint32_t vert = allocMeshVert(mesh);

    uint32_t next = hull_data.faceConflictLists[face];

    hull_data.faceConflictLists[face] = vert;
    mesh.verts[vert].next = next;
    mesh.verts[vert].prev = 0;

    if (next != 0) {
      mesh.verts[next].prev = vert;
    }

    mesh.verts[vert].pos = pos;

    return vert;
  }

  void removeConflictVert(HullBuildData &hull_data,
      uint32_t face,
      uint32_t vert)
  {
    auto &mesh = hull_data.mesh;

    uint32_t next = mesh.verts[vert].next;
    uint32_t prev = mesh.verts[vert].prev;

    if (prev == 0) {
      hull_data.faceConflictLists[face] = next;
    } else {
      mesh.verts[prev].next = next;
    }

    if (next != 0) {
      mesh.verts[next].prev = prev;
    }
  }

  // Gregorious, Implementing QuickHull, GDC 2014, Slide 77
  float computePlaneEpsilon(Span<const Vector3> verts)
  {
    AABB aabb = AABB::invalid();

    for (Vector3 v : verts) {
      aabb.expand(v);
    }

    Vector3 diff = aabb.pMax - aabb.pMin;

    return 3.f * (diff.x + diff.y + diff.z) * FLT_EPSILON;
  }

  // RTCD 12.4.2
  template <typename Fn>
  Plane computeNewellPlaneImpl(Fn &&iter_verts)
  {
    Vector3 centroid { 0, 0, 0 };
    Vector3 n { 0, 0, 0 };

    int64_t num_verts = 0;
    // Compute normal as being proportional to projected areas of polygon
    // onto the yz, xz, and xy planes. Also compute centroid as
    // representative point on the plane
    iter_verts([&centroid, &n, &num_verts](Vector3 vi, Vector3 vj) {
        n.x += (vi.y - vj.y) * (vi.z + vj.z); // projection on yz
        n.y += (vi.z - vj.z) * (vi.x + vj.x); // projection on xz
        n.z += (vi.x - vj.x) * (vi.y + vj.y); // projection on xy

        centroid += vj;
        num_verts += 1;
        });

    assert(num_verts != 0);

    centroid /= num_verts;

    n = normalize(n);
    return Plane {
      .normal = n,
        .d = dot(centroid, n),
    };
  }

  Plane computeNewellPlane(const Vector3 *verts,
      Span<const uint32_t> indices)
  {
    return computeNewellPlaneImpl([verts, indices](auto &&fn) {
        for (int64_t i = indices.size() - 1, j = 0; j < indices.size();
            i = j, j++) {
        Vector3 vi = verts[indices[i]];
        Vector3 vj = verts[indices[j]];

        fn(vi, vj);
        }
        });
  }

  Plane computeNewellPlane(EditMesh &mesh, uint32_t face)
  {
    return computeNewellPlaneImpl([mesh, face](auto &&fn) {
      uint32_t start_hedge_idx = mesh.faces[face].hedge;
      uint32_t cur_hedge_idx = start_hedge_idx;

      do {
        const EditMesh::HEdge &cur_hedge = mesh.hedges[cur_hedge_idx];
        uint32_t next_hedge_idx = cur_hedge.next;
        const EditMesh::HEdge &next_hedge = mesh.hedges[next_hedge_idx];

        uint32_t i = cur_hedge.vert;
        uint32_t j = next_hedge.vert;

        fn(mesh.verts[i].pos, mesh.verts[j].pos);

        cur_hedge_idx = next_hedge_idx;
      } while (cur_hedge_idx != start_hedge_idx);
    });
  }

  float distToPlane(Plane plane, Vector3 v)
  {
    return v.dot(plane.normal) - plane.d;
  }

  HullBuildData allocBuildData(const int64_t N, Runtime &rt)
  {
    // + 1 for fake starting point for linked lists
    const int64_t max_num_verts = N + 1;
    // Num edges = 3N - 6. Doubled for half edges, doubled for horizon
    const int64_t max_num_hedges = 4 * (3 * N - 6) + 1;
    // Num edges = 2N - 4. Doubled for horizon
    const int64_t max_num_faces = 2 * (2 * N - 4) + 1;

    const auto buffer_sizes = std::to_array({
        int64_t(sizeof(EditMesh::HEdge) * max_num_hedges), // hedges
        int64_t(sizeof(EditMesh::Face) * max_num_faces), // faces
        int64_t(sizeof(EditMesh::Vert) * max_num_verts), // verts
        int64_t(sizeof(uint32_t) * max_num_faces), // faceConflictLists
        });

    constexpr int64_t sub_buffer_alignment = 128;

    int64_t buffer_offsets[buffer_sizes.size() - 1];
    int64_t total_bytes = computeBufferOffsets(
        buffer_sizes, buffer_offsets, sub_buffer_alignment);

    char *buf_base =
      (char *)rt.tmpAlloc(total_bytes, sub_buffer_alignment);

    EditMesh mesh {
      .hedges = (EditMesh::HEdge *)(buf_base),
      .faces = (EditMesh::Face *)(buf_base + buffer_offsets[0]),
      .verts = (EditMesh::Vert *)(buf_base + buffer_offsets[1]),
      .numHedges = 0,
      .numFaces = 0,
      .numVerts = 0,
      .hedgeFreeHead = 1,
      .faceFreeHead = 1,
      .vertFreeHead = 1,
    };

    // Setup free lists
    for (int64_t i = 1; i < max_num_hedges; i++) {
      mesh.hedges[i].next = uint32_t(i + 1);
    }
    mesh.hedges[max_num_hedges].next = 0;

    for (int64_t i = 1; i < max_num_faces; i++) {
      mesh.faces[i].next = uint32_t(i + 1);
    }
    mesh.faces[max_num_faces].next = 0;

    for (int64_t i = 1; i < max_num_verts; i++) {
      mesh.verts[i].next = uint32_t(i + 1);
    }
    mesh.verts[max_num_verts].next = 0;

    // Elem 0 is fake head / tail to avoid special cases
    mesh.hedges[0].next = 0;
    mesh.hedges[0].prev = 0;

    mesh.faces[0].next = 0;
    mesh.faces[0].prev = 0;

    mesh.verts[0].next = 0;
    mesh.verts[0].prev = 0;

    uint32_t *face_conflict_lists = (uint32_t *)(buf_base + buffer_offsets[2]);
    for (int64_t i = 0; i < max_num_faces; i++) {
      face_conflict_lists[i] = 0;
    }

    return HullBuildData {
      .mesh = mesh,
      .faceConflictLists = face_conflict_lists,
    };
  }

  bool initHullTetrahedron(EditMesh &mesh,
      Span<const Vector3> verts,
      float epsilon,
      uint32_t *tet_fids,
      Plane *tet_face_planes)
  {
    // Choose the initial 4 points for the hull
    Vector3 v0 = verts[0];

    Vector3 v1, e1;
    float max_v1_dist = -FLT_MAX;
    for (int64_t i = 1; i < verts.size(); i++) {
      Vector3 v = verts[i];
      Vector3 e = v - v0;
      float e_len = e.length();
      if (e_len > max_v1_dist) {
        v1 = v;
        e1 = e;
        max_v1_dist = e_len;
      }
    }

    if (max_v1_dist < epsilon) {
      return false;
    }

    Vector3 v2, e2;
    float max_v2_area = -FLT_MAX;
    for (int64_t i = 1; i < verts.size(); i++) {
      Vector3 v = verts[i];
      Vector3 e = v - v0;

      float area = cross(e, e1).length();

      if (area > max_v2_area) {
        v2 = v;
        e2 = e;
        max_v2_area = area;
      }
    }

    if (max_v2_area < epsilon) {
      return false;
    }

    Vector3 v3;
    float max_v3_det = -FLT_MAX;
    for (int64_t i = 1; i < verts.size(); i++) {
      Vector3 v = verts[i];
      Vector3 e = v - v0;

      Mat3x3 vol_mat {{ e1, e2, e }};
      float det = vol_mat.determinant();

      if (det > max_v3_det) {
        v3 = v;
        max_v3_det = det;
      }
    }

    if (max_v3_det < epsilon) {
      return false;
    }

    // Setup initial halfedge mesh
    uint32_t vids[4];
    vids[0] = allocMeshVert(mesh);
    vids[1] = allocMeshVert(mesh);
    vids[2] = allocMeshVert(mesh);
    vids[3] = allocMeshVert(mesh);
    addVertToMesh(mesh, vids[0]);
    addVertToMesh(mesh, vids[1]);
    addVertToMesh(mesh, vids[2]);
    addVertToMesh(mesh, vids[3]);
    mesh.verts[vids[0]].pos = v0;
    mesh.verts[vids[1]].pos = v1;
    mesh.verts[vids[2]].pos = v2;
    mesh.verts[vids[3]].pos = v3;

    // Face 0:
    //   he0: 3 => 2, he1: 2 => 1, he2: 1 => 3,
    // Face 1:
    //   he3: 2 => 3, he4: 3 => 0, he5: 0 => 2,
    // Face 2:
    //   he6: 1 => 0, he7: 0 => 3, he8: 3 => 1,
    // Face 3:
    //   he9: 0 => 1, he10: 1 => 2, he11: 2 => 0,
    uint32_t eids[12];
    const uint32_t face_vert_indices[] = {
      3, 2, 1,
      2, 3, 0,
      1, 0, 3,
      0, 1, 2
    };

    const uint32_t twin_hedge_indices[] = {
      3, 10, 8,
      0, 7, 11,
      9, 4, 2,
      6, 1, 5,
    };

    // Allocate half edges
#pragma unroll
    for (int64_t i = 0; i < 12; i++) {
      eids[i] = allocMeshHedge(mesh);
    }

    // Create faces and create halfedges
    for (int64_t i = 0; i < 4; i++) {
      const uint32_t base_hedge_offset = i * 3;
      uint32_t fid = tet_fids[i] = createMeshFace(mesh);

#pragma unroll
      for (int64_t j = 0; j < 3; j++) {
        const uint32_t cur_hedge_offset = base_hedge_offset + j;
        const uint32_t next_hedge_offset = base_hedge_offset + ((j + 1) % 3);
        const uint32_t prev_hedge_offset = base_hedge_offset + ((j + 2) % 3);

        uint32_t vid = vids[face_vert_indices[cur_hedge_offset]];
        uint32_t cur_eid = eids[cur_hedge_offset];

        mesh.hedges[cur_eid].face = fid;
        mesh.hedges[cur_eid].vert = vid;

        mesh.hedges[cur_eid].next = eids[next_hedge_offset];
        mesh.hedges[cur_eid].prev = eids[prev_hedge_offset];

        mesh.hedges[cur_eid].twin = twin_hedge_indices[cur_hedge_offset];
      }

      mesh.faces[fid].hedge = eids[base_hedge_offset];

      Plane face_plane = computeNewellPlane(mesh, fid);
      mesh.faces[fid].plane = tet_face_planes[i] = face_plane;
    }

    return true;
  }

  bool initHullBuild(Span<const Vector3> verts,
      HullBuildData *out,
      Runtime &rt)
  {
    if (verts.size() < 4) {
      return false;
    }

    *out = allocBuildData(verts.size(), rt);
    EditMesh &mesh = out->mesh;

    float epsilon = computePlaneEpsilon(verts);

    uint32_t tet_face_ids[4];
    Plane tet_face_planes[4];
    // FIXME: choose proper epsilon not just plane epsilon
    bool tet_success = initHullTetrahedron(mesh, verts, epsilon, tet_face_ids,
        tet_face_planes);
    if (!tet_success) {
      return false;
    }

    // Initial vertex binning
    for (Vector3 pos : verts) {
      float closest_plane_dist = FLT_MAX;
      int64_t closest_plane_idx = -1;
      for (int64_t i = 0; i < 4; i++) {
        Plane cur_plane = tet_face_planes[i];
        float dist = distToPlane(cur_plane, pos);

        if (dist > epsilon) {
          if (dist < closest_plane_dist) {
            closest_plane_idx = i;
            closest_plane_dist = dist;
          }
        }
      }

      // This is an internal vertex
      if (closest_plane_idx == -1) {
        continue;
      }

      addConflictVert(*out, tet_face_ids[closest_plane_idx], pos);
    }

    return true;
  }

  void quickhullBuild(HullBuildData &build_data)
  {
    (void)build_data;
#if 0
    auto &mesh = build_data.mesh;
    // FIXME
    (void)mesh;
    (void)freeMeshHedge;
    (void)deleteMeshFace;
    (void)freeMeshVert;
    (void)removeVertFromMesh;
    (void)removeConflictVert;
#endif
  }

  HalfEdgeMesh editMeshToRuntimeMesh(
      EditMesh &edit_mesh,
      Runtime &rt)
  {
    uint32_t *hedge_remap = rt.tmpAllocN<uint32_t>(edit_mesh.numHedges);
    uint32_t *face_remap = rt.tmpAllocN<uint32_t>(edit_mesh.numFaces);
    uint32_t *vert_remap = rt.tmpAllocN<uint32_t>(edit_mesh.numVerts);

    for (int64_t i = 0; i < edit_mesh.numHedges; i++) {
      hedge_remap[i] = 0xFFFF'FFFF;
    }

    int64_t num_new_hedges = 0;
    for (uint32_t orig_eid = edit_mesh.hedges[0].next;
        orig_eid != 0; orig_eid = edit_mesh.hedges[orig_eid].next) {
      if (hedge_remap[orig_eid] != 0xFFFF'FFFF) {
        continue;
      }

      const EditMesh::HEdge &cur_hedge = edit_mesh.hedges[orig_eid];
      uint32_t twin_eid = cur_hedge.twin;
      assert(hedge_remap[twin_eid] = 0xFFFF'FFFF);

      hedge_remap[orig_eid] = num_new_hedges;
      hedge_remap[twin_eid] = num_new_hedges + 1;
      num_new_hedges += 2;
    }

    int64_t num_new_verts = 0;
    for (uint32_t orig_vid = edit_mesh.verts[0].next;
        orig_vid != 0; orig_vid = edit_mesh.verts[orig_vid].next) {
      vert_remap[orig_vid] = num_new_verts++;
    }

    int64_t num_new_faces = 0;
    for (uint32_t orig_fid = edit_mesh.faces[0].next;
        orig_fid != 0; orig_fid = edit_mesh.faces[orig_fid].next) {
      face_remap[orig_fid] = num_new_faces++;
    }

    auto hedges_out = rt.tmpAllocN<HalfEdge>(num_new_hedges);
    auto face_base_hedges_out = rt.tmpAllocN<uint32_t>(num_new_faces);
    auto face_planes_out = rt.tmpAllocN<Plane>(num_new_faces);
    auto positions_out = rt.tmpAllocN<Vector3>(num_new_verts);

    for (uint32_t orig_eid = edit_mesh.hedges[0].next;
        orig_eid != 0; orig_eid = edit_mesh.hedges[orig_eid].next) {
      const EditMesh::HEdge &orig_hedge = edit_mesh.hedges[orig_eid];

      hedges_out[hedge_remap[orig_eid]] = HalfEdge {
        .next = hedge_remap[orig_hedge.next],
        .rootVertex = vert_remap[orig_hedge.vert],
        .face = face_remap[orig_hedge.face],
      };
    }

    for (uint32_t orig_vid = edit_mesh.verts[0].next;
        orig_vid != 0; orig_vid = edit_mesh.verts[orig_vid].next) {
      const EditMesh::Vert &orig_vert = edit_mesh.verts[orig_vid];
      positions_out[vert_remap[orig_vid]] = orig_vert.pos;
    }

    for (uint32_t orig_fid = edit_mesh.faces[0].next;
        orig_fid != 0; orig_fid = edit_mesh.faces[orig_fid].next) {
      const EditMesh::Face &orig_face = edit_mesh.faces[orig_fid];

      uint32_t new_face_idx = face_remap[orig_fid];

      face_base_hedges_out[new_face_idx] = hedge_remap[orig_face.hedge];
      face_planes_out[new_face_idx] = orig_face.plane;
    }

    return HalfEdgeMesh {
      .halfEdges = hedges_out,
      .faceBaseHalfEdges = face_base_hedges_out,
      .facePlanes = face_planes_out,
      .vertices = positions_out,
      .numHalfEdges = uint32_t(num_new_hedges),
      .numFaces = uint32_t(num_new_faces),
      .numVertices = uint32_t(num_new_verts),
    };
  }

  inline HalfEdgeMesh buildHalfEdgeMesh(
      const SourceMesh &src_mesh,
      Runtime &rt)
  {
    auto numFaceVerts = [&src_mesh](int64_t face_idx) {
      if (src_mesh.faceCounts == nullptr) {
        return 3_u32;
      } else {
        return src_mesh.faceCounts[face_idx];
      }
    };

    uint32_t num_hedges = 0;
    for (int64_t face_idx = 0; face_idx < (int64_t)src_mesh.numFaces;
        face_idx++) {
      num_hedges += numFaceVerts(face_idx);
    }

    assert(num_hedges % 2 == 0);

    // We already know how many polygons there are
    auto hedges_out = rt.tmpAllocN<HalfEdge>(num_hedges * 2);
    auto face_base_hedges_out = rt.tmpAllocN<uint32_t>(src_mesh.numFaces * 2);
    auto face_planes_out = rt.tmpAllocN<Plane>(src_mesh.numFaces * 2);

    std::unordered_map<uint64_t, uint32_t> edge_to_hedge;

    auto makeEdgeID = [](uint32_t a_idx, uint32_t b_idx) {
      return ((uint64_t)a_idx << 32) | (uint64_t)b_idx;
    };

    int64_t num_assigned_hedges = 0;
    const uint32_t *cur_face_indices = src_mesh.indices;
    for (int64_t face_idx = 0; face_idx < (int64_t)src_mesh.numFaces;
        face_idx++) {
      int64_t num_face_vertices = numFaceVerts(face_idx);

      Plane face_plane = computeNewellPlane(src_mesh.positions,
          Span(cur_face_indices, num_face_vertices));

      assert(face_idx < src_mesh.numFaces);
      face_planes_out[face_idx] = face_plane;

      for (int64_t vert_offset = 0; vert_offset < num_face_vertices;
          vert_offset++) {
        uint32_t a_idx = cur_face_indices[vert_offset];
        uint32_t b_idx = cur_face_indices[
          (vert_offset + 1) % num_face_vertices];

        uint64_t cur_edge_id = makeEdgeID(a_idx, b_idx);

        auto cur_edge_lookup = edge_to_hedge.find(cur_edge_id);
        if (cur_edge_lookup == edge_to_hedge.end()) {
          uint32_t cur_hedge_id = num_assigned_hedges;
          uint32_t twin_hedge_id = num_assigned_hedges + 1;

          num_assigned_hedges += 2;

          uint64_t twin_edge_id = makeEdgeID(b_idx, a_idx);

          auto [new_edge_iter, cur_inserted] =
            edge_to_hedge.emplace(cur_edge_id, cur_hedge_id);
          assert(cur_inserted);

          auto [new_twin_iter, twin_inserted] =
            edge_to_hedge.emplace(twin_edge_id, twin_hedge_id);
          assert(twin_inserted);

          cur_edge_lookup = new_edge_iter;
        }

        uint32_t hedge_idx = cur_edge_lookup->second;
        if (vert_offset == 0) {
          assert(face_idx < src_mesh.numFaces);
          face_base_hedges_out[face_idx] = hedge_idx;
        }

        uint32_t c_idx = cur_face_indices[
          (vert_offset + 2) % num_face_vertices];

        auto next_edge_id = makeEdgeID(b_idx, c_idx);
        auto next_edge_lookup = edge_to_hedge.find(next_edge_id);

        // If next doesn't exist yet, we can assume it will be the next
        // allocated half edge
        uint32_t next_hedge_idx = next_edge_lookup == edge_to_hedge.end() ?
          num_assigned_hedges : next_edge_lookup->second;

        assert(hedge_idx < num_hedges);
        hedges_out[hedge_idx] = HalfEdge {
          .next = next_hedge_idx,
          .rootVertex = a_idx,
          .face = uint32_t(face_idx),
        };
      }

      cur_face_indices += num_face_vertices;
    }

    assert(num_assigned_hedges == num_hedges);

    return HalfEdgeMesh {
      .halfEdges = hedges_out,
      .faceBaseHalfEdges = face_base_hedges_out,
      .facePlanes = face_planes_out,
      .vertices = src_mesh.positions,
      .numHalfEdges = uint32_t(num_hedges),
      .numFaces = src_mesh.numFaces,
      .numVertices = src_mesh.numVertices,
    };
  }

  bool processConvexHull(
      const SourceMesh &src_mesh,
      bool build_hull,
      HalfEdgeMesh *out_mesh,
      Runtime &rt)
  {
    if (!build_hull) {
      // Just assume the input geometry is a convex hull with coplanar faces
      // merged
      *out_mesh = buildHalfEdgeMesh(src_mesh, rt);
    } else {
      HullBuildData hull_data;
      bool valid_input = initHullBuild(
          Span(src_mesh.positions, src_mesh.numVertices),
          &hull_data, rt);

      if (!valid_input) {
        return false;
      }

      quickhullBuild(hull_data);

      *out_mesh = editMeshToRuntimeMesh(hull_data.mesh, rt);
    }

    return true;
  }

  bool processConvexHulls(bool build_convex_hulls,
                          HalfEdgeMesh *out_meshes,
                          Runtime &rt)
  {
    auto &in_meshes = imported->srcHulls;
    for (int64_t hull_idx = 0; hull_idx < (int64_t)in_meshes.size(); hull_idx++) {
      const SourceMesh &mesh = in_meshes[hull_idx];

      bool success = processConvexHull(
          mesh, build_convex_hulls, &out_meshes[hull_idx], rt);

      if (!success) {
        return false;
      }
    }

    return true;
  }

  void setupSpherePrimitive(const SourceCollisionPrimitive &src_prim,
      CollisionPrimitive *out_prim,
      AABB *out_aabb)
  {
    out_prim->sphere = src_prim.sphere;

    const float r = src_prim.sphere.radius;

    *out_aabb = AABB {
      .pMin = { -r, -r, -r },
        .pMax = { r, r, r },
    };
  }

  void setupCapsulePrimitive(const SourceCollisionPrimitive &src_prim,
      CollisionPrimitive *out_prim,
      AABB *out_aabb)
  {
    out_prim->capsule = src_prim.capsule;

    const float r = src_prim.capsule.radius;
    const float h = src_prim.capsule.cylinderHeight * 0.5f;

    *out_aabb = AABB {
      .pMin = { -r, -r, -(r + h) },
        .pMax = { r, r, r + h },
    };
  }

  void setupBoxPrimitive(const SourceCollisionPrimitive &src_prim,
      CollisionPrimitive *out_prim,
      AABB *out_aabb)
  {
    out_prim->box = src_prim.box;
    Vector3 dimHalf = src_prim.box.dim * 0.5f;

    *out_aabb = AABB {
      .pMin = { -dimHalf.x, -dimHalf.y, -dimHalf.z },
        .pMax = { dimHalf.x, dimHalf.y, dimHalf.z },
    };
  }

  void setupPlanePrimitive(const SourceCollisionPrimitive &,
      CollisionPrimitive *out_prim,
      AABB *out_aabb)
  {
    out_prim->plane = CollisionPrimitive::Plane {};

    *out_aabb = AABB {
      .pMin = { -FLT_MAX, -FLT_MAX, -FLT_MAX },
        .pMax = { FLT_MAX, FLT_MAX, 0 },
    };
  }

  void setupHullPrimitive(const SourceCollisionPrimitive &src_prim,
      const HalfEdgeMesh *hull_meshes,
      CollisionPrimitive *out_prim,
      AABB *out_aabb)
  {
    const HalfEdgeMesh &hull_mesh = hull_meshes[src_prim.hullInput.hullIDX];

    AABB mesh_aabb = AABB::point(hull_mesh.vertices[0]);
    for (CountT vert_idx = 1; vert_idx < (CountT)hull_mesh.numVertices;
        vert_idx++) {
      mesh_aabb.expand(hull_mesh.vertices[vert_idx]);
    }

    out_prim->hull.halfEdgeMesh = hull_mesh;
    *out_aabb = mesh_aabb;
  }

  void setupRigidBodyAABBsAndPrimitives(
      HalfEdgeMesh *hull_meshes,
      Span<const SourceCollisionObject> collision_objs,
      CollisionPrimitive *out_prims,
      AABB *out_prim_aabbs,
      AABB *out_obj_aabbs,
      uint32_t *out_prim_offsets,
      uint32_t *out_prim_counts)
  {
    using Type = CollisionPrimitive::Type;

    uint32_t cur_prim_offset = 0;
    for (CountT obj_idx = 0; obj_idx < collision_objs.size(); obj_idx++) {
      const SourceCollisionObject &collision_obj = collision_objs[obj_idx];

      CountT num_prims = collision_obj.prims.size();
      CollisionPrimitive *obj_prims = out_prims + cur_prim_offset;
      AABB *prim_aabbs = out_prim_aabbs + cur_prim_offset;

      auto obj_aabb = AABB::invalid();

      for (CountT prim_idx = 0; prim_idx < num_prims; prim_idx++) {
        const SourceCollisionPrimitive &src_prim =
          collision_obj.prims[prim_idx];

        CollisionPrimitive *out_prim = &obj_prims[prim_idx];
        out_prim->type = src_prim.type;
        AABB prim_aabb;

        switch (src_prim.type) {
          case Type::Sphere: {
            setupSpherePrimitive(src_prim, out_prim, &prim_aabb);
          } break;
          case Type::Plane: {
            setupPlanePrimitive(src_prim, out_prim, &prim_aabb);
          } break;
          case Type::Hull: {
            setupHullPrimitive(src_prim, hull_meshes,
                out_prim, &prim_aabb);
          } break;
          case Type::Capsule: {
            setupCapsulePrimitive(src_prim, out_prim, &prim_aabb);
          } break;
          case Type::Box: {
            setupBoxPrimitive(src_prim, out_prim, &prim_aabb);
          } break;
        }

        prim_aabbs[prim_idx] = prim_aabb;
        obj_aabb = AABB::merge(obj_aabb, prim_aabb);
      }

      out_obj_aabbs[obj_idx] = obj_aabb;
      out_prim_offsets[obj_idx] = cur_prim_offset;
      out_prim_counts[obj_idx] = (uint32_t)num_prims;

      cur_prim_offset += (uint32_t)num_prims;
    }
  }

  ProcessedPhysicsAssets process(
      bool build_convex_hulls,
      Runtime &rt)
  {
    ArenaRegion tmp_region = rt.beginTmpRegion();

    HalfEdgeMesh *built_hulls = rt.tmpAllocN<HalfEdgeMesh>(
        imported->srcHulls.size());

    ArenaRegion hull_build_region = rt.beginTmpRegion();

    bool hull_success = processConvexHulls(
        build_convex_hulls, built_hulls, rt);

    if (!hull_success) {
      free(built_hulls);

      rt.endTmpRegion(hull_build_region);
      rt.endTmpRegion(tmp_region);

      return {};
    }

    auto &collision_objs = imported->objs;

    int64_t total_num_prims = 0;
    for (int64_t obj_idx = 0; obj_idx < (int64_t)collision_objs.size(); obj_idx++) {
      const SourceCollisionObject &collision_obj = collision_objs[obj_idx];
      int64_t cur_num_prims = collision_obj.prims.size();
      total_num_prims += cur_num_prims;
    }

    int64_t total_num_halfedges = 0;
    int64_t total_num_faces = 0;
    int64_t total_num_verts = 0;

    auto &convex_hull_meshes = imported->srcHulls;

    for (int64_t hull_idx = 0; hull_idx < (int64_t)convex_hull_meshes.size();
        hull_idx++) {
      const HalfEdgeMesh &hull_mesh = built_hulls[hull_idx];

      total_num_halfedges += hull_mesh.numHalfEdges;
      total_num_faces += hull_mesh.numFaces;
      total_num_verts += hull_mesh.numVertices;
    }

    auto buffer_sizes = std::to_array<int64_t>({
      (int64_t)sizeof(HalfEdge) * total_num_halfedges, // halfEdges
      (int64_t)sizeof(uint32_t) * total_num_faces, // faceBaseHalfEdges
      (int64_t)sizeof(Plane) * total_num_faces, // facePlanes
      (int64_t)sizeof(Vector3) * total_num_verts, // vertices
      (int64_t)sizeof(CollisionPrimitive) * total_num_prims, // prims
      (int64_t)sizeof(AABB) * total_num_prims, // primAABBs
      (int64_t)sizeof(AABB) * (int64_t)collision_objs.size(), // obj_aabbs
      (int64_t)sizeof(uint32_t) * (int64_t)collision_objs.size(), // prim_offsets
      (int64_t)sizeof(uint32_t) * (int64_t)collision_objs.size(), // prim_counts
    });

    int64_t buffer_offsets[buffer_sizes.size() - 1];
    int64_t num_buffer_bytes = computeBufferOffsets(
        buffer_sizes, buffer_offsets, 64);

    char *buffer = (char *)malloc(num_buffer_bytes);
    ProcessedPhysicsAssets assets {
      .buffer = buffer,
      .bufferSize = (uint64_t)num_buffer_bytes,
      .hullData = {
        .halfEdges = (HalfEdge *)buffer,
        .faceBaseHalfEdges = (uint32_t *)(buffer + buffer_offsets[0]),
        .facePlanes = (Plane *)(buffer + buffer_offsets[1]),
        .vertices = (Vector3 *)(buffer + buffer_offsets[2]),
        .numHalfEdges = (uint32_t)total_num_halfedges,
        .numFaces = (uint32_t)total_num_faces,
        .numVerts = (uint32_t)total_num_verts,
      },
      .primitives = (CollisionPrimitive *)(buffer + buffer_offsets[3]),
      .primitiveAABBs = (AABB *)(buffer + buffer_offsets[4]),
      .objAABBs = (AABB *)(buffer + buffer_offsets[5]),
      .primOffsets = (uint32_t *)(buffer + buffer_offsets[6]),
      .primCounts = (uint32_t *)(buffer + buffer_offsets[7]),
      .numConvexHulls = (uint32_t)convex_hull_meshes.size(),
      .totalNumPrimitives = (uint32_t)total_num_prims,
      .numObjs = (uint32_t)collision_objs.size(),
    };

    int64_t cur_halfedge_offset = 0;
    int64_t cur_face_offset = 0;
    int64_t cur_vert_offset = 0;
    for (int64_t hull_idx = 0; hull_idx < (int64_t)convex_hull_meshes.size();
        hull_idx++) {
      HalfEdgeMesh &hull_mesh = built_hulls[hull_idx];

      HalfEdge *he_out = &assets.hullData.halfEdges[cur_halfedge_offset];
      uint32_t *face_bases_out =
        &assets.hullData.faceBaseHalfEdges[cur_face_offset];
      Plane *face_planes_out = &assets.hullData.facePlanes[cur_face_offset];
      Vector3 *verts_out = &assets.hullData.vertices[cur_vert_offset];

      memcpy(he_out, hull_mesh.halfEdges,
          sizeof(HalfEdge) * hull_mesh.numHalfEdges);
      memcpy(face_bases_out, hull_mesh.faceBaseHalfEdges,
          sizeof(uint32_t) * hull_mesh.numFaces);
      memcpy(face_planes_out, hull_mesh.facePlanes,
          sizeof(Plane) * hull_mesh.numFaces);
      memcpy(verts_out, hull_mesh.vertices,
          sizeof(Vector3) * hull_mesh.numVertices);

      hull_mesh.halfEdges = he_out;
      hull_mesh.faceBaseHalfEdges = face_bases_out;
      hull_mesh.facePlanes = face_planes_out;
      hull_mesh.vertices = verts_out;

      cur_halfedge_offset += hull_mesh.numHalfEdges;
      cur_face_offset += hull_mesh.numFaces;
      cur_vert_offset += hull_mesh.numVertices;
    }

    rt.endTmpRegion(hull_build_region);

    setupRigidBodyAABBsAndPrimitives(built_hulls,
        collision_objs,
        assets.primitives,
        assets.primitiveAABBs,
        assets.objAABBs,
        assets.primOffsets,
        assets.primCounts);

    rt.endTmpRegion(tmp_region);

    return assets;
  }

  PhysicsBridgeData allocBridge(
      Backend *backend,
      ProcessedPhysicsAssets &processed_assets)
  {
    u32 max_prims_per_object = 0;
    u32 max_objects = processed_assets.numObjs;
    for (u32 i = 0; i < processed_assets.numObjs; ++i) {
      max_prims_per_object = std::max(
          max_prims_per_object, processed_assets.primCounts[i]);
    }

    size_t num_collision_prim_bytes =
      sizeof(CollisionPrimitive) * max_objects * max_prims_per_object; 

    size_t num_collision_aabb_bytes =
      sizeof(AABB) * max_objects * max_prims_per_object; 

    size_t num_obj_aabb_bytes =
      sizeof(AABB) * max_objects;

    size_t num_offset_bytes =
      sizeof(uint32_t) * max_objects;

    size_t num_count_bytes =
      sizeof(uint32_t) * max_objects;

    CollisionPrimitive *primitives_ptr;
    AABB *prim_aabb_ptr;

    AABB *obj_aabb_ptr;
    uint32_t *offsets_ptr;
    uint32_t *counts_ptr;

    ObjectManager *mgr;

    if (backendGPUID(backend) == -1) {
      primitives_ptr = (CollisionPrimitive *)malloc(
          num_collision_prim_bytes);

      prim_aabb_ptr = (AABB *)malloc(
          num_collision_aabb_bytes);

      obj_aabb_ptr = (AABB *)malloc(num_obj_aabb_bytes);

      offsets_ptr = (uint32_t *)malloc(num_offset_bytes);
      counts_ptr = (uint32_t *)malloc(num_count_bytes);

      mgr = new ObjectManager {
        primitives_ptr,
        prim_aabb_ptr,
        obj_aabb_ptr,
        offsets_ptr,
        counts_ptr,
      };
    } else {
#ifndef BOT_CUDA_SUPPORT
      FATAL("Not compiled with CUDA support");
#else
      primitives_ptr = (CollisionPrimitive *)allocGPU(
          num_collision_prim_bytes);

      prim_aabb_ptr = (AABB *)allocGPU(
          num_collision_aabb_bytes);

      obj_aabb_ptr = (AABB *)allocGPU(num_obj_aabb_bytes);

      offsets_ptr = (uint32_t *)allocGPU(num_offset_bytes);
      counts_ptr = (uint32_t *)allocGPU(num_count_bytes);

      mgr = (ObjectManager *)allocGPU(sizeof(ObjectManager));

      ObjectManager local {
        primitives_ptr,
        prim_aabb_ptr,
        obj_aabb_ptr,
        offsets_ptr,
        counts_ptr,
      };

      REQ_CUDA(cudaMemcpy(mgr, &local, sizeof(ObjectManager),
            cudaMemcpyHostToDevice));
#endif
    }

    return PhysicsBridgeData {
      .primitives = primitives_ptr,
      .primAABBs = prim_aabb_ptr,
      .objAABBs = obj_aabb_ptr,
      .rigidBodyPrimitiveOffsets = offsets_ptr,
      .rigidBodyPrimitiveCounts = counts_ptr,
      .curPrimOffset = 0,
      .curObjOffset = 0,
      .mgr = mgr,
      .maxPrims = max_objects * max_prims_per_object,
      .maxObjs = max_objects,
      .gpuID = backendGPUID(backend),
    };
  }

  BridgeData<ObjectManager> makeBridge(
      Backend *backend, ProcessedPhysicsAssets &assets)
  {
    PhysicsBridgeData bridge_data = allocBridge(backend, assets);

    CountT cur_obj_offset = bridge_data.curObjOffset;
    bridge_data.curObjOffset += assets.numObjs;
    CountT cur_prim_offset = bridge_data.curPrimOffset;
    bridge_data.curPrimOffset += assets.totalNumPrimitives;
    assert(bridge_data.curObjOffset <= bridge_data.maxObjs);
    assert(bridge_data.curPrimOffset <= bridge_data.maxPrims);

    CollisionPrimitive *prims_dst = &bridge_data.primitives[cur_prim_offset];
    AABB *prim_aabbs_dst = &bridge_data.primAABBs[cur_prim_offset];

    AABB *obj_aabbs_dst = &bridge_data.objAABBs[cur_obj_offset];
    uint32_t *offsets_dst = &bridge_data.rigidBodyPrimitiveOffsets[cur_obj_offset];
    uint32_t *counts_dst = &bridge_data.rigidBodyPrimitiveCounts[cur_obj_offset];

    // FIXME: redo all this, leaks memory, slow, etc. Very non optimal on the
    // CPU.

    uint32_t *offsets_tmp = (uint32_t *)malloc(
        sizeof(uint32_t) * assets.numObjs);

    for (CountT i = 0; i < (CountT)assets.numObjs; i++) {
      offsets_tmp[i] = assets.primOffsets[i] + cur_prim_offset;
    }

    HalfEdge *hull_halfedges;
    uint32_t *hull_face_base_halfedges;
    Plane *hull_face_planes;
    Vector3 *hull_verts;
    
    if (bridge_data.gpuID == -1) {
      memcpy(prim_aabbs_dst, assets.primitiveAABBs,
          sizeof(AABB) * assets.totalNumPrimitives);

      memcpy(obj_aabbs_dst, assets.objAABBs,
          sizeof(AABB) * assets.numObjs);
      memcpy(offsets_dst, offsets_tmp,
          sizeof(uint32_t) * assets.numObjs);
      memcpy(counts_dst, assets.primCounts,
          sizeof(uint32_t) * assets.numObjs);

      hull_halfedges = (HalfEdge *)malloc(
          sizeof(HalfEdge) * assets.hullData.numHalfEdges);
      hull_face_base_halfedges = (uint32_t *)malloc(
          sizeof(uint32_t) * assets.hullData.numFaces);
      hull_face_planes = (Plane *)malloc(
          sizeof(Plane) * assets.hullData.numFaces);
      hull_verts = (Vector3 *)malloc(
          sizeof(Vector3) * assets.hullData.numVerts);

      memcpy(hull_halfedges, assets.hullData.halfEdges,
          sizeof(HalfEdge) * assets.hullData.numHalfEdges);
      memcpy(hull_face_base_halfedges, assets.hullData.faceBaseHalfEdges,
          sizeof(uint32_t) * assets.hullData.numFaces);
      memcpy(hull_face_planes, assets.hullData.facePlanes,
          sizeof(Plane) * assets.hullData.numFaces);
      memcpy(hull_verts, assets.hullData.vertices,
          sizeof(Vector3) * assets.hullData.numVerts);
    } else {
#ifndef BOT_CUDA_SUPPORT
      FATAL("Did not compile with CUDA support");
#else
      cudaMemcpy(prim_aabbs_dst, assets.primitiveAABBs,
          sizeof(AABB) * assets.totalNumPrimitives,
          cudaMemcpyHostToDevice);

      cudaMemcpy(obj_aabbs_dst, assets.objAABBs,
          sizeof(AABB) * assets.numObjs,
          cudaMemcpyHostToDevice);
      cudaMemcpy(offsets_dst, offsets_tmp,
          sizeof(uint32_t) * assets.numObjs,
          cudaMemcpyHostToDevice);
      cudaMemcpy(counts_dst, assets.primCounts,
          sizeof(uint32_t) * assets.numObjs,
          cudaMemcpyHostToDevice);

      hull_halfedges = (HalfEdge *)allocGPU(
          sizeof(HalfEdge) * assets.hullData.numHalfEdges);
      hull_face_base_halfedges = (uint32_t *)allocGPU(
          sizeof(uint32_t) * assets.hullData.numFaces);
      hull_face_planes = (Plane *)allocGPU(
          sizeof(Plane) * assets.hullData.numFaces);
      hull_verts = (Vector3 *)allocGPU(
          sizeof(Vector3) * assets.hullData.numVerts);

      cudaMemcpy(hull_halfedges, assets.hullData.halfEdges,
          sizeof(HalfEdge) * assets.hullData.numHalfEdges,
          cudaMemcpyHostToDevice);
      cudaMemcpy(hull_face_base_halfedges, assets.hullData.faceBaseHalfEdges,
          sizeof(uint32_t) * assets.hullData.numFaces,
          cudaMemcpyHostToDevice);
      cudaMemcpy(hull_face_planes, assets.hullData.facePlanes,
          sizeof(Plane) * assets.hullData.numFaces,
          cudaMemcpyHostToDevice);
      cudaMemcpy(hull_verts, assets.hullData.vertices,
          sizeof(Vector3) * assets.hullData.numVerts,
          cudaMemcpyHostToDevice);
#endif
    }

    auto primitives_tmp = (CollisionPrimitive *)malloc(
        sizeof(CollisionPrimitive) * assets.totalNumPrimitives);
    memcpy(primitives_tmp, assets.primitives,
        sizeof(CollisionPrimitive) * assets.totalNumPrimitives);

    for (CountT i = 0; i < (CountT)assets.totalNumPrimitives; i++) {
      CollisionPrimitive &cur_primitive = primitives_tmp[i];
      if (cur_primitive.type != CollisionPrimitive::Type::Hull) continue;

      HalfEdgeMesh &he_mesh = cur_primitive.hull.halfEdgeMesh;

      // FIXME: incoming HalfEdgeMeshes should have offsets or something
      CountT hedge_offset = he_mesh.halfEdges - assets.hullData.halfEdges;
      CountT face_offset =
        he_mesh.facePlanes - assets.hullData.facePlanes;
      CountT vert_offset = he_mesh.vertices - assets.hullData.vertices;

      he_mesh.halfEdges = hull_halfedges + hedge_offset;
      he_mesh.faceBaseHalfEdges = hull_face_base_halfedges + face_offset;
      he_mesh.facePlanes = hull_face_planes + face_offset;
      he_mesh.vertices = hull_verts + vert_offset;
    }

    if (bridge_data.gpuID == -1) {
      memcpy(prims_dst, primitives_tmp,
          sizeof(CollisionPrimitive) * assets.totalNumPrimitives);
    } else {
#ifdef BOT_CUDA_SUPPORT
      cudaMemcpy(prims_dst, primitives_tmp,
          sizeof(CollisionPrimitive) * assets.totalNumPrimitives,
          cudaMemcpyHostToDevice);
#else
      FATAL("Not compiled with CUDA support");
#endif
    }

    free(primitives_tmp);
    free(offsets_tmp);

    return backendBridgeData<ObjectManager>(backend, bridge_data.mgr);
  }
};

PhysicsAssetProcessor::PhysicsAssetProcessor(ImportedPhysicsAssets &imported)
  : impl_(Impl::make(imported))
{
}

PhysicsAssetProcessor::~PhysicsAssetProcessor()
{
  // Free the temporary data
}

BridgeData<ObjectManager> PhysicsAssetProcessor::process(
    Backend *backend,
    bool build_convex_hulls)
{
  Runtime rt = Runtime(backendRTStateHandle(backend), 0);
  ProcessedPhysicsAssets processed = impl_->process(build_convex_hulls, rt);
  return impl_->makeBridge(backend, processed);
}

}
