#include "sim.hpp"
#include "physics.hpp"

#ifdef BOT_GPU
#include <rt/gpu_prims.hpp>
#endif

namespace bot {

void updateLeaf(World *world, BodyRef ref)
{
  BVH &bvh = world->physics.bvh;
  ObjectManager &obj_mgr = *world->physics.objMgr;

  AABB obj_aabb = obj_mgr.rigidBodyAABBs[*ref.objectID];
  uint32_t prim_offset = obj_mgr.rigidBodyPrimitiveOffsets[*ref.objectID];
  uint32_t prim_count = obj_mgr.rigidBodyPrimitiveCounts[*ref.objectID];

  bool is_capsule = false;
  if (prim_count == 1 && 
      obj_mgr.collisionPrimitives[prim_offset].type == 
      CollisionPrimitive::Type::Capsule) {
    is_capsule = true;
  }

  // Check if this is a capsule. If this is a capsule, we need a special
  // way to update the AABB.
  bvh.updateLeafPosition(
      *ref.leafID, 
      *ref.position,
      *ref.rotation,
      *ref.scale,
      *ref.linVelocity,
      obj_aabb,
      is_capsule);
}

void refitTree(World *world, BodyRef ref)
{
  BVH &bvh = world->physics.bvh;
  bvh.refitLeaf(*ref.leafID, bvh.getLeafAABB(*ref.leafID));
}

void findIntersecting(SimRT &rt, World *world, BodyID id, BodyRef ref)
{
  BVH &bvh = world->physics.bvh;
  LeafID leaf_id = *ref.leafID;

  ObjectManager &obj_mgr = *(world->physics.objMgr);

  // FIXME: should have a flag for passing this
  // directly into the system
  bool a_is_static = (*ref.responseType == ResponseType::Static);

  int32_t a_obj = *ref.objectID;

  // TODO: Figure out what to do with this
  uint32_t cv_a_is_static = false;

  CountT a_num_prims = obj_mgr.rigidBodyPrimitiveCounts[a_obj];

  bvh.findLeafIntersecting(leaf_id, [&](BodyID intersecting_entity) {
      BodyRef intersecting_ref = world->physics.bodyStore.get(
          rt, intersecting_entity);

      if (id.linearID() < intersecting_entity.linearID()) {
        // TODO: Check if collisions are disabled between the two entities
#if 0
        if (PhysicsSystem::isCollisionDisabled(ctx, e, intersecting_entity)) {
          return;
        }
#endif

        // FIXME: Change this so static objects are kept in a separate BVH
        // and this check can be removed.
        if (a_is_static && *ref.responseType == ResponseType::Static) {
          return;
        }

        // We don't expand the primitive AABBs by movement (only object
        // AABBs) so we just unconditionally emit narrowphase checks
        // between each pair of primitives in the entity. Narrowphase
        // will check transformed AABBs.

        int32_t b_obj = *ref.objectID;

        // TODO: Figure out what to do with this
#if 0
        uint32_t cv_b_is_static = ctx.getDirect<RigidBodyStatic>(
            RGDCols::RigidBodyStatic, b_loc).isStatic;
#endif
        uint32_t cv_b_is_static = false;

        if (cv_a_is_static && cv_b_is_static) {
          return;
        }

        i64 b_num_prims = obj_mgr.rigidBodyPrimitiveCounts[b_obj];

        // FIXME: would be nice to be able to make N temporaries all at
        // once

        i64 total_narrowphase_checks = a_num_prims * b_num_prims;

#ifdef BOT_GPU
        static constexpr uint32_t kMaxTmpsPerWarp = 256*256;

        for (uint32_t i = 0; i < total_narrowphase_checks; i += kMaxTmpsPerWarp) {
          MidphaseCandidate cand = {
            .a = ref,
            .b = intersecting_ref,
            .offset = i,
            .count = (u32)std::min(kMaxTmpsPerWarp, 
                (u32)total_narrowphase_checks - i),
          };

          world->physics.midphaseStore.add(rt, cand);
        }
#else
        MidphaseCandidate cand = {
          .a = ref,
          .b = intersecting_ref,
          .offset = 0,
          .count = (u32)total_narrowphase_checks,
        };

        world->physics.midphaseStore.add(rt, cand);
#endif
      }
  });
}

void makeNarrowphaseCandidates(SimRT &rt, World *world, MidphaseCandidate *cand)
{
  ObjectManager &obj_mgr = *world->physics.objMgr;

  BodyRef a_ref = cand->a;
  BodyRef b_ref = cand->b;

  int32_t a_obj = *a_ref.objectID;
  // int64_t a_num_prims = obj_mgr.rigidBodyPrimitiveCounts[a_obj];

  int32_t b_obj = *b_ref.objectID;
  int64_t b_num_prims = obj_mgr.rigidBodyPrimitiveCounts[b_obj];

#if 0
  int64_t total_narrowphase_checks = a_num_prims * b_num_prims;
  (void)total_narrowphase_checks;
#endif

  const uint32_t a_prim_offset = obj_mgr.rigidBodyPrimitiveOffsets[a_obj];
  const uint32_t b_prim_offset = obj_mgr.rigidBodyPrimitiveOffsets[b_obj];

  // Log::log("cand->count = {}\n", cand->count);

#ifdef BOT_GPU
  gpu_prims::warpLoop(
      cand->count, [&](uint32_t prim_check_idx) {
#else
      for (int64_t prim_check_idx = 0;
          prim_check_idx < cand->count;
          prim_check_idx++) {
#endif
      int64_t a_prim_idx = (prim_check_idx + cand->offset) / b_num_prims;
      int64_t b_prim_idx = (prim_check_idx + cand->offset) % b_num_prims;

      // First check the primitive's AABB overlap. This used to be
      // done in narrowphase. Moving it to here to check how
      // performance compares.
      {
        uint32_t glob_a_prim_idx = a_prim_offset + a_prim_idx;
        uint32_t glob_b_prim_idx = b_prim_offset + b_prim_idx;

        const CollisionPrimitive *a_prim =
        &obj_mgr.collisionPrimitives[glob_a_prim_idx];
        const CollisionPrimitive *b_prim =
        &obj_mgr.collisionPrimitives[glob_b_prim_idx];

        uint32_t raw_type_a = static_cast<uint32_t>(a_prim->type);
        uint32_t raw_type_b = static_cast<uint32_t>(b_prim->type);

        // Swap a & b to be properly ordered based on object type
        if (raw_type_a > raw_type_b) {
          std::swap(a_ref, b_ref);
          std::swap(a_prim, b_prim);
          std::swap(glob_a_prim_idx, glob_b_prim_idx);
          std::swap(raw_type_a, raw_type_b);
        }

        const Vector3 a_pos = *a_ref.position;
        const Vector3 b_pos = *b_ref.position;
        const Quat a_rot = *a_ref.rotation;
        const Quat b_rot = *b_ref.rotation;
        const Diag3x3 a_scale = *a_ref.scale;
        const Diag3x3 b_scale = *b_ref.scale;

        {
          AABB a_obj_aabb = obj_mgr.primitiveAABBs[glob_a_prim_idx];
          AABB b_obj_aabb = obj_mgr.primitiveAABBs[glob_b_prim_idx];

          // TODO: Have a better way of handling this capsule edge case.
          AABB a_world_aabb = [&]() {
            if (raw_type_a == (uint32_t)CollisionPrimitive::Type::Capsule) {
              // assert(a_scale.d0 == a_scale.d1);

              float r = a_scale.d0;
              float half_h = a_scale.d2;

              AABB capsule_aabb = {
                .pMin = { -r, -r, -(r + half_h) },
                .pMax = { r, r, r + half_h },
              };

              return capsule_aabb.applyTRS(a_pos, a_rot, { 1.f, 1.f, 1.f });
            } else {
              return a_obj_aabb.applyTRS(a_pos, a_rot, a_scale);
            }
          } ();

          AABB b_world_aabb = [&]() {
            if (raw_type_b == (uint32_t)CollisionPrimitive::Type::Capsule) {
              assert(b_scale.d0 == b_scale.d1);

              float r = b_scale.d0;
              float half_h = b_scale.d2;

              AABB capsule_aabb = {
                .pMin = { -r, -r, -(r + half_h) },
                .pMax = { r, r, r + half_h },
              };

              return capsule_aabb.applyTRS(b_pos, b_rot, { 1.f, 1.f, 1.f });
            } else {
              return b_obj_aabb.applyTRS(b_pos, b_rot, b_scale);
            }
          } ();

          if (!a_world_aabb.intersects(b_world_aabb)) {
#ifdef BOT_GPU
            return;
#else
            continue;
#endif
          } else {
            NarrowphaseCandidate narrow_cand = {
              .a = a_ref,
              .b = b_ref,
              .aPrim = (u32)a_prim_idx,
              .bPrim = (u32)b_prim_idx,
            };

            world->physics.narrowphaseStore.add(rt, narrow_cand);
          }
        }
      }
#ifdef BOT_GPU
      });
#else
      }
#endif
}

enum class NarrowphaseTest : uint32_t {
  SphereSphere = 0b1,
  HullHull = 0b10,
  PlanePlane = 0b100,
  CapsuleCapsule = 0b1000,
  BoxBox = 0b10000,

  SphereHull = 0b11,
  SpherePlane = 0b101,
  SphereCapsule = 0b1001,
  SphereBox = 0b10001,
  HullPlane = 0b110,
  HullCapsule = 0b1010,
  HullBox = 0b10010,
  PlaneCapsule = 0b1100,
  PlaneBox = 0b10100,
  CapsuleBox = 0b11000,
};

struct FaceQuery {
  float separation;
  int64_t faceIdx;
  Plane plane;
};

struct EdgeQuery {
  float separation;
  Vector3 normal;
  int32_t edgeIdxA;
  int32_t edgeIdxB;
};

struct HullState {
  HalfEdgeMesh mesh;
  Vector3 center;
};

struct Manifold {
  Vector3 contactPoints[4];
  float penetrationDepths[4];
  int32_t numContactPoints;
  Vector3 normal;
};

enum class ContactType {
  None,
  Sphere,
  Convex,
  SATPlane,
  SATFace,
  SATEdge,
  SATCapsule,
};

struct SphereContact {
  Vector3 normal;
  Vector3 pt;
  float depth;
};

struct ConvexContact {
  Vector3 contactPoints[4];
  float penetrationDepths[4];
  Vector3 normals[4];
  int32_t numContactPoints;
};

struct SATContact {
  Vector3 normal;
  float planeDOrSeparation;
  uint32_t refFaceIdxOrEdgeIdxA;
  uint32_t incidentFaceIdxOrEdgeIdxB;
};

static HullState makeHullState(
    BOT_GPU_COND(const int32_t mwgpu_lane_id,)
    const HalfEdgeMesh &mesh,
    Vector3 translation,
    Quat rotation,
    Diag3x3 scale,
    Vector3 *dst_vertices,
    Plane *dst_planes)
{
  Mat3x3 unscaled_rot = Mat3x3::fromQuat(rotation);
  Mat3x3 vertex_txfm = unscaled_rot * scale;
  Mat3x3 normal_txfm = unscaled_rot * scale.inv();

#ifdef BOT_GPU
  const int64_t start_offset = mwgpu_lane_id;
  constexpr int64_t elems_per_iter = 32;
#else
  constexpr int64_t start_offset = 0;
  constexpr int64_t elems_per_iter = 1;
#endif

  // FIXME: get rid of this center computation - store the offset from
  // COM in each CollisionPrimitive and translate it
  Vector3 center = Vector3::zero();

  const int64_t num_vertices = mesh.numVertices;
  for (int64_t i = start_offset; i < num_vertices; i += elems_per_iter) {
    Vector3 world_pos = vertex_txfm * mesh.vertices[i] + translation;
    dst_vertices[i] = world_pos;
    center += world_pos;
  }

  center /= num_vertices;

  // FIXME: could significantly optimize this with a uniform scale
  // version
  const int64_t num_faces = mesh.numFaces;
  for (int64_t i = start_offset; i < num_faces; i += elems_per_iter) {
    Plane obj_plane = mesh.facePlanes[i];
    Vector3 plane_origin =
      vertex_txfm * (obj_plane.normal * obj_plane.d) + translation;

    Vector3 txfmed_normal = (normal_txfm * obj_plane.normal).normalize();
    float new_d = dot(txfmed_normal, plane_origin);

    dst_planes[i] = {
      txfmed_normal,
      new_d,
    };

    // Center should be behind each face plane (otherwise face normals
    // are probably facing the wrong way) - this is too tight a loop
    // to have this assert running all the time though
    //assert(center.dot(txfmed_normal) - new_d < 0.0f);
  }

  HalfEdgeMesh new_mesh {
    .halfEdges = mesh.halfEdges,
    .faceBaseHalfEdges = mesh.faceBaseHalfEdges,
    .facePlanes = dst_planes,
    .vertices = dst_vertices,
    .numHalfEdges = mesh.numHalfEdges,
    .numFaces = uint32_t(num_faces),
    .numVertices = uint32_t(num_vertices),
  };

  return HullState {
    new_mesh,
      center,
  };

}

// Returns the signed distance
static inline float getDistanceFromPlane(
    const Plane &plane, const Vector3 &a)
{
  float adotn = dot(a, plane.normal);
  return adotn - plane.d;
}

// Get intersection on plane of the line passing through 2 points
inline Vector3 planeIntersection(
    const Plane &plane, const Vector3 &p1, const Vector3 &p2) {
  float distance = getDistanceFromPlane(plane, p1);

  return p1 + (p2 - p1) * (-distance / plane.normal.dot(p2 - p1));
}

#ifdef BOT_GPU

BOT_ALWAYS_INLINE static inline std::pair<float, int32_t> 
warpFloatMaxAndIdx(float val, int32_t idx)
{
#pragma unroll
  for (int32_t w = 16; w >= 1; w /= 2) {
    float other_val =
      __shfl_xor_sync(0xFFFF'FFFF, val, w);
    int32_t other_idx =
      __shfl_xor_sync(0xFFFF'FFFF, idx, w);

    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }

  return { val, idx };
}

BOT_ALWAYS_INLINE static inline std::pair<float, int32_t>
warpFloatMinAndIdx(float val, int32_t idx)
{
#pragma unroll
  for (int32_t w = 16; w >= 1; w /= 2) {
    float other_val =
      __shfl_xor_sync(0xFFFF'FFFF, val, w);
    int32_t other_idx =
      __shfl_xor_sync(0xFFFF'FFFF, idx, w);

    if (other_val < val) {
      val = other_val;
      idx = other_idx;
    }
  }

  return { val, idx };
}

BOT_ALWAYS_INLINE static inline  float warpFloatMin(float val)
{
#pragma unroll
  for (int32_t w = 16; w >= 1; w /= 2) {
    float other_val =
      __shfl_xor_sync(0xFFFF'FFFF, val, w);

    if (other_val < val) {
      val = other_val;
    }
  }

  return val;
}

#endif

std::pair<Vector3, Vector3> findShortestConnection(
        const Vector3 p1, const Vector3 p2,
        const Vector3 q1, const Vector3 q2)
{
  Vector3 u = p2 - p1;
  Vector3 v = q2 - q1;
  Vector3 w = p1 - q1;

  float a = u.dot(u);
  float b = u.dot(v);
  float c = v.dot(v);
  float d = u.dot(w);
  float e = v.dot(w);
  float sc, sN, sD = a*c - b*b;
  float tc, tN, tD = a*c - b*b;
  float tol = 1e-15;

  if (sD < tol) {
    sN = 0.0;
    sD = 1.0;
    tN = e;
    tD = c;
  } else {
    sN = (b*e - c*d);
    tN = (a*e - b*d);
    if (sN < 0.0) {
      sN = 0.0;
      tN = e;
      tD = c;
    } else if (sN > sD) {
      sN = sD;
      tN = e + b;
      tD = c;
    }
  }

  if (tN < 0.0) {
    tN = 0.0;
    if (-d < 0.0) {
      sN = 0.0;
    } else if (-d > a) {
      sN = sD;
    } else {
      sN = -d;
      sD = a;
    }
  } else if (tN > tD) {
    tN = tD;
    if ((-d + b) < 0.0) {
      sN = 0;
    } else if ((-d + b) > a) {
      sN = sD;
    } else {
      sN = (-d +  b);
      sD = a;
    }
  }
  // finally do the division to get sc and tc
  sc = (fabs(sN) < tol ? 0.0 : sN / sD);
  tc = (fabs(tN) < tol ? 0.0 : tN / tD);

  Vector3 r1 = p1 + (sc * u);
  Vector3 r2 = q1 + (tc * v);

  return {r1, r2};
}

std::pair<bool, SphereContact> sphereSphereContact(
    const Vector3 a_pos,
    const Vector3 b_pos,
    const Quat a_rot,
    const Quat b_rot,
    const float a_radius,
    const float b_radius)
{
  Vector3 diff = a_pos - b_pos;
  float dist_sq = diff.length2();
  float min_dist = a_radius + b_radius;
  if (dist_sq > min_dist * min_dist) {
    return { false, {} };
  }

  float d = sqrtf(dist_sq) - a_radius - b_radius;
  Vector3 normal = b_pos - a_pos;
  float norm = normal.length();
  if (norm < 1e-15f) {
    Vector3 axis1 = a_rot.rotateVec(Vector3 { 0, 0, 1 });
    Vector3 axis2 = b_rot.rotateVec(Vector3 { 0, 0, 1 });
    normal = axis1.cross(axis2).normalize();
  } else {
    normal = normal.normalize();
  }

  Vector3 contact_pt = a_pos + (a_radius + 0.5f * d) * normal;

  SphereContact contact {
    .normal = -normal, // always points from b to a
      .pt = contact_pt,
      .depth = -d,
  };
  return { true, contact };
}

// Contact between sphere (a) and plane (b)
std::pair<bool, SphereContact> spherePlaneContact(
    const Vector3 a_pos,
    const Vector3 b_pos,
    const Quat b_rot,
    const float sphere_radius)
{
  constexpr Vector3 base_normal = { 0, 0, 1 };
  Vector3 plane_normal = b_rot.rotateVec(base_normal);

  Vector3 to_sphere = a_pos - b_pos;
  float dist = to_sphere.dot(plane_normal);
  if (dist > sphere_radius) {
    return { false, {} };
  }

  float d = dist - sphere_radius;
  Vector3 contact_point = a_pos - (d / 2.f + sphere_radius) * plane_normal;

  SphereContact sphere_contact {
    .normal = plane_normal,
      .pt = contact_point,
      .depth = -d,
  };
  return { true, sphere_contact };
}


static float getHullDistanceFromPlane(
    BOT_GPU_COND(const int32_t mwgpu_lane_id,)
    const Plane &plane, const HullState &h)
{
#ifdef BOT_GPU
  constexpr int64_t elems_per_iter = 32;
#else
  constexpr int64_t elems_per_iter = 1;
#endif

  float min_dot_n = FLT_MAX;

  auto computeVertexDotN = [&h, &plane](int64_t vert_idx) {
    Vector3 vertex = h.mesh.vertices[vert_idx];
    return dot(vertex, plane.normal);
  };

  const int64_t num_verts = (int64_t)h.mesh.numVertices;
  for (int32_t offset = 0; offset < num_verts; offset += elems_per_iter) {
#ifdef BOT_GPU
    int32_t vert_idx = offset + mwgpu_lane_id;
    float cur_dot;
    if (vert_idx < num_verts) {
      cur_dot = computeVertexDotN(vert_idx);
    } else {
      cur_dot = FLT_MAX;
    }
#else
    float cur_dot = computeVertexDotN(offset);
#endif

    if (cur_dot < min_dot_n) {
      min_dot_n = cur_dot;
    }
  }

#ifdef BOT_GPU
  min_dot_n = warpFloatMin(min_dot_n);
#endif

  return min_dot_n - plane.d;
}

struct CapsuleState {
  float radius;
  float cylinderHeight;
  Vector3 p1;
  Vector3 p2;
};

static FaceQuery queryFaceDirections(
    BOT_GPU_COND(int32_t mwgpu_lane_id,)
    const HullState &a, const HullState &b)
{
  Plane max_face_plane;
  int64_t max_dist_face = -1;
  float max_dist = -FLT_MAX;

  const int64_t num_a_faces = (int64_t)a.mesh.numFaces;
  for (int64_t face_idx = 0; face_idx < num_a_faces; face_idx++) {
    Plane plane = a.mesh.facePlanes[face_idx];
    float face_dist = getHullDistanceFromPlane(
        BOT_GPU_COND(mwgpu_lane_id,) plane, b);

    if (face_dist > max_dist) {
      max_dist = face_dist;
      max_dist_face = face_idx;
      max_face_plane = plane;

      if (max_dist > 0) {
        break;
      }
    }
  }

  return { max_dist, max_dist_face, max_face_plane };
}

static bool isMinkowskiFace(
        const Vector3 &a, const Vector3 &b,
        const Vector3 &c, const Vector3 &d)
{
  Vector3 bxa = b.cross(a);
  Vector3 dxc = d.cross(c);

  float cba = c.dot(bxa);
  float dba = d.dot(bxa);
  float adc = a.dot(dxc);
  float bdc = b.dot(dxc);

  return cba * dba < 0.0f && adc * bdc < 0.0f && cba * bdc > 0.0f;
}

static inline std::pair<Vector3, Vector3> getEdgeNormals(
        const HalfEdgeMesh &mesh, HalfEdge cur_hedge, HalfEdge twin_hedge)
{
  Vector3 normal1 = mesh.facePlanes[cur_hedge.face].normal;
  Vector3 normal2 = mesh.facePlanes[twin_hedge.face].normal;

  return { normal1, normal2 };
}

static inline Segment getEdgeSegment(const Vector3 *vertices,
                                     const HalfEdge *hedges,
                                     HalfEdge start)
{
  Vector3 a = vertices[start.rootVertex];

  // FIXME: probably should put both vertex indices inline in the half edge
  Vector3 b = vertices[hedges[start.next].rootVertex];

  return { a, b };
}

static inline bool buildsMinkowskiFace(
        const HalfEdgeMesh &a_mesh, const HalfEdgeMesh &b_mesh,
        HalfEdge cur_hedge_a, HalfEdge twin_hedge_a,
        HalfEdge cur_hedge_b, HalfEdge twin_hedge_b)
{
  auto [aNormal1, aNormal2] =
    getEdgeNormals(a_mesh, cur_hedge_a, twin_hedge_a);
  auto [bNormal1, bNormal2] =
    getEdgeNormals(b_mesh, cur_hedge_b, twin_hedge_b);

  return isMinkowskiFace(aNormal1, aNormal2, -bNormal1, -bNormal2);
}

struct EdgeTestResult {
  Vector3 normal;
  float separation;
};

static inline EdgeTestResult edgeDistance(
        const HullState &a, const HullState &b,
        HalfEdge hedge_a, HalfEdge hedge_b)
{
  Segment segment_a =
    getEdgeSegment(a.mesh.vertices, a.mesh.halfEdges, hedge_a);
  Segment segment_b =
    getEdgeSegment(b.mesh.vertices, b.mesh.halfEdges, hedge_b);

  Vector3 dir_a = segment_a.p2 - segment_a.p1;
  Vector3 dir_b = segment_b.p2 - segment_b.p1;

  Vector3 unnormalized_cross = dir_a.cross(dir_b);
  float normal_len2 = unnormalized_cross.length2();

  if (normal_len2 == 0) {
    EdgeTestResult result;
    result.separation = -FLT_MAX;

    return result;
  }

  float inv_normal_len =
#ifdef BOT_GPU
    rsqrtf(normal_len2);
#else
  1.f / sqrtf(normal_len2);
#endif

  Vector3 normal = unnormalized_cross * inv_normal_len;

  if (normal.dot(segment_a.p1 - a.center) < 0.0f) {
    normal = -normal;
  }

  float separation = normal.dot(segment_b.p1 - segment_a.p1);

  return {
    normal,
      separation,
  };
}

static EdgeQuery queryEdgeDirections(
    BOT_GPU_COND(int32_t mwgpu_lane_id,)
    const HullState &a, const HullState &b)
{
  Vector3 normal {};
  int edgeAMaxDistance = 0;
  int edgeBMaxDistance = 0;
  float maxDistance = -FLT_MAX;

  auto testEdgeSeparation = [&a, &b](uint32_t hedge_idx_a,
      uint32_t hedge_idx_b) {
    HalfEdge cur_hedge_a = a.mesh.halfEdges[hedge_idx_a];
    HalfEdge twin_hedge_a = a.mesh.halfEdges[a.mesh.twinIDX(hedge_idx_a)];
    HalfEdge cur_hedge_b = b.mesh.halfEdges[hedge_idx_b];
    HalfEdge twin_hedge_b = b.mesh.halfEdges[b.mesh.twinIDX(hedge_idx_b)];

    if (buildsMinkowskiFace(a.mesh, b.mesh, cur_hedge_a, twin_hedge_a,
          cur_hedge_b, twin_hedge_b)) {
      return edgeDistance(a, b, cur_hedge_a, cur_hedge_b);
    } else {
      EdgeTestResult result;
      result.separation = -FLT_MAX;
      return result;
    }
  };

  const int64_t a_num_edges = a.mesh.numEdges();
  const int64_t b_num_edges = b.mesh.numEdges();

#ifdef BOT_GPU
  const int32_t num_edge_tests = a_num_edges * b_num_edges;
  for (int32_t edge_offset_linear = 0; edge_offset_linear < num_edge_tests;
      edge_offset_linear += 32) {
    int32_t edge_idx_linear = edge_offset_linear + mwgpu_lane_id;

    // FIXME: get rid of this level of indirection
    int32_t edge_idx_a = edge_idx_linear / b_num_edges;
    int32_t edge_idx_b = edge_idx_linear % b_num_edges;

    EdgeTestResult edge_cmp;
    int32_t he_a_idx;
    int32_t he_b_idx;

    if (edge_idx_linear >= num_edge_tests ) {
      edge_cmp.separation = -FLT_MAX;
    } else {
      he_a_idx = a.mesh.edgeToHalfEdge(edge_idx_a);
      he_b_idx = b.mesh.edgeToHalfEdge(edge_idx_b);

      edge_cmp = testEdgeSeparation(he_a_idx, he_b_idx);
    }

    if (edge_cmp.separation > maxDistance) {
      maxDistance = edge_cmp.separation;
      normal = edge_cmp.normal;
      edgeAMaxDistance = he_a_idx;
      edgeBMaxDistance = he_b_idx;
    }

    if (__ballot_sync(0xFFFF'FFFF, maxDistance > 0) != 0) {
      break;
    }
  }

  int32_t max_lane_idx;
  std::tie(maxDistance, max_lane_idx) =
    warpFloatMaxAndIdx(maxDistance, mwgpu_lane_id);

  normal.x =  __shfl_sync(0xFFFF'FFFF, normal.x, max_lane_idx);
  normal.y =  __shfl_sync(0xFFFF'FFFF, normal.y, max_lane_idx);
  normal.z =  __shfl_sync(0xFFFF'FFFF, normal.z, max_lane_idx);

  edgeAMaxDistance = __shfl_sync(0xFFFF'FFFF,
      edgeAMaxDistance, max_lane_idx);
  edgeBMaxDistance = __shfl_sync(0xFFFF'FFFF,
      edgeBMaxDistance, max_lane_idx);

#else
  for (int64_t edge_idx_a = 0; edge_idx_a < a_num_edges; edge_idx_a++) {
    int32_t he_idx_a = a.mesh.edgeToHalfEdge(edge_idx_a);
    for (int64_t edge_idx_b = 0; edge_idx_b < b_num_edges; edge_idx_b++) {
      int32_t he_idx_b = b.mesh.edgeToHalfEdge(edge_idx_b);

      EdgeTestResult edge_cmp = testEdgeSeparation(he_idx_a, he_idx_b);

      if (edge_cmp.separation > maxDistance) {
        maxDistance = edge_cmp.separation;
        normal = edge_cmp.normal;
        edgeAMaxDistance = he_idx_a;
        edgeBMaxDistance = he_idx_b;

        if (maxDistance > 0) {
          // FIXME: this goto probably kills autovectorization
          goto early_out;
        }
      }
    }
  }

early_out:
#endif

  return { maxDistance, normal, edgeAMaxDistance, edgeBMaxDistance };
}

static int64_t findIncidentFace(BOT_GPU_COND(int32_t mwgpu_lane_id,)
    const HullState &h, Vector3 ref_normal)
{
#ifdef BOT_GPU
  constexpr int64_t elems_per_iter = 32;
#else
  constexpr int64_t elems_per_iter = 1;
#endif

  auto computeFaceDotRef = [&h, ref_normal](int64_t face_idx) {
    Plane face_plane = h.mesh.facePlanes[face_idx];
    return dot(face_plane.normal, ref_normal);
  };

  float min_dot = FLT_MAX;
  int64_t minimizing_face = -1;

  const int64_t num_faces = (int64_t)h.mesh.numFaces;
  for (int64_t offset = 0; offset < num_faces; offset += elems_per_iter) {
#ifdef BOT_GPU
    const int64_t face_idx = offset + mwgpu_lane_id;

    float face_dot_ref;
    if (face_idx < num_faces) {
      face_dot_ref = computeFaceDotRef(face_idx);
    } else{
      face_dot_ref = FLT_MAX;
    }
#else
    const int64_t face_idx = offset;
    float face_dot_ref = computeFaceDotRef(face_idx);
#endif

    if (face_dot_ref < min_dot) {
      min_dot = face_dot_ref;
      minimizing_face = face_idx;
    }
  }

#ifdef BOT_GPU
  std::tie(min_dot, minimizing_face) =
    warpFloatMinAndIdx(min_dot, minimizing_face);
#endif

  assert(minimizing_face != -1);
  return minimizing_face;
}

static inline int64_t clipPolygon(Vector3 *dst_vertices,
                                 Plane clipping_plane,
                                 const Vector3 *input_vertices,
                                 int64_t num_input_vertices)
{
  int64_t num_new_vertices = 0;

  Vector3 v1 = input_vertices[num_input_vertices - 1];
  float d1 = getDistanceFromPlane(clipping_plane, v1);

  for (int64_t i = 0; i < num_input_vertices; ++i) {
    Vector3 v2 = input_vertices[i];
    float d2 = getDistanceFromPlane(clipping_plane, v2);

    if (d1 <= 0.0f && d2 <= 0.0f) {
      // Both vertices are behind the plane, keep the second vertex
      dst_vertices[num_new_vertices++] = v2;
    }
    else if (d1 <= 0.0f && d2 > 0.0f) {
      // v1 is behind the plane, the other is in front (out)
      Vector3 intersection = planeIntersection(clipping_plane, v1, v2);
      dst_vertices[num_new_vertices++] = intersection;
    }
    else if (d2 <= 0.0f && d1 > 0.0f) {
      Vector3 intersection = planeIntersection(clipping_plane, v1, v2);
      dst_vertices[num_new_vertices++] = intersection;
      dst_vertices[num_new_vertices++] = v2;
    }

    // Now use v2 as the starting vertex
    v1 = v2;
    d1 = d2;
  }

  return num_new_vertices;
}

struct SATResult {
  ContactType type;
  SATContact contact;
};

static inline SATResult doSAT(BOT_GPU_COND(int32_t mwgpu_lane_id,)
                              const HullState &a, const HullState &b)
{
  FaceQuery faceQueryA =
    queryFaceDirections(BOT_GPU_COND(mwgpu_lane_id,) a, b);
  if (faceQueryA.separation > 0.0f) {
    // There is a separating axis - no collision
    SATResult result;
    result.type = ContactType::None;

    return result;
  }

  FaceQuery faceQueryB =
    queryFaceDirections(BOT_GPU_COND(mwgpu_lane_id,) b, a);
  if (faceQueryB.separation > 0.0f) {
    // There is a separating axis - no collision
    SATResult result;
    result.type = ContactType::None;

    return result;
  }

  EdgeQuery edgeQuery =
    queryEdgeDirections(BOT_GPU_COND(mwgpu_lane_id,) a, b);
  if (edgeQuery.separation > 0.0f) {
    // There is a separating axis - no collision
    SATResult result;
    result.type = ContactType::None;

    return result;
  }

  bool bIsFaceContactA = faceQueryA.separation > edgeQuery.separation;
  bool bIsFaceContactB = faceQueryB.separation > edgeQuery.separation;

  if (bIsFaceContactA || bIsFaceContactB) {
    bool a_is_ref = faceQueryA.separation >= faceQueryB.separation;

    Plane ref_plane = a_is_ref ? faceQueryA.plane : faceQueryB.plane;
    int64_t ref_face_idx =
      a_is_ref ? faceQueryA.faceIdx : faceQueryB.faceIdx;
    const HullState &incident_hull = a_is_ref ? b : a;

    // Find incident face
    int64_t incident_face_idx = findIncidentFace(
        BOT_GPU_COND(mwgpu_lane_id,) incident_hull, ref_plane.normal);


    SATResult result;
    result.type = ContactType::SATFace,
      result.contact.normal = ref_plane.normal;
    result.contact.planeDOrSeparation = ref_plane.d;
    uint32_t mask;
    if (a_is_ref) {
      mask = 0_u32;
    } else {
      mask = 1_u32 << 31_u32;
    }
    result.contact.refFaceIdxOrEdgeIdxA = uint32_t(ref_face_idx) | mask;
    result.contact.incidentFaceIdxOrEdgeIdxB = uint32_t(incident_face_idx);

    return result;
  } else {
    SATResult result;
    result.type = ContactType::SATEdge;
    result.contact.normal = edgeQuery.normal;
    result.contact.planeDOrSeparation = edgeQuery.separation;
    result.contact.refFaceIdxOrEdgeIdxA = edgeQuery.edgeIdxA;
    result.contact.incidentFaceIdxOrEdgeIdxB = edgeQuery.edgeIdxB;
    return result;
  }
}

SATResult doSATPlane(BOT_GPU_COND(const int32_t mwgpu_lane_id,)
                     const Plane &plane, const HullState &h)
{
  float separation = getHullDistanceFromPlane(
      BOT_GPU_COND(mwgpu_lane_id,) plane, h);

  if (separation > 0.0f) {
    SATResult result;
    result.type = ContactType::None;

    return result;
  }

  // Find incident face
  int64_t incident_face_idx = findIncidentFace(
      BOT_GPU_COND(mwgpu_lane_id,) h, plane.normal);

  SATResult result;
  result.type = ContactType::SATPlane;
  result.contact.normal = plane.normal;
  result.contact.planeDOrSeparation = plane.d;
  result.contact.incidentFaceIdxOrEdgeIdxB = uint32_t(incident_face_idx);

  return result;
}

static Manifold buildFaceContactManifold(
    Vector3 contact_normal,
    Vector3 *contacts,
    float *penetration_depths,
    int64_t num_contacts,
    Vector3 world_offset,
    Quat to_world_frame)
{
  Manifold manifold;
  if (num_contacts <= 4) {
    manifold.numContactPoints = num_contacts;
    for (int64_t i = 0; i < num_contacts; i++) {
      manifold.contactPoints[i] = contacts[i];
      manifold.penetrationDepths[i] = penetration_depths[i];
    }
  } else {
    // Going to select contact manifold comprised of points
    // A B C and Q following Gregorious presentation.

    // Select point A as first point in contact list
    manifold.numContactPoints = 4;
    manifold.contactPoints[0] = contacts[0];
    manifold.penetrationDepths[0] = penetration_depths[0];

    // Find point B furthest from point A
    float max_dist_sq = 0.f;
    for (int64_t i = 1; i < num_contacts; i++) {
      Vector3 cur_contact = contacts[i];
      float dist_sq = manifold.contactPoints[0].distance2(cur_contact);
      if (dist_sq > max_dist_sq) {
        max_dist_sq = dist_sq;

        manifold.contactPoints[1] = cur_contact;
        manifold.penetrationDepths[1] = penetration_depths[i];
      }
    }

    Vector3 ba =
      manifold.contactPoints[1] - manifold.contactPoints[0];

    // Find point C which maximizes area of triangle ABC
    float max_tri_area = 0.0f;
    bool max_tri_sign = 0.f;
    for (int64_t i = 1; i < num_contacts; i++) {
      Vector3 cur_contact = contacts[i];
      Vector3 bc = cur_contact - manifold.contactPoints[1];
      float signed_area = contact_normal.dot(cross(ba, bc));
      float area = copysignf(signed_area, 1.f);

      if (area > max_tri_area) {
        max_tri_area = area;
        max_tri_sign = copysignf(1.f, signed_area);

        manifold.contactPoints[2] = cur_contact;
        manifold.penetrationDepths[2] = penetration_depths[i];
      }
    }

    // If we ultimately selected a triangle ABC with clockwise winding,
    // flip around edge BA to make the triangle counterclockwise, so the
    // next part only needs to search for negative area.
    if (max_tri_sign == -1.f) {
      ba = -ba;
      std::swap(manifold.contactPoints[0], manifold.contactPoints[1]);
    }

    // Select point Q that adds the most area to ABC
    // Need to check ABQ (BA x AQ), BCQ (CB x QC), and CAQ (AC x QA)

    Vector3 cb = manifold.contactPoints[2] - manifold.contactPoints[1];
    Vector3 ac = manifold.contactPoints[0] - manifold.contactPoints[2];

    float most_neg_area = 0.f;
    for (int64_t i = 1; i < num_contacts; i++) {
      Vector3 cur_contact = contacts[i];

      Vector3 aq = manifold.contactPoints[0] - cur_contact;
      Vector3 qc = cur_contact - manifold.contactPoints[2];

      float abq_area = contact_normal.dot(cross(ba, aq));
      float bcq_area = contact_normal.dot(cross(cb, qc));
      float caq_area = contact_normal.dot(cross(aq, ac));

      float q_min_area = fminf(abq_area, fminf(bcq_area, caq_area));
      if (q_min_area < most_neg_area) {
        most_neg_area = q_min_area;

        manifold.contactPoints[3] = cur_contact;
        manifold.penetrationDepths[3] = penetration_depths[i];
      }
    }

    if (max_dist_sq == 0.f || max_tri_area == 0.f || most_neg_area == 0.f) {
      // FIXME: should not be possible
      manifold.numContactPoints = 0;
      manifold.normal = Vector3::zero();
      return manifold;
    }
  }

  for (int64_t i = 0; i < (int64_t)manifold.numContactPoints; i++) {
    manifold.contactPoints[i] =
      to_world_frame.rotateVec(manifold.contactPoints[i]) + world_offset;
  }

  manifold.normal = to_world_frame.rotateVec(contact_normal);

  return manifold;
}

BOT_ALWAYS_INLINE static inline Manifold createFaceContact(
    Plane ref_plane,
    int32_t ref_face_idx,
    int32_t incident_face_idx,
    const Vector3 *ref_vertices,
    const Vector3 *other_vertices,
    const HalfEdge *ref_hedges,
    const HalfEdge *other_hedges,
    const uint32_t *ref_face_hedges,
    const uint32_t *other_face_hedges,
    void *tmp_buf1, void *tmp_buf2,
#ifdef BOT_GPU
    Mat3x4 ref_txfm, Mat3x4 other_txfm,
#endif
    Vector3 world_offset, Quat to_world_frame)
{
  // Collect incident vertices: FIXME should have face indices
  Vector3 *incident_vertices_tmp = (Vector3 *)tmp_buf1;
  int64_t num_incident_vertices = 0;
  {
    int64_t hedge_idx = other_face_hedges[incident_face_idx];
    int64_t start_hedge_idx = hedge_idx;

    do {
      const auto &cur_hedge = other_hedges[hedge_idx];
      hedge_idx = cur_hedge.next;

      Vector3 cur_point = other_vertices[cur_hedge.rootVertex];
#ifdef BOT_GPU
      cur_point = other_txfm.txfmPoint(cur_point);
#endif

      incident_vertices_tmp[num_incident_vertices++] = cur_point;
    } while (hedge_idx != start_hedge_idx);
  }

  Vector3 *clipping_input = incident_vertices_tmp;
  int64_t num_clipped_vertices = num_incident_vertices;

  Vector3 *clipping_dst = (Vector3 *)tmp_buf2;

  // max output vertices is num_incident_vertices + num planes
  // but we don't know num planes ahead of time without iterating
  // through the reference face twice! Alternative would be to cache the
  // side planes, or store max face size in each mesh. The worst case
  // buffer sizes here is just the sum of the max face sizes - 1

  // FIXME, this code assumes that clipping_input & clipping_dst have space
  // to write incident_vertices + num_planes new vertices
  // Loop over side planes
  {
    int64_t hedge_idx = ref_face_hedges[ref_face_idx];
    int64_t start_hedge_idx = hedge_idx;

    auto *cur_hedge = &ref_hedges[hedge_idx];
    Vector3 cur_point = ref_vertices[cur_hedge->rootVertex];
#ifdef BOT_GPU
    cur_point = ref_txfm.txfmPoint(cur_point);
#endif
    do {
      hedge_idx = cur_hedge->next;
      cur_hedge = &ref_hedges[hedge_idx];
      Vector3 next_point = ref_vertices[cur_hedge->rootVertex];
#ifdef BOT_GPU
      next_point = ref_txfm.txfmPoint(next_point);
#endif

      Vector3 edge = next_point - cur_point;
      Vector3 plane_normal = cross(edge, ref_plane.normal);

      float d = dot(plane_normal, cur_point);
      cur_point = next_point;

      Plane side_plane {
        plane_normal,
          d,
      };

      num_clipped_vertices = clipPolygon(clipping_dst, side_plane,
          clipping_input, num_clipped_vertices);

      std::swap(clipping_dst, clipping_input);
    } while (hedge_idx != start_hedge_idx);
  }

  // assert(num_clipped_vertices > 0);

  // clipping_input has the result due to the final swap

  // Filter clipping_input to ones below ref_plane and save penetration depth
  float *penetration_depths = (float *)clipping_dst;

  int64_t num_below_plane = 0;
  for (int64_t i = 0; i < num_clipped_vertices; ++i) {
    Vector3 vertex = clipping_input[i];
    if (float d = getDistanceFromPlane(ref_plane, vertex); d <= 0.0f) {
      // Project the point onto the reference plane
      // (d guaranteed to be negative)
      clipping_input[num_below_plane] = vertex - d * ref_plane.normal;
      penetration_depths[num_below_plane] = -d;

      num_below_plane += 1;
    }
  }

  return buildFaceContactManifold(ref_plane.normal, clipping_input,
      penetration_depths, num_below_plane,
      world_offset, to_world_frame);
}

static Manifold createFacePlaneContact(Plane plane,
    int32_t incident_face_idx,
    const Vector3 *vertices,
    const HalfEdge *hedges,
    const uint32_t *face_hedge_roots,
    Vector3 *contacts_tmp,
    float *penetration_depths_tmp,
#ifdef BOT_GPU
    Mat3x4 hull_txfm,
#endif
    Vector3 world_offset,
    Quat to_world_frame)
{
  // Collect incident vertices: FIXME should have face indices
  int64_t num_incident_vertices = 0;
  {
    int64_t hedge_idx = face_hedge_roots[incident_face_idx];
    int64_t start_hedge_idx = hedge_idx;

    do {
      const auto &cur_hedge = hedges[hedge_idx];
      hedge_idx = cur_hedge.next;
      Vector3 vertex = vertices[cur_hedge.rootVertex];

#ifdef BOT_GPU
      vertex = hull_txfm.txfmPoint(vertex);
#endif

      if (float d = getDistanceFromPlane(plane, vertex); d <= 0.0f) {
        // Project the point onto the reference plane
        // (d guaranteed to be negative)
        contacts_tmp[num_incident_vertices] =
          vertex - d * plane.normal;
        penetration_depths_tmp[num_incident_vertices] = -d;

        num_incident_vertices += 1;
      }
    } while (hedge_idx != start_hedge_idx);
  }

  return buildFaceContactManifold(plane.normal, contacts_tmp,
      penetration_depths_tmp, num_incident_vertices,
      world_offset, to_world_frame);
}


static Segment shortestSegmentBetween(const Segment &seg1, const Segment &seg2)
{
  Vector3 v1 = seg1.p2 - seg1.p1;
  Vector3 v2 = seg2.p2 - seg2.p1;

  Vector3  v21 = seg2.p1 - seg1.p1;

  float dotv22 = v2.dot(v2);
  float dotv11 = v1.dot(v1);
  float dotv21 = v2.dot(v1);
  float dotv211 = v21.dot(v1);
  float dotv212 = v21.dot(v2);

  float denom = dotv21 * dotv21 - dotv22 * dotv11;

  float s, t;

  // FIXME: validate this epsilon
  if (fabsf(denom) < 0.00001f) {
    s = 0.0f;
    t = (dotv11 * s - dotv211) / dotv21;
  }
  else {
    s = (dotv212 * dotv21 - dotv22 * dotv211) / denom;
    t = (-dotv211 * dotv21 + dotv11 * dotv212) / denom;
  }

  s = fmaxf(fminf(s, 1.0f), 0.0f);
  t = fmaxf(fminf(t, 1.0f), 0.0f);

  return { seg1.p1 + s * v1, seg2.p1 + t * v2 };
}

static Manifold createEdgeContact(Segment segA,
    Segment segB,
    Vector3 normal,
    float separation,
    Vector3 world_offset,
    Quat to_world_frame)
{
#if 0
  Segment s = shortestSegmentBetween(segA, segB);
  Vector3 contact = 0.5f * (s.p1 + s.p2);
  float depth = 0.5f * (s.p2 - s.p1).length();
#endif

  // Deviation from Gregorius GDC 2015:
  // Currently the solver expects the contact point to be ON object A.
  // For Face-Face this means the point is on object A's face, which is
  // the same as the presentation. In the edge-edge case, the presentation
  // has the contact point between the edges.
  // Our solver currently reconstructs the contact points on point B
  // using the normal depth, so the presentation's method will
  // reconstruct the wrong contact points.
  // FIXME: revisit this after modifying solver to handle multi-point
  // manifolds better

  // FIXME: don't need this full function call here.
  Segment s = shortestSegmentBetween(segA, segB);
  Vector3 contact = s.p1;

  Manifold manifold;
  manifold.contactPoints[0] =
    to_world_frame.rotateVec(contact) + world_offset,
    manifold.penetrationDepths[0] = -separation;
  manifold.numContactPoints = 1;
  manifold.normal = to_world_frame.rotateVec(normal);

  return manifold;
}

static Manifold createEdgeContact(Vector3 normal,
                                  float separation,
                                  int32_t hedge_idx_a,
                                  int32_t hedge_idx_b,
                                  const Vector3 *a_vertices,
                                  const Vector3 *b_vertices,
                                  const HalfEdge *a_hedges,
                                  const HalfEdge *b_hedges,
#ifdef BOT_GPU
                                  Vector3 a_pos, Quat a_rot, Diag3x3 a_scale,
                                  Vector3 b_pos, Quat b_rot, Diag3x3 b_scale,
#endif
                                  Vector3 world_offset,
                                  Quat to_world_frame)
{
  Segment segA = getEdgeSegment(a_vertices, a_hedges,
      a_hedges[hedge_idx_a]);
  Segment segB = getEdgeSegment(b_vertices, b_hedges,
      b_hedges[hedge_idx_b]);

#ifdef BOT_GPU
  segA.p1 = a_rot.rotateVec(a_scale * segA.p1) + a_pos;
  segA.p2 = a_rot.rotateVec(a_scale * segA.p2) + a_pos;
  segB.p1 = b_rot.rotateVec(b_scale * segB.p1) + b_pos;
  segB.p2 = b_rot.rotateVec(b_scale * segB.p2) + b_pos;
#endif

  return createEdgeContact(segA, segB,
      normal, separation,
      world_offset, to_world_frame);
}

static inline void addManifoldContacts(
    SimRT &rt,
    World *world,
    Manifold manifold,
    BodyRef ref_loc, BodyRef other_loc)
{
  ContactConstraint new_contact = {
    ref_loc,
    other_loc,
    {
      Vector4::fromVec3W(manifold.contactPoints[0],
          manifold.penetrationDepths[0]),
      Vector4::fromVec3W(manifold.contactPoints[1],
          manifold.penetrationDepths[1]),
      Vector4::fromVec3W(manifold.contactPoints[2],
          manifold.penetrationDepths[2]),
      Vector4::fromVec3W(manifold.contactPoints[3],
          manifold.penetrationDepths[3]),
    },
    manifold.numContactPoints,
    manifold.normal,
  };

  world->physics.contactStore.add(rt, new_contact);
}

static inline void addSinglePointContact(
    SimRT &rt,
    World *world,
    Vector3 point,
    Vector3 normal,
    float depth,
    BodyRef ref_loc,
    BodyRef other_loc)
{
  ContactConstraint new_contact = {
    ref_loc,
    other_loc,
    {
      Vector4::fromVec3W(point, depth),
      Vector4::zero(),
      Vector4::zero(),
      Vector4::zero(),
    },
    1,
    normal,
  };

  world->physics.contactStore.add(rt, new_contact);
}

#ifdef BOT_GPU
namespace gpuImpl {
// FIXME: do something actually intelligent here
inline constexpr int32_t maxNumPlanes = 40;
inline constexpr int32_t numPlaneFloats = maxNumPlanes * 4;
}
#endif

struct NarrowphaseResult {
  ContactType type;
  SphereContact sphere;
  ConvexContact convex;
  SATContact sat;
  const Vector3 *aVertices;
  const Vector3 *bVertices;
  const HalfEdge *aHalfEdges;
  const HalfEdge *bHalfEdges;
  const uint32_t *aFaceHedgeRoots;
  const uint32_t *bFaceHedgeRoots;
};

BOT_ALWAYS_INLINE static inline NarrowphaseResult narrowphaseDispatch(
    BOT_GPU_COND(const int32_t mwgpu_lane_id,)
    NarrowphaseTest test_type,
    Vector3 a_pos, Vector3 b_pos,
    Quat a_rot, Quat b_rot,
    Diag3x3 a_scale, Diag3x3 b_scale,
    const CollisionPrimitive *a_prim, const CollisionPrimitive *b_prim,
    int64_t max_num_tmp_vertices,
    int64_t max_num_tmp_faces,
    Vector3 *txfm_vertex_buffer,
    Plane *txfm_face_buffer)
{
  switch (test_type) {
  case NarrowphaseTest::SphereSphere: {
    float a_radius, b_radius;
    {
      assert(a_scale.d0 == a_scale.d1 && a_scale.d0 == a_scale.d2);
      assert(b_scale.d0 == b_scale.d1 && b_scale.d0 == b_scale.d2);

      a_radius = a_scale.d0 * a_prim->sphere.radius;
      b_radius = b_scale.d0 * b_prim->sphere.radius;
    }
    NarrowphaseResult result;
    auto [is_contact, contact] = sphereSphereContact(
        a_pos, b_pos, a_rot, b_rot, a_radius, b_radius);
    if (is_contact) {
      result.type = ContactType::Sphere;
      result.sphere = contact;
      return result;
    } else {
      result.type = ContactType::None;
      return result;
    }
  } break;
  case NarrowphaseTest::HullHull: {
    // Get half edge mesh for hull A and hull B
    const auto &a_he_mesh = a_prim->hull.halfEdgeMesh;
    const auto &b_he_mesh = b_prim->hull.halfEdgeMesh;

    assert(a_he_mesh.numFaces + b_he_mesh.numFaces <
        max_num_tmp_faces);

    assert(a_he_mesh.numVertices + b_he_mesh.numVertices <
        max_num_tmp_vertices);

    HullState a_hull_state = makeHullState(BOT_GPU_COND(mwgpu_lane_id,)
        a_he_mesh, a_pos, a_rot, a_scale, txfm_vertex_buffer,
        txfm_face_buffer);

    txfm_vertex_buffer += a_hull_state.mesh.numVertices;
    txfm_face_buffer += a_hull_state.mesh.numFaces;

    HullState b_hull_state = makeHullState(BOT_GPU_COND(mwgpu_lane_id,)
        b_he_mesh, b_pos, b_rot, b_scale, txfm_vertex_buffer,
        txfm_face_buffer);

    BOT_GPU_COND(__syncwarp(0xFFFF'FFFF));

    const SATResult sat = doSAT(BOT_GPU_COND(mwgpu_lane_id,)
        a_hull_state, b_hull_state);

    NarrowphaseResult result;
    result.type = sat.type;
    result.sat = sat.contact;
#ifdef BOT_GPU
    result.aVertices = a_he_mesh.vertices;
    result.bVertices = b_he_mesh.vertices;
#else
    result.aVertices = a_hull_state.mesh.vertices;
    result.bVertices = b_hull_state.mesh.vertices;
#endif
    result.aHalfEdges = a_hull_state.mesh.halfEdges;
    result.bHalfEdges = b_hull_state.mesh.halfEdges;
    result.aFaceHedgeRoots = a_hull_state.mesh.faceBaseHalfEdges;
    result.bFaceHedgeRoots = b_hull_state.mesh.faceBaseHalfEdges;

    return result;
  } break;
  case NarrowphaseTest::HullCapsule: {
    NarrowphaseResult result;
    result.type = ContactType::None;
    return result;
  } break;
  case NarrowphaseTest::HullBox: {
    // todo: hull box collision detection
    NarrowphaseResult result;
    result.type = ContactType::None;
    return result;
  } break;
  case NarrowphaseTest::SphereHull: {
    float sphere_radius;
    {
      auto sphere = a_prim->sphere;
      assert(a_scale.d0 == a_scale.d1 && a_scale.d0 == a_scale.d2);
      sphere_radius = a_scale.d0 * sphere.radius;
    }

    const auto &b_he_mesh = b_prim->hull.halfEdgeMesh;
    assert(b_he_mesh.numFaces < max_num_tmp_faces);
    assert(b_he_mesh.numVertices <  max_num_tmp_vertices);

    Vector3 hull_origin = b_pos - a_pos;

    HullState b_hull_state = makeHullState(BOT_GPU_COND(mwgpu_lane_id,)
        b_he_mesh, hull_origin, b_rot, b_scale,
        txfm_vertex_buffer, txfm_face_buffer);

    BOT_GPU_COND(__syncwarp(0xFFFF'FFFF));

    Vector3 to_hull_closest_pt;
    float hull_dist2 = hullClosestPointToOriginGJK(
        b_hull_state.mesh, 1e-10f, &to_hull_closest_pt);

    if (hull_dist2 > sphere_radius * sphere_radius) {
      NarrowphaseResult result;
      result.type = ContactType::None;
      return result;
    }

    SphereContact sphere_contact;

    if (hull_dist2 == 0.f) {
      // Need to do SAT
      float max_sep = -FLT_MAX;
      Vector3 sep_normal;
      const int64_t num_faces = b_hull_state.mesh.numFaces;
      for (int64_t i = 0; i < num_faces; i++) {
        Plane plane = b_hull_state.mesh.facePlanes[i];
        // hull has already been moved so sphere is at origin
        float face_dist = -plane.d;

        if (face_dist > max_sep) {
          max_sep = face_dist;
          sep_normal = plane.normal;
        }
      }

      // Discrepancy between SAT and GJK
      if (max_sep > 0.f) {
        assert(max_sep < 1e-5f);
        NarrowphaseResult result;
        result.type = ContactType::None;
        return result;
      }
      sphere_contact.normal = sep_normal;
      sphere_contact.pt = a_pos + sep_normal * sphere_radius;
      sphere_contact.depth = -max_sep;
    } else {
      float to_hull_len = sqrtf(hull_dist2);

      float depth = sphere_radius - to_hull_len;
      Vector3 normal = to_hull_closest_pt / to_hull_len;

      sphere_contact.normal = -normal;
      sphere_contact.pt = a_pos + normal * sphere_radius;
      sphere_contact.depth = depth;
    }

    NarrowphaseResult result;
    result.type = ContactType::Sphere;
    result.sphere = sphere_contact;
    result.aVertices = nullptr;
    result.bVertices = nullptr;
    result.aVertices = nullptr;
    result.bVertices = nullptr;
    result.aHalfEdges = nullptr;
    result.bHalfEdges = nullptr;
    result.aFaceHedgeRoots = nullptr;
    result.bFaceHedgeRoots = nullptr;
    return result;
  } break;
  case NarrowphaseTest::PlanePlane: {
    // Planes must be static, this should never be called
    assert(false);
    BOT_UNREACHABLE();
  } break;
  {
    assert(a_scale.d0 == a_scale.d1 && a_scale.d0 == a_scale.d2);
    float sphere_radius = a_scale.d0 * a_prim->sphere.radius;

    // Scale capsule properly
    constexpr Vector3 base_normal = { 0, 0, 1 };
    assert(b_scale.d0 == b_scale.d1);
    float cap_radius = b_scale.d0;
    float cap_len = b_scale.d2; // half height
    Vector3 cap_axis = b_rot.rotateVec(base_normal);

    // Find projection of sphere center onto capsule axis
    Vector3 to_sphere = a_pos - b_pos;
    float t = clamp(cap_axis.dot(to_sphere), -cap_len, cap_len);
    Vector3 closest_pt = b_pos + t * cap_axis;

    // Check for collision
    NarrowphaseResult result;
    auto [is_contact, contact] = sphereSphereContact(
        a_pos, closest_pt, a_rot, b_rot, sphere_radius, cap_radius);
    if (is_contact) {
      result.type = ContactType::Sphere;
      result.sphere = contact;
      return result;
    } else {
      result.type = ContactType::None;
      return result;
    }

  } break;
  case NarrowphaseTest::CapsuleCapsule: {
    assert(a_scale.d0 == a_scale.d1);
    assert(b_scale.d0 == b_scale.d1);
    NarrowphaseResult result;

    // Scale capsules properly
    float a_radius = a_scale.d0;
    float a_len = a_scale.d2; // half height
    float b_radius = b_scale.d0;
    float b_len = b_scale.d2;
    // Compute axes and scale
    constexpr Vector3 base_normal = { 0, 0, 1 };
    Vector3 a_axis = a_rot.rotateVec(base_normal);
    Vector3 b_axis = b_rot.rotateVec(base_normal);
    a_axis *= a_len;
    b_axis *= b_len;

    Vector3 diff = a_pos - b_pos;
    float ma = a_axis.dot(a_axis);
    float mb = -a_axis.dot(b_axis);
    float mc = b_axis.dot(b_axis);
    float u = -a_axis.dot(diff);
    float v = b_axis.dot(diff);
    float det = ma * mc - mb * mb;

    if (fabsf(det) >= 1e-15f) {
      // Projections, clip to capsule lengths
      float x1 = (mc * u - mb * v) / det;
      float x2 = (ma * v - mb * u) / det;

      if (x1 > 1) {
        x1 = 1;
        x2 = (v - mb) / mc;
      } else if (x1 < -1) {
        x1 = -1;
        x2 = (v+ mb) / mc;
      }
      if (x2 > 1) {
        x2 = 1;
        x1 = clamp((u - mb) / ma, -1.f, 1.f);
      } else if (x2 < -1) {
        x2 = -1;
        x1 = clamp((u + mb) / ma, -1.f, 1.f);
      }
      // Nearest points, sphere-sphere collision
      Vector3 vec1 = a_pos + x1 * a_axis;
      Vector3 vec2 = b_pos + x2 * b_axis;
      auto [is_contact, contact] = sphereSphereContact(
          vec1, vec2, a_rot, b_rot, a_radius, b_radius);
      if (is_contact) {
        result.type = ContactType::Sphere;
        result.sphere = contact;
        return result;
      } else {
        result.type = ContactType::None;
        return result;
      }
    }
    // Parallel axes
    else {
      int32_t num_contacts = 0;
      ConvexContact cv_contact = {};
      result.type = ContactType::Convex;

      // x1 = 1
      Vector3 vec1 = a_pos + a_axis;
      float x2 = clamp((v - mb) / mc, -1.f, 1.f);
      Vector3 vec2 = b_pos + x2 * b_axis;
      auto [is_contact1, contact1] = sphereSphereContact(
          vec1, vec2, a_rot, b_rot, a_radius, b_radius);
      if (is_contact1) {
        cv_contact.penetrationDepths[num_contacts] = contact1.depth;
        cv_contact.normals[num_contacts] = contact1.normal;
        cv_contact.contactPoints[num_contacts] = contact1.pt;
        num_contacts++;
      }

      // x1 = -1
      vec1 = a_pos - a_axis;
      x2 = clamp((v + mb) / mc, -1.f, 1.f);
      vec2 = b_pos + x2 * b_axis;
      auto [is_contact2, contact2] = sphereSphereContact(
          vec1, vec2, a_rot, b_rot, a_radius, b_radius);
      if (is_contact2) {
        cv_contact.penetrationDepths[num_contacts] = contact2.depth;
        cv_contact.normals[num_contacts] = contact2.normal;
        cv_contact.contactPoints[num_contacts] = contact2.pt;
        num_contacts++;
      }

      if (num_contacts >= 2) {
        cv_contact.numContactPoints = num_contacts;
        result.convex = cv_contact;
        return result;
      }

      // x2 = 1
      vec2 = b_pos + b_axis;
      float x1 = clamp((u - mb) / ma, -1.f, 1.f);
      vec1 = a_pos + x1 * a_axis;
      auto [is_contact3, contact3] = sphereSphereContact(
          vec1, vec2, a_rot, b_rot, a_radius, b_radius);
      if (is_contact3) {
        cv_contact.penetrationDepths[num_contacts] = contact3.depth;
        cv_contact.normals[num_contacts] = contact3.normal;
        cv_contact.contactPoints[num_contacts] = contact3.pt;
        num_contacts++;
      }
      if (num_contacts >= 2) {
        cv_contact.numContactPoints = num_contacts;
        result.convex = cv_contact;
        return result;
      }

      // x2 = -1
      vec2 = b_pos - b_axis;
      x1 = clamp((u + mb) / ma, -1.f, 1.f);
      vec1 = a_pos + x1 * a_axis;
      auto [is_contact4, contact4] = sphereSphereContact(
          vec1, vec2, a_rot, b_rot, a_radius, b_radius);
      if (is_contact4) {
        cv_contact.penetrationDepths[num_contacts] = contact4.depth;
        cv_contact.normals[num_contacts] = contact4.normal;
        cv_contact.contactPoints[num_contacts] = contact4.pt;
        num_contacts++;
      }
      cv_contact.numContactPoints = num_contacts;
      result.convex = cv_contact;
      return result;
    }

  } break;
  case NarrowphaseTest::PlaneBox: {
    assert(b_prim->type == CollisionPrimitive::Type::Box);
    constexpr Vector3 base_normal = { 0, 0, 1 };
    Vector3 plane_normal = a_rot.rotateVec(base_normal);
    Vector3 normal = -plane_normal; // always points b to a

    // Distance between centers
    Vector3 diff = b_pos - a_pos;
    float dist = plane_normal.dot(diff);

    // Check contacts with corners
    ConvexContact cv_contact = {};
    int32_t num_contacts = 0;
    Vector3 halfExtents = b_scale * b_prim->box.dim * 0.5f;
    for (int32_t i = 0; i < 8; i++) {
      // Corner position with respect to box center
      Vector3 offset = halfExtents;
      offset.x *= (i & 1) ? 1.f : -1.f;
      offset.y *= (i & 2) ? 1.f : -1.f;
      offset.z *= (i & 4) ? 1.f : -1.f;
      Vector3 corner = b_rot.rotateVec(offset);

      // Distance to plane
      float ldist = plane_normal.dot(corner);
      if (dist + ldist > 0.f || ldist > 0) {
        continue;
      }

      float d = dist + ldist;
      cv_contact.penetrationDepths[num_contacts] = -d;
      cv_contact.normals[num_contacts] = normal;
      cv_contact.contactPoints[num_contacts] = (b_pos + corner) -
        plane_normal * (d / 2.f);
      num_contacts++;
      // Max number of contacts is 4
      if (num_contacts == 4) {
        break;
      }
    }

    if (num_contacts == 0) {
      NarrowphaseResult result;
      result.type = ContactType::None;
      return result;
    }

    cv_contact.numContactPoints = num_contacts;
    NarrowphaseResult result = {};
    result.type = ContactType::Convex;
    result.convex = cv_contact;
    return result;
  }
  case NarrowphaseTest::BoxBox: {
    NarrowphaseResult result;
    result.type = ContactType::None;
    return result;
  }
  case NarrowphaseTest::CapsuleBox: {
    NarrowphaseResult result;
    result.type = ContactType::None;
    return result;
  }
  case NarrowphaseTest::SphereBox: {
    NarrowphaseResult result;
    assert(a_scale.d0 == a_scale.d1 && a_scale.d0 == a_scale.d2);
    assert(a_prim->type == CollisionPrimitive::Type::Sphere);
    assert(b_prim->type == CollisionPrimitive::Type::Box);
    float sphere_radius = a_scale.d0 * a_prim->sphere.radius;
    Vector3 box_s = b_scale * b_prim->box.dim * 0.5f;

    Vector3 tmp{}, center{}, clamped{}, deepest{};
    Vector3 pos{}, normal{};
    float dist, closest;

    // Box->Sphere as vector within box
    tmp = a_pos - b_pos;
    center = b_rot.rotateVec(tmp);
    // Clamp to within the box space
    clamped = center;
    clamped.x = clamp(clamped.x, -box_s.x, box_s.x);
    clamped.y = clamp(clamped.y, -box_s.y, box_s.y);
    clamped.z = clamp(clamped.z, -box_s.z, box_s.z);
    // Distance between clamped and sphere center
    deepest = center;
    tmp = clamped - center;
    dist = tmp.length();

    if (dist - sphere_radius > 0.f) {
      result.type = ContactType::None;
      return result;
    }

    // Sphere center is inside the box
    uint32_t k;
    if (dist <= 1e-12) {
      closest = (box_s.x + box_s.y + box_s.z) * 2.f;

      // Find the closest box face
      for (uint32_t i = 0; i < 6; i++) {
        float c = fabsf((i % 2 ? 1.f : -1.f) * box_s[i / 2] - center[i / 2]);
        if (c < closest) {
          closest = c;
          k = i;
        }
      }

      Vector3 nearest = Vector3::zero();
      nearest[k / 2] = (k % 2 ? 1.f : -1.f);

      pos = center;
      pos += nearest * (sphere_radius - closest) / 2.f;
      normal = b_rot.rotateVec(nearest);
      dist = -closest;
    } else {
      deepest += sphere_radius * tmp;
      pos = 0.5f * clamped;
      pos += 0.5f * deepest;
      normal = b_rot.rotateVec(tmp);
    }

    tmp = b_rot.rotateVec(pos);
    float d = dist - sphere_radius;
    SphereContact contact {
      .normal = -normal,
        .pt = tmp + b_pos,
        .depth = -d,
    };
    result.type = ContactType::Sphere;
    result.sphere = contact;
    return result;
  }
  case NarrowphaseTest::PlaneCapsule: {
    assert(b_prim->type == CollisionPrimitive::Type::Capsule);
    assert(b_scale.d0 == b_scale.d1);
    ConvexContact cv_contact = {};
    NarrowphaseResult result = {};
    result.type = ContactType::Convex;

    float cap_radius = b_scale.d0;
    float cap_len = b_scale.d2; // half height

    constexpr Vector3 base_normal = { 0, 0, 1 };
    Vector3 cap_axis = b_rot.rotateVec(base_normal);
    Vector3 cap_p1 = b_pos - cap_axis * cap_len;
    Vector3 cap_p2 = b_pos + cap_axis * cap_len;

    // Do plane / capsule collision test
    //  check the collision between the two spheres
    auto [is_contact_a, contact_a] = spherePlaneContact(
        cap_p1, a_pos, a_rot, cap_radius);
    auto [is_contact_b, contact_b] = spherePlaneContact(
        cap_p2, a_pos, a_rot, cap_radius);
    // Negate since objects a and b are swapped
    contact_a.normal *= -1.f;
    contact_b.normal *= -1.f;

    int32_t num_contacts = 0;
    if (is_contact_a) {
      cv_contact.contactPoints[num_contacts] = contact_a.pt;
      cv_contact.penetrationDepths[num_contacts] = contact_a.depth;
      cv_contact.normals[num_contacts] = contact_a.normal;
      num_contacts++;
    }
    if (is_contact_b) {
      cv_contact.contactPoints[num_contacts] = contact_b.pt;
      cv_contact.penetrationDepths[num_contacts] = contact_b.depth;
      cv_contact.normals[num_contacts] = contact_b.normal;
      num_contacts++;
    }

    cv_contact.numContactPoints = num_contacts;
    result.convex = cv_contact;
    return result;
  } break;
  case NarrowphaseTest::SpherePlane: {
    float sphere_radius;
    {
      auto sphere = a_prim->sphere;
      assert(a_scale.d0 == a_scale.d1 && a_scale.d0 == a_scale.d2);
      sphere_radius = a_scale.d0 * sphere.radius;
    }

    auto [is_contact, sphere_contact] = spherePlaneContact(
        a_pos, b_pos, b_rot, sphere_radius);
    if (!is_contact) {
      NarrowphaseResult result;
      result.type = ContactType::None;
      return result;
    }

    NarrowphaseResult result;
    result.type = ContactType::Sphere;
    result.sphere = sphere_contact;
    result.aVertices = nullptr;
    result.bVertices = nullptr;
    result.aVertices = nullptr;
    result.bVertices = nullptr;
    result.aHalfEdges = nullptr;
    result.bHalfEdges = nullptr;
    result.aFaceHedgeRoots = nullptr;
    result.bFaceHedgeRoots = nullptr;
    return result;
  } break;
  case NarrowphaseTest::HullPlane: {
  // Get half edge mesh for entity a (the hull)
  const auto &a_he_mesh = a_prim->hull.halfEdgeMesh;

  assert(a_he_mesh.numFaces < max_num_tmp_faces);
  assert(a_he_mesh.numVertices <  max_num_tmp_vertices);

  HullState a_hull_state = makeHullState(BOT_GPU_COND(mwgpu_lane_id,)
      a_he_mesh, a_pos, a_rot, a_scale,
      txfm_vertex_buffer, txfm_face_buffer);

  BOT_GPU_COND(__syncwarp(0xFFFF'FFFF));

  constexpr Vector3 base_normal = { 0, 0, 1 };

  Vector3 plane_normal = b_rot.rotateVec(base_normal);

  Plane plane {
    plane_normal,
      dot(plane_normal, b_pos),
  };

  const SATResult sat = doSATPlane(
      BOT_GPU_COND(mwgpu_lane_id,) plane, a_hull_state);

  NarrowphaseResult result;
  result.type = sat.type;
  result.sat = sat.contact;
#ifdef BOT_GPU
  result.aVertices = a_he_mesh.vertices;
  result.bVertices = nullptr;
#else
  result.aVertices = a_hull_state.mesh.vertices;
  result.bVertices = nullptr;
#endif
    result.aHalfEdges = a_hull_state.mesh.halfEdges;
    result.bHalfEdges = nullptr;
    result.aFaceHedgeRoots = a_hull_state.mesh.faceBaseHalfEdges;
    result.bFaceHedgeRoots = nullptr;
    return result;
  } break;
  default: BOT_UNREACHABLE();
  }
}

BOT_ALWAYS_INLINE static inline void generateContacts(
    SimRT &rt,
    World *world,
    NarrowphaseResult narrowphase_result,
    BodyRef a_loc, BodyRef b_loc,
#ifdef BOT_GPU
    Vector3 a_pos, Quat a_rot, Diag3x3 a_scale,
    Vector3 b_pos, Quat b_rot, Diag3x3 b_scale,
#endif
    void *thread_tmp_storage_a, void *thread_tmp_storage_b)
{
  switch (narrowphase_result.type) {
    case ContactType::None: {
      return;
    } break;
    case ContactType::Sphere: {
      SphereContact sphere_contact = narrowphase_result.sphere;

      addSinglePointContact(rt, world, sphere_contact.pt, sphere_contact.normal,
          sphere_contact.depth, b_loc, a_loc);
    } break;
    case ContactType::Convex: {
      ConvexContact convex_contact = narrowphase_result.convex;
      for (int i=0; i<convex_contact.numContactPoints; i++) {
        addSinglePointContact(rt, world, convex_contact.contactPoints[i],
            convex_contact.normals[i],
            convex_contact.penetrationDepths[i],
            b_loc, a_loc);
      }
    } break;
    case ContactType::SATPlane: {
      // Plane is always b, always reference
      BodyRef ref_loc = b_loc;
      BodyRef other_loc = a_loc;

#ifdef BOT_GPU
      Mat3x4 hull_txfm = Mat3x4::fromTRS(a_pos, a_rot, a_scale);
#endif

      Plane plane {
        narrowphase_result.sat.normal,
          narrowphase_result.sat.planeDOrSeparation,
      };

      // Create plane contact
      Manifold manifold = createFacePlaneContact(
          plane,
          int32_t(narrowphase_result.sat.incidentFaceIdxOrEdgeIdxB),
          narrowphase_result.aVertices,
          narrowphase_result.aHalfEdges,
          narrowphase_result.aFaceHedgeRoots,
          (Vector3 *)thread_tmp_storage_a,
          (float *)thread_tmp_storage_b,
#ifdef BOT_GPU
          hull_txfm,
#endif
          { 0, 0, 0, },
          { 1, 0, 0, 0 });

        // Sadly there are cases where two objects are just barely
        // touching and post contact clipping all the clipped contacts
        // are just barely separated due to FP32. For now just don't
        // make a Contact in this situation.
        if (manifold.numContactPoints > 0) {
          addManifoldContacts(rt, world, manifold, ref_loc, other_loc);
        }
      } break;
    case ContactType::SATFace: {
      const Vector3 *ref_vertices;
      const Vector3 *other_vertices;
      const HalfEdge *ref_hedges;
      const HalfEdge *other_hedges;
      const uint32_t *ref_face_hedges;
      const uint32_t *other_face_hedges;

#ifdef BOT_GPU
      Mat3x4 ref_txfm;
      Mat3x4 other_txfm;
#endif

      uint32_t ref_face_idx_and_ref_mask =
        narrowphase_result.sat.refFaceIdxOrEdgeIdxA;
      uint32_t incident_face_idx =
        narrowphase_result.sat.incidentFaceIdxOrEdgeIdxB;

      uint32_t ref_face_idx = ref_face_idx_and_ref_mask & 0x7FFF'FFFF;
      bool a_is_ref = ref_face_idx == ref_face_idx_and_ref_mask;

      BodyRef ref_loc, other_loc;
      if (a_is_ref) {
        ref_loc = a_loc;
        other_loc = b_loc;
        ref_vertices = narrowphase_result.aVertices;
        other_vertices = narrowphase_result.bVertices;
        ref_hedges = narrowphase_result.aHalfEdges;
        other_hedges = narrowphase_result.bHalfEdges;
        ref_face_hedges = narrowphase_result.aFaceHedgeRoots;
        other_face_hedges = narrowphase_result.bFaceHedgeRoots;
#ifdef BOT_GPU
        ref_txfm = Mat3x4::fromTRS(a_pos, a_rot, a_scale);
        other_txfm = Mat3x4::fromTRS(b_pos, b_rot, b_scale);
#endif
      } else {
        ref_loc = b_loc;
        other_loc = a_loc;
        ref_vertices = narrowphase_result.bVertices;
        other_vertices = narrowphase_result.aVertices;
        ref_hedges = narrowphase_result.bHalfEdges;
        other_hedges = narrowphase_result.aHalfEdges;
        ref_face_hedges = narrowphase_result.bFaceHedgeRoots;
        other_face_hedges = narrowphase_result.aFaceHedgeRoots;
#ifdef BOT_GPU
        ref_txfm = Mat3x4::fromTRS(b_pos, b_rot, b_scale);
        other_txfm = Mat3x4::fromTRS(a_pos, a_rot, a_scale);
#endif
      }

      Plane ref_plane {
        narrowphase_result.sat.normal,
          narrowphase_result.sat.planeDOrSeparation,
      };

      // Create face contact
      Manifold manifold = createFaceContact(
          ref_plane,
          int32_t(ref_face_idx),
          int32_t(incident_face_idx),
          ref_vertices,
          other_vertices,
          ref_hedges,
          other_hedges,
          ref_face_hedges,
          other_face_hedges,
          thread_tmp_storage_a, thread_tmp_storage_b,
#ifdef BOT_GPU
          ref_txfm,
          other_txfm,
#endif
          { 0, 0, 0, },
          { 1, 0, 0, 0 });

        // Sadly there are cases where two objects are just barely
        // touching and post contact clipping all the clipped contacts
        // are just barely separated due to FP32. For now just don't
        // make a Contact in this situation.
        if (manifold.numContactPoints > 0) {
          addManifoldContacts(rt, world, manifold, ref_loc, other_loc);
        }
      } break;
    case ContactType::SATEdge: {
      // A is always reference
      BodyRef ref_loc = a_loc;
      BodyRef other_loc = b_loc;

      // Create edge contact
      Manifold manifold = createEdgeContact(
          narrowphase_result.sat.normal,
          narrowphase_result.sat.planeDOrSeparation,
          int32_t(narrowphase_result.sat.refFaceIdxOrEdgeIdxA),
          int32_t(narrowphase_result.sat.incidentFaceIdxOrEdgeIdxB),
          narrowphase_result.aVertices,
          narrowphase_result.bVertices,
          narrowphase_result.aHalfEdges,
          narrowphase_result.bHalfEdges,
#ifdef BOT_GPU
          a_pos, a_rot, a_scale,
          b_pos, b_rot, b_scale,
#endif
          { 0, 0, 0 }, { 1, 0, 0, 0 });

      addManifoldContacts(rt, world, manifold, ref_loc, other_loc);
    } break;
    default: BOT_UNREACHABLE();
  }
}

void runNarrowphase(
    SimRT &rt,
    World *world,
    NarrowphaseCandidate *candidate_collision
    BOT_GPU_COND(,
      const int32_t mwgpu_warp_id,
      const int32_t mwgpu_lane_id,
      bool lane_active))
{
#ifdef BOT_GPU
    const int32_t num_smem_bytes_per_warp =
      GPU_DEFAULT_SM_CFG::NUM_SMEM_BYTES_PER_WARP;
    const int32_t num_smem_floats = num_smem_bytes_per_warp / sizeof(float);
    int32_t num_vertex_floats = num_smem_floats - gpuImpl::numPlaneFloats;
    int32_t max_num_vertices = num_vertex_floats / 3;

    constexpr int32_t max_num_tmp_faces = gpuImpl::maxNumPlanes;
    int32_t max_num_tmp_vertices = max_num_vertices;

    Plane tmp_faces_buffer[max_num_tmp_faces];

    Plane * smem_faces_buffer;
    Vector3 * smem_vertices_buffer;
    {
      auto smem_buf = gpuSMem();
      char *warp_smem_base =
        smem_buf + num_smem_bytes_per_warp * mwgpu_warp_id;

      smem_faces_buffer = (Plane *)warp_smem_base;
      smem_vertices_buffer =
        (Vector3 *)(smem_faces_buffer + gpuImpl::maxNumPlanes);
    }
#else

    constexpr int32_t max_num_tmp_faces = 512;
    constexpr int32_t max_num_tmp_vertices = 512;
    Plane tmp_faces_buffer[max_num_tmp_faces];
    Vector3 tmp_vertices_buffer[max_num_tmp_vertices];
#endif

  uint32_t raw_type_a, raw_type_b;
  Vector3 a_pos;
  Vector3 b_pos;
  Quat a_rot;
  Quat b_rot;
  Diag3x3 a_scale;
  Diag3x3 b_scale;
  const CollisionPrimitive *a_prim;
  const CollisionPrimitive *b_prim;
  BodyRef a_loc;
  BodyRef b_loc;

  if (!candidate_collision) {
#ifdef BOT_GPU
    lane_active = false;
#else
    return;
#endif
  } else {
    a_loc = candidate_collision->a;
    b_loc = candidate_collision->b;

    const ObjectManager &obj_mgr = *(world->physics.objMgr);

    uint32_t a_prim_idx, b_prim_idx;
    {
      i32 a_obj = *a_loc.objectID;
      i32 b_obj = *b_loc.objectID;

      const uint32_t a_prim_offset =
        obj_mgr.rigidBodyPrimitiveOffsets[a_obj];
      const uint32_t b_prim_offset  =
        obj_mgr.rigidBodyPrimitiveOffsets[b_obj];

      a_prim_idx = a_prim_offset + candidate_collision->aPrim;
      b_prim_idx = b_prim_offset + candidate_collision->bPrim;
    }

    a_prim = &obj_mgr.collisionPrimitives[a_prim_idx];
    b_prim = &obj_mgr.collisionPrimitives[b_prim_idx];

    raw_type_a = static_cast<uint32_t>(a_prim->type);
    raw_type_b = static_cast<uint32_t>(b_prim->type);

    // Swap a & b to be properly ordered based on object type
    if (raw_type_a > raw_type_b) {
      std::swap(a_loc, b_loc);
      std::swap(a_prim, b_prim);
      std::swap(a_prim_idx, b_prim_idx);
      std::swap(raw_type_a, raw_type_b);
    }

    a_pos = *a_loc.position;
    b_pos = *b_loc.position;
    a_rot = *a_loc.rotation;
    b_rot = *b_loc.rotation;
    a_scale = *a_loc.scale;
    b_scale = *b_loc.scale;

    {
      AABB a_obj_aabb = obj_mgr.primitiveAABBs[a_prim_idx];
      AABB b_obj_aabb = obj_mgr.primitiveAABBs[b_prim_idx];

      // TODO: Have a better way of handling this capsule edge case.
      AABB a_world_aabb = [&]() {
        if (raw_type_a == (uint32_t)CollisionPrimitive::Type::Capsule) {
          assert(a_scale.d0 == a_scale.d1);

          float r = a_scale.d0;
          float half_h = a_scale.d2;

          AABB capsule_aabb = {
            .pMin = { -r, -r, -(r + half_h) },
            .pMax = { r, r, r + half_h },
          };

          return capsule_aabb.applyTRS(a_pos, a_rot, { 1.f, 1.f, 1.f });
        } else {
          return a_obj_aabb.applyTRS(a_pos, a_rot, a_scale);
        }
      } ();

      AABB b_world_aabb = [&]() {
        if (raw_type_b == (uint32_t)CollisionPrimitive::Type::Capsule) {
          assert(b_scale.d0 == b_scale.d1);

          float r = b_scale.d0;
          float half_h = b_scale.d2;

          AABB capsule_aabb = {
            .pMin = { -r, -r, -(r + half_h) },
            .pMax = { r, r, r + half_h },
          };

          return capsule_aabb.applyTRS(b_pos, b_rot, { 1.f, 1.f, 1.f });
        } else {
          return b_obj_aabb.applyTRS(b_pos, b_rot, b_scale);
        }
      } ();

      if (!a_world_aabb.intersects(b_world_aabb)) {
#ifdef BOT_GPU
        lane_active = false;
#else
        return;
#endif
      }
    }
  }

#ifdef BOT_GPU
  const uint32_t active_mask = __ballot_sync(0xFFFF'FFFF, lane_active);

  if (active_mask == 0) {
    return;
  }
#endif

  const NarrowphaseTest test_type {raw_type_a | raw_type_b};

#ifdef BOT_GPU
  NarrowphaseResult thread_result;

#pragma unroll
    for (int32_t leader_idx = 0; leader_idx < 32; leader_idx++) {
      if (!__shfl_sync(0xFFFF'FFFF, lane_active, leader_idx)) {
        continue;
      }

      auto warp_test_type = (NarrowphaseTest)__shfl_sync(
          0xFFFF'FFFF, (uint32_t)test_type, leader_idx);

      Vector3 warp_a_pos {
        __shfl_sync(0xFFFF'FFFF, a_pos.x, leader_idx),
          __shfl_sync(0xFFFF'FFFF, a_pos.y, leader_idx),
          __shfl_sync(0xFFFF'FFFF, a_pos.z, leader_idx),
      };

      Vector3 warp_b_pos {
        __shfl_sync(0xFFFF'FFFF, b_pos.x, leader_idx),
          __shfl_sync(0xFFFF'FFFF, b_pos.y, leader_idx),
          __shfl_sync(0xFFFF'FFFF, b_pos.z, leader_idx),
      };

      Quat warp_a_rot {
        __shfl_sync(0xFFFF'FFFF, a_rot.w, leader_idx),
          __shfl_sync(0xFFFF'FFFF, a_rot.x, leader_idx),
          __shfl_sync(0xFFFF'FFFF, a_rot.y, leader_idx),
          __shfl_sync(0xFFFF'FFFF, a_rot.z, leader_idx),
      };

      Quat warp_b_rot {
        __shfl_sync(0xFFFF'FFFF, b_rot.w, leader_idx),
          __shfl_sync(0xFFFF'FFFF, b_rot.x, leader_idx),
          __shfl_sync(0xFFFF'FFFF, b_rot.y, leader_idx),
          __shfl_sync(0xFFFF'FFFF, b_rot.z, leader_idx),
      };

      Diag3x3 warp_a_scale {
        __shfl_sync(0xFFFF'FFFF, a_scale.d0, leader_idx),
          __shfl_sync(0xFFFF'FFFF, a_scale.d1, leader_idx),
          __shfl_sync(0xFFFF'FFFF, a_scale.d2, leader_idx),
      };

      Diag3x3 warp_b_scale {
        __shfl_sync(0xFFFF'FFFF, b_scale.d0, leader_idx),
          __shfl_sync(0xFFFF'FFFF, b_scale.d1, leader_idx),
          __shfl_sync(0xFFFF'FFFF, b_scale.d2, leader_idx),
      };

      auto warp_a_prim = (CollisionPrimitive *)__shfl_sync(0xFFFF'FFFF,
          (uint64_t)a_prim, leader_idx);

      auto warp_b_prim = (CollisionPrimitive *)__shfl_sync(0xFFFF'FFFF,
          (uint64_t)b_prim, leader_idx);

      NarrowphaseResult warp_result = narrowphaseDispatch(
          mwgpu_lane_id,
          warp_test_type,
          warp_a_pos, warp_b_pos,
          warp_a_rot, warp_b_rot,
          warp_a_scale, warp_b_scale,
          warp_a_prim, warp_b_prim,
          max_num_tmp_vertices, max_num_tmp_faces,
          smem_vertices_buffer, smem_faces_buffer);

      if (mwgpu_lane_id == leader_idx) {
        thread_result = warp_result;
      }
  }

  __syncwarp(0xFFFF'FFFF);

  if (lane_active) {
    generateContacts(rt, world, thread_result,
        a_loc, b_loc,
        a_pos, a_rot, a_scale,
        b_pos, b_rot, b_scale,
        tmp_faces_buffer,
        tmp_faces_buffer + max_num_tmp_faces / 2);

  }
#else
  NarrowphaseResult result = narrowphaseDispatch(
      test_type,
      a_pos, b_pos,
      a_rot, b_rot,
      a_scale, b_scale,
      a_prim, b_prim,
      max_num_tmp_vertices, max_num_tmp_faces,
      tmp_vertices_buffer, tmp_faces_buffer);

  generateContacts(rt, world, result,
      a_loc, b_loc,
      tmp_faces_buffer,
      tmp_faces_buffer + max_num_tmp_faces / 2);
#endif
}
  
}
