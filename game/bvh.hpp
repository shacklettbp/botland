#pragma once

#include <cassert>
#include "physics_obj.hpp"
#include "physics_store.hpp"

namespace bot {

// For now, let's just port this as a test case. Anyway, the BVH/broadphase
// has never really been a bottleneck in the physics code.
struct BVH {
  void init(Runtime &rt,
      const ObjectManager *obj_mgr,
      int64_t max_leaves,
      float leaf_velocity_expansion,
      float leaf_accel_expansion,
      MemArena &persistent_arena);

  inline LeafID reserveLeaf(BodyID e, int32_t obj_id);
  inline AABB getLeafAABB(LeafID leaf_id) const;

  template <typename Fn>
  inline void findIntersecting(const AABB &aabb, Fn &&fn) const;

  template <typename Fn>
  inline void findLeafIntersecting(LeafID leaf_id, Fn &&fn) const;

  BodyID traceRay(Vector3 o,
      Vector3 d,
      float *out_hit_t,
      Vector3 *out_hit_normal,
      float t_max = float(INFINITY));

  void updateLeafPosition(LeafID leaf_id,
      const Vector3 &pos,
      const Quat &rot,
      const Diag3x3 &scale,
      const Vector3 &linear_vel,
      const AABB &obj_aabb,
      bool is_capsule = false);

  AABB expandLeaf(LeafID leaf_id,
      const Vector3 &linear_vel);

  void refitLeaf(LeafID leaf_id, const AABB &leaf_aabb);

  inline void rebuildOnUpdate();
  void updateTree();

  inline void clearLeaves();

private:
  static constexpr int32_t sentinel_ = 0xFFFF'FFFF_i32;

  struct Node {
    float minX[4];
    float minY[4];
    float minZ[4];
    float maxX[4];
    float maxY[4];
    float maxZ[4];
    int32_t children[4];
    int32_t parentID;

    inline bool isLeaf(int64_t child) const;
    inline int32_t leafIDX(int64_t child) const;

    inline void setLeaf(int64_t child, int32_t idx);
    inline void setInternal(int64_t child, int32_t internal_idx);
    inline bool hasChild(int64_t child) const;
    inline void clearChild(int64_t child);
  };

  // FIXME: evaluate whether storing this in-line in the tree
  // makes sense or if we should force a lookup through the entity ID
  struct LeafTransform {
    Vector3 pos;
    Quat rot;
    Diag3x3 scale;
  };

  inline int64_t numInternalNodes(int64_t num_leaves) const;

  void rebuild();
  void refit(LeafID *leaf_ids, int64_t num_moved);

  bool traceRayIntoLeaf(int32_t leaf_idx,
      Vector3 world_ray_o,
      Vector3 world_ray_d,
      float t_min,
      float t_max,
      float *hit_t,
      Vector3 *hit_normal);

  Node *nodes_;
  int64_t num_nodes_;
  int64_t num_allocated_nodes_;
  BodyID *leaf_entities_;
  const ObjectManager *obj_mgr_;
  int32_t *leaf_obj_ids_;
  AABB *leaf_aabbs_; // FIXME: remove this, it's duplicated data
  LeafTransform  *leaf_transforms_;
  uint32_t *leaf_parents_;
  int32_t *sorted_leaves_;
  AtomicI32 num_leaves_ = 0;
  int32_t num_allocated_leaves_;
  float leaf_velocity_expansion_;
  float leaf_accel_expansion_;
  bool force_rebuild_;
};
  
}

#include "bvh.inl"
