#pragma once

#include "gas/gas_input.hpp"
#include "render.hpp"
#include <scene/scene.hpp>

namespace bot {

inline constexpr i32 NUM_BODIES_PER_CHUNK = 128;
inline constexpr i32 NUM_CONTACTS_PER_CHUNK = 32;
static_assert(NUM_BODIES_PER_CHUNK % 32 == 0);

// For the BVH
struct LeafID {
  int32_t id;
};

#define BODY_FIELDS(F) \
  F(Vector3, position) \
  F(Quat, rotation) \
  F(Vector3, linVelocity) \
  F(Vector3, angVelocity) \
  F(Diag3x3, scale) \
  F(RenderInstanceRef, renderInstance) \
  F(i32, objectID) \
  F(LeafID, leafID) \
  F(ResponseType, responseType)

BOT_PERSISTENT_STORE(Body, 128, BODY_FIELDS)

#undef BODY_FIELDS

struct ContactConstraint {
  static inline constexpr u32 MAX_POINTS = 4;

  BodyRef ref;
  BodyRef alt;
  Vector4 points[MAX_POINTS];
  i32 numPoints;
  Vector3 normal;
};

using ContactStore = TemporaryStore<ContactConstraint, 32>;

struct MidphaseCandidate {
  BodyRef a;
  BodyRef b;
  u32 offset;
  u32 count;
};

using MidphaseStore = TemporaryStore<MidphaseCandidate, 32>;

struct NarrowphaseCandidate {
  BodyRef a;
  BodyRef b;
  u32 aPrim;
  u32 bPrim;
};

using NarrowphaseStore = TemporaryStore<NarrowphaseCandidate, 32>;

template <typename ElemT, int ItemsPerChunk>
struct CompactedTemporaryStore {
  struct Item {
    u32 worldID;
    ElemT *elem;
  };

  u32 numItems;
  Item *items;
  u32 numWorlds;
  i32 *worldCounts;
  i32 *worldOffsets;

  MemArena *tmpArena;

  void init(
      Runtime &rt,
      u32 num_worlds,
      MemArena &persistent_arena,
      MemArena &tmp_arena);

  template <typename GetStoreT> // Returns CompactedTemporaryStore &
  void compact(Runtime &rt, TaskExec &exec, GetStoreT &&fn);
};

template <typename ID>
struct CompactedPersistentStore {
  struct Item {
    u32 worldID;
    ID id;
  };

  u32 numItems;
  Item *items;
  u32 numWorlds;
  i32 *worldCounts;
  i32 *worldOffsets;

  MemArena *tmpArena;

  void init(
      Runtime &rt,
      u32 num_worlds,
      MemArena &persistent_arena,
      MemArena &tmp_arena);

  template <typename GetStoreT>
  void compact(Runtime &rt, TaskExec &exec, GetStoreT &&fn);
};
  
}

#include "physics_store.inl"
