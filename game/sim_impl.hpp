#pragma once

#include "physics.hpp"
#include "rt/rand.hpp"

namespace bot {

inline constexpr i32 NUM_BODIES_PER_CHUNK = 65536;
inline constexpr i32 MAX_NUM_BODY_CHUNKS = 65536;

struct BodyChunkHandle {
  u32 hdl = 0;
};

struct BodyGroupHandle {
};

struct alignas(256) BodiesChunk {
  Vector3 positions[NUM_BODIES_PER_CHUNK];
  BodyGroupHandle bodyGroups[NUM_BODIES_PER_CHUNK];
};

struct PhysicsWorld {
  MemArena persistentArena = {};

  u64 worldID = 0;

  BodyChunkHandle worldBodiesBaseChunkHdl = {};
  u32 bodiesBaseChunkOffset = 0;
  u32 numBodies = 0;
};

struct PhysicsSystem {
  MemArena arena = {};

  alignas(BOT_CACHE_LINE) u32 totalNumContacts = 0;
  alignas(BOT_CACHE_LINE) u32 totalNumBodies = 0;
  BodiesChunk bodies[MAX_NUM_BODY_CHUNKS] = {};
};

}
