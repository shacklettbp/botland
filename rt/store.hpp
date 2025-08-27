#pragma once

#include "fwd.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "err.hpp"
#include "sync.hpp"

namespace bot {


template <typename ElemT, int ItemsPerChunk>
struct TemporaryStore {
  static_assert(ItemsPerChunk % 32 == 0);

  static inline constexpr u32 NUM_INIT_CHUNK_PTRS = 16;

  struct Chunk {
    ElemT items[ItemsPerChunk];
  };

  MemArena *arena = nullptr;
  Chunk **chunks = nullptr;
  i32 chunksArrayCapacity = 0;
  u32 chunksExpandLock = 0;
  u64 chunksRange = 0;

  void init(Runtime &rt, MemArena &arena);
  void add(Runtime &rt, ElemT contact);
  void clear();

  template <typename FnT>
  inline void iterate(FnT fn);

#ifdef BOT_GPU
  template <typename FnT>
  inline void warpIterate(FnT fn);
#endif

  ElemT *get(i32 idx);
};

template <typename ID, typename ChunkT, typename RefT>
struct PersistentStore {
  static_assert(ChunkT::SIZE % 32 == 0);

  static inline constexpr u32 CHUNK_SIZE = ChunkT::SIZE;

  using StoreID = ID;
  using StoreRef = RefT;

  struct StoreChunk {
    ChunkT user;
    ID nextFrees[ChunkT::SIZE];
    u16 gens[ChunkT::SIZE];
    u32 activeMask[ChunkT::SIZE / 32];
  };

  MemArena *arena = nullptr;
  StoreChunk **chunks = nullptr;
  i32 numChunks = 0;
  i32 chunksCapacity = 0;
  i32 lastChunkNumAllocated = 0;

  ID freeList = {};

  void init(Runtime &rt, MemArena &arena);
  ID create(Runtime &rt, u32 type_id);
  void destroy(Runtime &rt, ID actor);
  inline RefT get(Runtime &rt, ID actor, bool verify = true);

  inline StoreChunk * getChunk(Runtime &rt, ID actor);

  i32 size();
};

struct GenericID {
  u32 type : 9 = 0;
  u32 offset : 7 = 0;
  u32 gen : 16 = 0;
  u32 chunk = 0;

  static GenericID none()
  {
    return GenericID {
      .type = 0,
      .offset = 0,
      .gen = 0,
      .chunk = 0,
    };
  }

  inline bool operator==(GenericID o) const
  {
    return type == o.type && offset == o.offset &&
      gen == o.gen && chunk == o.chunk;
  }

  operator bool() const
  {
    return *this != GenericID::none();
  }
};

/* Used to create persistent store classes. Example:
 * #define ACTOR_FIELDS(F) \
 *  F(Vector3, position), \
 *  F(Quat, rotation), \
 *  F(Diag3x3, scale)
 *
 * BOT_PERSISTENT_STORE(Actor, 128, ACTOR_FIELDS);
 *
 * #undef ACTOR_FIELDS
 *
 * Will produce the following:
 * struct ActorID {
 *   u32 type : 9 = 0;
 *   u32 offset : 7 = 0;
 *   u32 gen : 16 = 0;
 *   u32 chunk = 0;
 * };
 *
 * struct ActorChunk {
 *   static constexpr inline i32 SIZE = 128;
 *   Vector3 position[SIZE];
 *   Quat rotation[SIZE];
 *   Diag3x3 scale[SIZE];
 * };
 *
 * struct ActorRef {
 *   Vector3 *position;
 *   Quat *rotation;
 *   Diag3x3 *scale;
 * };
 *
 * using ActorStore = PersistentStore<ActorID, ActorChunk, ActorRef>;
 */

#define BOT_PERSISTENT_STORE(name, CHUNK_SIZE, FIELDS)\
  struct name##ID { \
    u32 type   : 9 = 0; \
    u32 offset : 7 = 0; \
    u32 gen    : 16 = 0; \
    u32 chunk = 0; \
    static name##ID none() \
    { \
      return name##ID { \
        .type = 0, \
        .offset = 0, \
        .gen = 0, \
        .chunk = 0, \
      }; \
    } \
    inline bool operator==(name##ID o) const \
    { \
      return type == o.type && offset == o.offset && \
        gen == o.gen && chunk == o.chunk; \
    } \
    inline GenericID toGeneric() const \
    { \
      return GenericID { type, offset, gen, chunk }; \
    } \
    static inline name##ID fromGeneric(GenericID id) \
    { \
      return name##ID { id.type, id.offset, id.gen, id.chunk }; \
    } \
  }; \
  struct name##Ref;\
  struct name##Chunk { \
    static constexpr inline i32 SIZE = CHUNK_SIZE; \
    FIELDS(BOT_PERSISTENT_STORE_CHUNK_DEF) \
    \
    inline name##Ref get(i32 offset); \
  }; \
  struct name##Ref { \
    FIELDS(BOT_PERSISTENT_STORE_REF_DEF) \
    \
    operator bool() const { \
      return FIELDS(BOT_PERSISTENT_STORE_VALID_DEF) true; \
    } \
  }; \
  name##Ref name##Chunk::get(i32 offset) { \
    return name##Ref { \
      FIELDS(BOT_PERSISTENT_STORE_GETTER_DEF) \
    };\
  } \
  using name##Store = PersistentStore<name##ID, name##Chunk, name##Ref>;

#define BOT_PERSISTENT_STORE_CHUNK_DEF(type, name) type name[SIZE];
#define BOT_PERSISTENT_STORE_REF_DEF(type, name) type *name = nullptr;
#define BOT_PERSISTENT_STORE_GETTER_DEF(type, name) .name = &this->name[offset],
#define BOT_PERSISTENT_STORE_VALID_DEF(type, name) name != nullptr &&

}

#include "store.inl"
