#pragma once

#include "fwd.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "err.hpp"
#include "sync.hpp"
#include "rt.hpp"

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

template <typename ID, typename ChunkT, typename PtrT>
struct PersistentStore {
  static_assert(ChunkT::SIZE % 32 == 0);

  static inline constexpr u32 CHUNK_SIZE = ChunkT::SIZE;

  struct StoreChunk {
    ChunkT user;
    u32 activeMask[ChunkT::SIZE / 32];
  };

  MemArena *arena = nullptr;
  StoreChunk **chunks = nullptr;
  i32 numChunks = 0;
  i32 chunksCapacity = 0;
  i32 lastChunkNumAllocated = 0;

  ID freeList = {};

  void init(Runtime &rt, MemArena &arena);
  PtrT create(Runtime &rt, u32 type_id);
  void destroy(Runtime &rt, ID actor);
  inline PtrT get(Runtime &rt, ID actor, bool verify = true);

  inline StoreChunk * getChunk(Runtime &rt, ID actor);

  i32 size();

  class Iterator {
  private:
    PersistentStore *store;
    Runtime *rt;
    i32 chunkIdx;
    i32 itemIdx;
    
    void advance();
    
  public:
    Iterator(PersistentStore *s, Runtime *r, i32 chunk, i32 item)
      : store(s), rt(r), chunkIdx(chunk), itemIdx(item) {
      advance();
    }
    
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<ID, PtrT>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type;
    
    reference operator*() const;
    Iterator& operator++();
    Iterator operator++(int);
    bool operator==(const Iterator& other) const;
    bool operator!=(const Iterator& other) const;
  };
  
  Iterator begin(Runtime &rt);
  Iterator end(Runtime &rt);
  
  struct Range {
    PersistentStore *store;
    Runtime *rt;
    
    Range(PersistentStore *s, Runtime *r) : store(s), rt(r) {}
    Iterator begin() { return store->begin(*rt); }
    Iterator end() { return store->end(*rt); }
  };
  
  Range iterate(Runtime &rt) { return Range(this, &rt); }
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

#define BOT_PERSISTENT_ID(NAME) \
  struct NAME { \
    u32 type   : 9 = 0; \
    u32 offset : 7 = 0; \
    u32 gen    : 16 = 0; \
    u32 chunk = 0; \
    static NAME none() \
    { \
      return NAME { \
        .type = 0, \
        .offset = 0, \
        .gen = 0, \
        .chunk = 0, \
      }; \
    } \
    inline bool operator==(NAME o) const \
    { \
      return type == o.type && offset == o.offset && \
        gen == o.gen && chunk == o.chunk; \
    } \
    inline GenericID toGeneric() const \
    { \
      return GenericID { type, offset, gen, chunk }; \
    } \
    static inline NAME fromGeneric(GenericID id) \
    { \
      return NAME { id.type, id.offset, id.gen, id.chunk }; \
    } \
    operator bool() const \
    { \
      return *this != NAME::none(); \
    } \
  };

#define BOT_PERSISTENT_STORE(name, ID, CHUNK_SIZE, FIELDS)\
  struct name##Ref;\
  struct name##Chunk { \
    static constexpr inline i32 SIZE = CHUNK_SIZE; \
    ID id[CHUNK_SIZE]; \
    FIELDS(BOT_PERSISTENT_STORE_CHUNK_DEF) \
    \
    inline name##Ref get(i32 offset); \
  }; \
  struct name##Ref { \
    ID id = {}; \
    FIELDS(BOT_PERSISTENT_STORE_REF_DEF) \
    const name##Ref * operator->() const { return this; } \
    name##Ref * operator->() { return this; } \
  }; \
  class name##Ptr { \
  public: \
    name##Ptr(name##Chunk *chunk, i32 offset) : chunk_(chunk), offset_(offset) {} \
    operator bool() const { return !!chunk_; } \
    name##Ref operator->() const \
    { \
      return name##Ref { \
        .id = chunk_->id[offset_], \
        FIELDS(BOT_PERSISTENT_STORE_PTR_DEREF_DEF) \
      }; \
    } \
  private: \
    name##Chunk *chunk_ = nullptr; \
    i32 offset_ = 0; \
  }; \
  using name##Store = PersistentStore<ID, name##Chunk, name##Ptr>;

#define BOT_PERSISTENT_STORE_CHUNK_DEF(type, name) type name[SIZE];
#define BOT_PERSISTENT_STORE_REF_DEF(type, name) type &name;
#define BOT_PERSISTENT_STORE_PTR_DEREF_DEF(type, name) .name = chunk_->name[offset_],

}

#include "store.inl"
