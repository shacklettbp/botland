#include "utils.hpp"

namespace bot {

template <typename ElemT, int ItemsPerChunk>
void TemporaryStore<ElemT, ItemsPerChunk>::init(Runtime &rt, MemArena &arena_in)
{
  (void)rt;
  arena = &arena_in;
}

template <typename ElemT, int ItemsPerChunk>
void TemporaryStore<ElemT, ItemsPerChunk>::add(
    Runtime &rt,
    ElemT contact)
{
  AtomicU64Ref chunks_range_atomic(chunksRange);

  i32 chunk_idx, chunk_offset;
  while (true) {
    u64 cur_range = chunks_range_atomic.fetch_add_acq_rel(1);
    u32 offset = u32(cur_range);
    u32 size = u32(cur_range >> 32);

    if (offset < size) [[likely]] {
      chunk_idx = i32(offset / ItemsPerChunk);
      chunk_offset = i32(offset % ItemsPerChunk);

      break;
    }

    spinLock(&chunksExpandLock);
    cur_range = chunks_range_atomic.load_relaxed();
    offset = u32(cur_range);
    size = u32(cur_range >> 32);

    // Double check
    if (offset < size) {
      spinUnlock(&chunksExpandLock);
      continue;
    }

    offset = size;

    chunk_idx = i32(offset / ItemsPerChunk);
    chunk_offset = i32(offset % ItemsPerChunk);

    if (chunk_idx >= chunksArrayCapacity) {
      if (chunksArrayCapacity == 0) {
        chunks = rt.arenaAllocN<Chunk *>(
            *arena, ItemsPerChunk);
        chunksArrayCapacity = NUM_INIT_CHUNK_PTRS;
      } else {
        i32 new_capacity = chunksArrayCapacity * 2;
        auto **new_chunks = rt.arenaAllocN<Chunk *>(
            *arena, new_capacity);

        copyN<Chunk *>(new_chunks, chunks, chunksArrayCapacity);
        chunks = new_chunks;
        chunksArrayCapacity = new_capacity;
      }
    }

    chunks[chunk_idx] = 
      rt.arenaAlloc<Chunk>(*arena);

    u64 new_range = (u64(offset + ItemsPerChunk) << 32) | 
      u64(offset + 1);

    chunks_range_atomic.store<sync::release>(new_range);
    
    spinUnlock(&chunksExpandLock);

    break;
  }

  Chunk *chunk = chunks[chunk_idx];
  chunk->items[chunk_offset] = contact;
}

template <typename ElemT, int ItemsPerChunk>
void TemporaryStore<ElemT, ItemsPerChunk>::clear()
{
  chunksRange = 0;
  chunksArrayCapacity = 0;
  chunksExpandLock = 0;
  chunks = nullptr;
}

template <typename ElemT, int ItemsPerChunk>
template <typename FnT>
void TemporaryStore<ElemT, ItemsPerChunk>::iterate(FnT fn)
{
#if 0
  u32 used_chunks = (u32(chunksRange) + 
    ((u32)ItemsPerChunk-1)) / (u32)ItemsPerChunk;
#endif

  u32 offset = u32(chunksRange);

  for (u32 i = 0; i < offset; ++i) {
    u32 chunk_idx = i / ItemsPerChunk;
    u32 sub_idx = i % ItemsPerChunk;
    Chunk *chk = chunks[chunk_idx];
    fn(&chk->items[sub_idx]);
  }
}

#ifdef BOT_GPU
template <typename ElemT, int ItemsPerChunk>
template <typename FnT>
void TemporaryStore<ElemT, ItemsPerChunk>::warpIterate(FnT fn)
{
  u32 offset = u32(chunksRange);

  for (u32 i = threadIdx.x % 32;
      i < 32 * divideRoundUp((i32)offset, 32); i += 32) {
    ElemT *item = nullptr;

    if (i < offset) {
      u32 chunk_idx = i / ItemsPerChunk;
      u32 sub_idx = i % ItemsPerChunk;
      Chunk *chk = chunks[chunk_idx];
      item = &chk->items[sub_idx];
    }

    fn(item);
  }
}
#endif

template <typename ElemT, int ItemsPerChunk>
ElemT * TemporaryStore<ElemT, ItemsPerChunk>::get(i32 idx)
{
  u32 chunk_idx = idx / ItemsPerChunk;
  u32 sub_idx = idx % ItemsPerChunk;
  Chunk *chk = chunks[chunk_idx];
  return &chk->items[sub_idx];
}
  
inline constexpr i32 NUM_INIT_CHUNK_PTRS = 16;

template <typename ID, typename ChunkT, typename RefT>
void PersistentStore<ID, ChunkT, RefT>::init(Runtime &rt, MemArena &arena_in)
{
  arena = &arena_in;

  chunks = rt.arenaAllocN<StoreChunk *>(*arena, NUM_INIT_CHUNK_PTRS);

  chunksCapacity = NUM_INIT_CHUNK_PTRS;
  lastChunkNumAllocated = ChunkT::SIZE;
}

template <typename ID, typename ChunkT, typename RefT>
ID PersistentStore<ID, ChunkT, RefT>::create(Runtime &rt, u32 type_id)
{
  if (freeList.chunk != 0) {
    ID actor = freeList;

    StoreChunk *chunk = getChunk(rt, actor);

    freeList = chunk->nextFrees[actor.offset];

    actor.type = type_id;

    return actor;
  }

  if (lastChunkNumAllocated == ChunkT::SIZE) {
    if (chunksCapacity == numChunks) {
      i32 new_capacity = chunksCapacity * 2;

      StoreChunk **new_chunks =
          rt.arenaAllocN<StoreChunk *>(*arena, new_capacity);

      copyN<StoreChunk *>(new_chunks, chunks, chunksCapacity);

      chunks = new_chunks;
      chunksCapacity = new_capacity;
    }

    StoreChunk *new_chunk = chunks[numChunks] = rt.arenaAlloc<StoreChunk>(*arena);

    BOT_UNROLL
    for (i32 i = 0; i < ChunkT::SIZE / 32; i++) {
      new_chunk->activeMask[i] = 0;
    }

    numChunks += 1;
    lastChunkNumAllocated = 0;
  }

  i32 chunk_idx = numChunks - 1;
  i32 offset = lastChunkNumAllocated += 1;

  StoreChunk *chunk = chunks[chunk_idx];
  chunk->gens[offset] = 0;
  chunk->activeMask[offset / 32] |= (1 << (offset % 32));

  chk(offset <= ChunkT::SIZE);

  u32 chunk_hdl = ((char *)chunk - (char *)rt.state()) / alignof(StoreChunk);

  return ID {
    .type = type_id,
    .offset = (u32)offset,
    .gen = 0,
    .chunk = chunk_hdl,
  };
}

template <typename ID, typename ChunkT, typename RefT>
void PersistentStore<ID, ChunkT, RefT>::destroy(Runtime &rt, ID actor)
{
  StoreChunk *chunk = getChunk(rt, actor);

  if (actor.gen != chunk->gens[actor.offset]) {
    return;
  }

  chunk->gens[actor.offset]++;

  chunk->activeMask[actor.offset / 32] &= ~(1 << (actor.offset % 32));

  chunk->nextFrees[actor.offset] = freeList;
  freeList = actor;
}

template <typename ID, typename ChunkT, typename RefT>
i32 PersistentStore<ID, ChunkT, RefT>::size()
{
  i32 num_actors = 0;
  for (i32 i = 0; i < numChunks; i++) {
    for (i32 j = 0; j < ChunkT::SIZE / 32; j++) {
      num_actors += std::popcount(chunks[i]->activeMask[j]);
    }
  }

  return num_actors;
}

template <typename ID, typename ChunkT, typename RefT>
RefT PersistentStore<ID, ChunkT, RefT>::get(Runtime &rt, ID actor, bool verify)
{
  StoreChunk *chunk = getChunk(rt, actor);

  i32 offset = actor.offset;

  if (verify) {
    if (!chunk) {
      return {};
    }

    u16 gen = chunk->gens[offset];

    if (actor.gen != gen) {
      return {};
    }
  }

  return chunk->user.get(offset);
}

template <typename ID, typename ChunkT, typename RefT>
PersistentStore<ID, ChunkT, RefT>::StoreChunk * 
PersistentStore<ID, ChunkT, RefT>::getChunk(Runtime &rt, ID id)
{
  if (id.chunk == 0) [[unlikely]] {
    return nullptr;
  }

  return (StoreChunk *)((char *)rt.state() + (u64)id.chunk * alignof(StoreChunk));
}

#if 0
template <typename ID, typename ChunkT, typename RefT>
template <typename FnT>
void PersistentStore<ID, ChunkT, RefT>::iterate(Runtime &rt, FnT fn)
{
  i32 total_iters = numChunks * ChunkT::SIZE;

  for (i32 iter = 0; iter < total_iters; ++iter) {
    i32 i = iter / ChunkT::SIZE;
    i32 j = iter % ChunkT::SIZE;

    StoreChunk *chunk = chunks[i];

    bool active = (chunk->activeMask[j/32] & (1 << (j % 32)));

    if (active) {
      RefT ref = chunk->user.get(j);

      u32 chunk_hdl = ((char *)chunk - (char *)rt.state()) / alignof(StoreChunk);

      ID id = {
        .type = 0,
        .offset = (u32)j,
        .gen = chunk->gens[j],
        .chunk = chunk_hdl
      };

      fn(id, ref);
    }
  }
}
#endif

}
