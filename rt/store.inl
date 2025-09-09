#include "utils.hpp"

namespace bot {

template <typename ElemT, int ItemsPerChunk>
void TemporaryStore<ElemT, ItemsPerChunk>::init(Runtime &rt, MemArena &arena_in)
{
  stateHdl = rt.stateHandle();
  arena = &arena_in;
}

template <typename ElemT, int ItemsPerChunk>
void TemporaryStore<ElemT, ItemsPerChunk>::add(ElemT contact)
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
        chunks = (Chunk **)arenaAlloc(stateHdl, *arena,
          sizeof(Chunk *) * NUM_INIT_CHUNK_PTRS, alignof(Chunk *));
        chunksArrayCapacity = NUM_INIT_CHUNK_PTRS;
      } else {
        i32 new_capacity = chunksArrayCapacity * 2;
        auto **new_chunks = (Chunk **)arenaAlloc(stateHdl, *arena,
            sizeof(Chunk *) * new_capacity, alignof(Chunk *));

        copyN<Chunk *>(new_chunks, chunks, chunksArrayCapacity);
        chunks = new_chunks;
        chunksArrayCapacity = new_capacity;
      }
    }

    chunks[chunk_idx] = (Chunk *)arenaAlloc(stateHdl, *arena,
      sizeof(Chunk), alignof(Chunk));

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

#if 0
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
#endif

template <typename ElemT, int ItemsPerChunk>
ElemT * TemporaryStore<ElemT, ItemsPerChunk>::get(i32 idx)
{
  u32 chunk_idx = idx / ItemsPerChunk;
  u32 sub_idx = idx % ItemsPerChunk;
  Chunk *chk = chunks[chunk_idx];
  return &chk->items[sub_idx];
}

template <typename ElemT, int ItemsPerChunk>
typename TemporaryStore<ElemT, ItemsPerChunk>::Iterator::reference
TemporaryStore<ElemT, ItemsPerChunk>::Iterator::operator*() const
{
  u32 chunk_idx = currentIdx / ItemsPerChunk;
  u32 sub_idx = currentIdx % ItemsPerChunk;
  Chunk *chk = store->chunks[chunk_idx];
  return chk->items[sub_idx];
}

template <typename ElemT, int ItemsPerChunk>
typename TemporaryStore<ElemT, ItemsPerChunk>::Iterator::pointer
TemporaryStore<ElemT, ItemsPerChunk>::Iterator::operator->() const
{
  u32 chunk_idx = currentIdx / ItemsPerChunk;
  u32 sub_idx = currentIdx % ItemsPerChunk;
  Chunk *chk = store->chunks[chunk_idx];
  return &chk->items[sub_idx];
}

template <typename ElemT, int ItemsPerChunk>
typename TemporaryStore<ElemT, ItemsPerChunk>::Iterator&
TemporaryStore<ElemT, ItemsPerChunk>::Iterator::operator++()
{
  currentIdx++;
  return *this;
}

template <typename ElemT, int ItemsPerChunk>
typename TemporaryStore<ElemT, ItemsPerChunk>::Iterator
TemporaryStore<ElemT, ItemsPerChunk>::Iterator::operator++(int)
{
  Iterator tmp = *this;
  ++(*this);
  return tmp;
}

template <typename ElemT, int ItemsPerChunk>
bool TemporaryStore<ElemT, ItemsPerChunk>::Iterator::operator==(const Iterator& other) const
{
  return currentIdx == other.currentIdx;
}

template <typename ElemT, int ItemsPerChunk>
bool TemporaryStore<ElemT, ItemsPerChunk>::Iterator::operator!=(const Iterator& other) const
{
  return !(*this == other);
}

template <typename ElemT, int ItemsPerChunk>
typename TemporaryStore<ElemT, ItemsPerChunk>::Iterator
TemporaryStore<ElemT, ItemsPerChunk>::begin()
{
  return Iterator(this, 0);
}

template <typename ElemT, int ItemsPerChunk>
typename TemporaryStore<ElemT, ItemsPerChunk>::Iterator
TemporaryStore<ElemT, ItemsPerChunk>::end()
{
  u32 offset = u32(chunksRange);
  return Iterator(this, offset);
}
  
inline constexpr i32 NUM_INIT_CHUNK_PTRS = 16;

template <typename ID, typename ChunkT, typename PtrT>
void PersistentStore<ID, ChunkT, PtrT>::init(Runtime &rt, MemArena &arena_in)
{
  stateHdl = rt.stateHandle();
  arena = &arena_in;

  chunks = (StoreChunk **)arenaAlloc(stateHdl, *arena, 
    sizeof(StoreChunk *) * NUM_INIT_CHUNK_PTRS, alignof(StoreChunk *));

  chunksCapacity = NUM_INIT_CHUNK_PTRS;
  lastChunkNumAllocated = ChunkT::SIZE;
}

template <typename ID, typename ChunkT, typename PtrT>
PtrT PersistentStore<ID, ChunkT, PtrT>::create(u32 type_id)
{
  if (freeList.chunk != 0) {
    ID actor = freeList;

    StoreChunk *chunk = getChunk(actor);

    freeList = chunk->user.id[actor.offset];

    actor.type = type_id;
    
    chunk->user.id[actor.offset] = actor;
    chunk->activeMask[actor.offset / 32] |= (1 << (actor.offset % 32));

    return PtrT(&chunk->user, actor.offset);
  }

  if (lastChunkNumAllocated == ChunkT::SIZE) {
    if (chunksCapacity == numChunks) {
      i32 new_capacity = chunksCapacity * 2;

      StoreChunk **new_chunks = (StoreChunk **)arenaAlloc(stateHdl, *arena,
          sizeof(StoreChunk *) * new_capacity, alignof(StoreChunk *));

      copyN<StoreChunk *>(new_chunks, chunks, chunksCapacity);

      chunks = new_chunks;
      chunksCapacity = new_capacity;
    }

    StoreChunk *new_chunk = chunks[numChunks] = (StoreChunk *)arenaAlloc(stateHdl, *arena,
        sizeof(StoreChunk), alignof(StoreChunk));

    BOT_UNROLL
    for (i32 i = 0; i < ChunkT::SIZE / 32; i++) {
      new_chunk->activeMask[i] = 0;
    }

    numChunks += 1;
    lastChunkNumAllocated = 0;
  }

  i32 chunk_idx = numChunks - 1;
  i32 offset = lastChunkNumAllocated += 1;

  chk(offset <= ChunkT::SIZE);

  StoreChunk *chunk = chunks[chunk_idx];
  
  u32 chunk_hdl = ((char *)chunk - (char *)getRuntimeState(stateHdl)) / alignof(StoreChunk);

  ID id = {
    .type = type_id,
    .offset = (u32)offset,
    .gen = 0,
    .chunk = chunk_hdl,
  };
  
  chunk->user.id[offset] = id;
  chunk->activeMask[offset / 32] |= (1 << (offset % 32));

  return PtrT(&chunk->user, offset);
}

template <typename ID, typename ChunkT, typename PtrT>
void PersistentStore<ID, ChunkT, PtrT>::destroy(ID actor)
{
  StoreChunk *chunk = getChunk(actor);

  if (actor.gen != chunk->user.id[actor.offset].gen) {
    return;
  }

  chunk->user.id[actor.offset].gen++;

  chunk->activeMask[actor.offset / 32] &= ~(1 << (actor.offset % 32));

  chunk->user.id[actor.offset] = freeList;
  freeList = actor;
}

template <typename ID, typename ChunkT, typename PtrT>
i32 PersistentStore<ID, ChunkT, PtrT>::size()
{
  i32 num_actors = 0;
  for (i32 i = 0; i < numChunks; i++) {
    for (i32 j = 0; j < ChunkT::SIZE / 32; j++) {
      num_actors += std::popcount(chunks[i]->activeMask[j]);
    }
  }

  return num_actors;
}

template <typename ID, typename ChunkT, typename PtrT>
PtrT PersistentStore<ID, ChunkT, PtrT>::get(ID actor, bool verify)
{
  StoreChunk *chunk = getChunk(actor);

  i32 offset = actor.offset;

  if (verify) {
    if (!chunk) {
      return PtrT(nullptr, 0);
    }

    u16 gen = chunk->user.id[offset].gen;

    if (actor.gen != gen) {
      return PtrT(nullptr, 0);
    }
  }

  return PtrT(&chunk->user, offset);
}

template <typename ID, typename ChunkT, typename PtrT>
PersistentStore<ID, ChunkT, PtrT>::StoreChunk * 
PersistentStore<ID, ChunkT, PtrT>::getChunk(ID id)
{
  if (id.chunk == 0) [[unlikely]] {
    return nullptr;
  }

  return (StoreChunk *)((char *)getRuntimeState(stateHdl) + (u64)id.chunk * alignof(StoreChunk));
}

template <typename ID, typename ChunkT, typename PtrT>
void PersistentStore<ID, ChunkT, PtrT>::Iterator::advance()
{
  while (chunkIdx < store->numChunks) {
    if (itemIdx >= ChunkT::SIZE) {
      chunkIdx++;
      itemIdx = 0;
      continue;
    }
    
    StoreChunk *chunk = store->chunks[chunkIdx];
    bool active = (chunk->activeMask[itemIdx / 32] & (1 << (itemIdx % 32))) != 0;
    
    if (active) {
      return;
    }
    
    itemIdx++;
  }
}

template <typename ID, typename ChunkT, typename PtrT>
typename PersistentStore<ID, ChunkT, PtrT>::Iterator::reference 
PersistentStore<ID, ChunkT, PtrT>::Iterator::operator*() const
{
  StoreChunk *chunk = store->chunks[chunkIdx];
  PtrT ptr(&chunk->user, itemIdx);
  return *ptr;
}

template <typename ID, typename ChunkT, typename PtrT>
typename PersistentStore<ID, ChunkT, PtrT>::Iterator&
PersistentStore<ID, ChunkT, PtrT>::Iterator::operator++()
{
  itemIdx++;
  advance();
  return *this;
}

template <typename ID, typename ChunkT, typename PtrT>
typename PersistentStore<ID, ChunkT, PtrT>::Iterator
PersistentStore<ID, ChunkT, PtrT>::Iterator::operator++(int)
{
  Iterator tmp = *this;
  ++(*this);
  return tmp;
}

template <typename ID, typename ChunkT, typename PtrT>
bool PersistentStore<ID, ChunkT, PtrT>::Iterator::operator==(const Iterator& other) const
{
  return chunkIdx == other.chunkIdx && itemIdx == other.itemIdx;
}

template <typename ID, typename ChunkT, typename PtrT>
bool PersistentStore<ID, ChunkT, PtrT>::Iterator::operator!=(const Iterator& other) const
{
  return !(*this == other);
}

template <typename ID, typename ChunkT, typename PtrT>
typename PersistentStore<ID, ChunkT, PtrT>::Iterator
PersistentStore<ID, ChunkT, PtrT>::begin()
{
  return Iterator(this, 0, 0);
}

template <typename ID, typename ChunkT, typename PtrT>
typename PersistentStore<ID, ChunkT, PtrT>::Iterator
PersistentStore<ID, ChunkT, PtrT>::end()
{
  return Iterator(this, numChunks, 0);
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
