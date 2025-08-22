#include "prims.hpp"
#include <rt/rt.hpp>

namespace bot {

template <typename ElemT, int ItemsPerChunk>
void CompactedTemporaryStore<ElemT, ItemsPerChunk>::init(
  Runtime &rt,
  u32 num_worlds,
  MemArena &persistent_arena,
  MemArena &tmp_arena)
{
  tmpArena = &tmp_arena;

  numItems = 0;
  items = nullptr;
  numWorlds = num_worlds;

  worldCounts = rt.arenaAllocN<i32>(persistent_arena, numWorlds);
  worldOffsets = rt.arenaAllocN<i32>(persistent_arena, numWorlds);
}

template <typename ElemT, int ItemsPerChunk>
template <typename GetStoreT>
void CompactedTemporaryStore<ElemT, ItemsPerChunk>::compact(
  Runtime &rt, TaskExec &exec, GetStoreT &&fn)
{
  u32 total_num_items = exec.reduceTask<Reduction::Sum>(
    rt, numWorlds, true,
    [&](i32 idx) {
      auto &store = fn(idx);
      u32 num_items = u32(store->chunksRange);
      store.worldCounts[idx] = (i32)num_items;
      return num_items;
    });

  TaskPrimitives::prefixSum(
      rt, exec, *tmpArena,
      worldCounts, worldOffsets,
      total_num_items);

  exec.serialTask(
    rt, [&]() {
      numItems = total_num_items;
      items = rt.arenaAllocN<i32>(*tmpArena, numItems);
    });

#ifdef BOT_GPU
  exec.warpForEachTask(
    rt, numWorlds,
    [&](i32 idx) {
      auto &store = fn(idx);
      Item *start = &items[worldOffsets[idx]];
      u32 count = worldCounts[idx];

      gpu_prims::warpLoop(count,
        [&](i32 item_idx) {
          start[item_idx].worldID = idx;
          start[item_idx].elem = store.get(item_idx);
        });
    });
#else
  exec.forEachTask(
    rt, numWorlds,
    [&](i32 idx) {
      auto &store = fn(idx);
      Item *start = &items[worldOffsets[idx]];
      u32 count = worldCounts[idx];

      for (u32 item_idx = 0; item_idx < count; ++item_idx) {
        start[item_idx].worldID = idx;
        start[item_idx].elem = store.get(item_idx);
      }
    });
#endif
}

template <typename ID>
void CompactedPersistentStore<ID>::init(
  Runtime &rt,
  u32 num_worlds,
  MemArena &persistent_arena,
  MemArena &tmp_arena)
{
  tmpArena = &tmp_arena;

  numItems = 0;
  items = nullptr;
  numWorlds = num_worlds;

  worldCounts = rt.arenaAllocN<i32>(persistent_arena, numWorlds);
  worldOffsets = rt.arenaAllocN<i32>(persistent_arena, numWorlds);
}

template <typename ID>
template <typename GetStoreT>
void CompactedPersistentStore<ID>::compact(
  Runtime &rt,
  TaskExec &exec,
  GetStoreT &&fn)
{
  using StoreT = decltype(fn(0));
  using RefT = typename StoreT::StoreRef;

  u32 total_num_items = exec.reduceTask<Reduction::Sum>(
    rt, numWorlds, true,
    [&](i32 idx) {
      auto &store = fn(idx);
      i32 num_items = store->size();
      store.worldCounts[idx] = num_items;
      return num_items;
    });

  TaskPrimitives::prefixSum(
      rt, exec, *tmpArena,
      worldCounts, worldOffsets,
      total_num_items);

  exec.serialTask(
    rt, [&]() {
      numItems = total_num_items;
      items = rt.arenaAllocN<i32>(*tmpArena, numItems);
    });

#ifdef BOT_GPU
  exec.warpForEachTask(
    rt, numWorlds,
    [&](i32 idx) {
      auto &store = fn(idx);
      Item *start = &items[worldOffsets[idx]];
      u32 count = worldCounts[idx];

      u32 total_offset = 0;

      gpu_prims::warpLoopSync(rt,
        [&](ID id, RefT ref) {
          __syncwarp();
          if (id != StoreT::StoreID::none()) {
            u32 offset = __popc(__activemask() & __lanemask_lt());

            start[total_offset + offset].worldID = idx;
            start[total_offset + offset].id = id;

            if (__lanemask_lt() == 0) {
              total_offset += __popc(__activemask());
            }
          }

          total_offset = __shfl_sync(total_offset, 0xFFFF'FFFF, 0);
        });
    });
#else
  exec.forEachTask(
    rt, numWorlds,
    [&](i32 idx) {
      auto &store = fn(idx);
      Item *start = &items[worldOffsets[idx]];

      u32 offset = 0;
      store.iterate(rt,
        [&](ID id, RefT) {
          start[offset].worldID = idx;
          start[offset].id = id;
          offset++;
        });
    });
#endif
}

}
