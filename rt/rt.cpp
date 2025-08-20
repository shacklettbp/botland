#include "rt.hpp"

#include "rt_impl.hpp"

#include <cstdio>

#include "log.hpp"

namespace bot {

void * memArenaGrow(RTStateHandle state_hdl,
                    MemArena &arena,
                    u64 num_bytes,
                    u64 alignment)
{
  MemHandle old_mem = arena.mem;

  constexpr u64 chunk_data_offset = 
    roundToAlignment(sizeof(MemHandle), ARENA_NATIVE_ALIGNMENT);

  u64 min_alloc_bytes = chunk_data_offset + num_bytes;
  u64 default_alloc_size = (u64)arena.mem.numBlks * (u64)GLOBAL_ALLOC_BLOCK_SIZE;

  u64 num_alloc_bytes = default_alloc_size;
  u64 num_alloc_blks = arena.mem.numBlks;
  if (min_alloc_bytes > num_alloc_bytes) {
    num_alloc_bytes = roundToAlignment(
        min_alloc_bytes, (u64)GLOBAL_ALLOC_BLOCK_SIZE);
    num_alloc_blks = num_alloc_bytes / GLOBAL_ALLOC_BLOCK_SIZE;
  } 

  MemHandle mem = globalAlloc(state_hdl, num_alloc_blks);
  if (mem.hdl == GLOBAL_ALLOC_OOM) [[unlikely]] {
    return nullptr;
  }

  char *blk_base = (char *)memHandleToPtr(state_hdl, mem);
  MemHandle *chunk_prev_mem = (MemHandle *)blk_base;
  *chunk_prev_mem = old_mem;

  arena.offset = min_alloc_bytes;
  arena.mem = mem;

  if (alignment > ARENA_NATIVE_ALIGNMENT) {
    return blk_base + roundToAlignment(chunk_data_offset, alignment);
  } else {
    return blk_base + chunk_data_offset;
  }
}

u32 memArenaShrink(RTStateHandle state_hdl, MemArena &arena, char *shrink_to)
{
  u32 num_blks_released = 0;
  MemHandle mem = arena.mem;
  while (mem.hdl != 0) {
    char *blk_ptr = (char *)memHandleToPtr(state_hdl, mem);

    u64 offset = shrink_to - blk_ptr;

    if (offset <= (u64)mem.numBlks * (u64)GLOBAL_ALLOC_BLOCK_SIZE) {
      arena.offset = offset;
      arena.mem = mem;
      return num_blks_released;
    }

    MemHandle *chunk_prev_mem = (MemHandle *)blk_ptr;
    MemHandle prev_mem = *chunk_prev_mem;

    globalDealloc(state_hdl, mem);

    num_blks_released += mem.numBlks;

    mem = prev_mem;
  }

  arena = {};

  return num_blks_released;
}

void TaskManager::reset(Runtime &rt)
{
  rt.endArenaRegion(arena, tasksRegion);
  head = nullptr;
  numFinished = 0;
}

TaskExec TaskManager::start(Runtime &rt)
{
#if defined(BOT_GPU) && 0
  TaskExec exec;
  exec.mgr = this;

  if (rt.isLeaderLane()) {
    AtomicRef<TaskState *> task_ptr_atomic(head);

    TaskState *sentinel_ptr = (TaskState *)(0x1);
    TaskState * prev = nullptr;

    bool success = task_ptr_atomic.compare_exchange_strong<
      sync::relaxed, sync::relaxed>(prev, sentinel_ptr);

    if (success) {
      tasksRegion = rt.beginArenaRegion(arena);
      exec.curTask = rt.arenaAlloc<TaskState>(arena);
      new (exec.curTask) TaskState {};

      task_ptr_atomic.store_release(exec.curTask);
    } else {
      TaskState *cur_task;
      while ((cur_task = task_ptr_atomic.load_relaxed()) == sentinel_ptr) {}
      atomic_thread_fence(sync::acquire);

      exec.curTask = cur_task;
    }
  }

  exec.curTask = (TaskState *)__shfl_sync(
      0xFFFF'FFFF, (uintptr_t)exec.curTask, 0);

  return exec;
#else
  TaskExec exec;
  exec.mgr = this;

  AtomicRef<TaskState *> task_ptr_atomic(head);

  // This is really hacky, the address of the TaskManager is treated
  // as a sentinel value that all threads share, allowing task_ptr_atomic
  // to be treated as a lock initially.
  TaskState * sentinel_ptr = (TaskState *)(0x1);
  TaskState * prev = nullptr;

  bool success = task_ptr_atomic.compare_exchange_strong<
    sync::relaxed, sync::relaxed>(prev, sentinel_ptr);

  if (success) {
    tasksRegion = rt.beginArenaRegion(arena);
    exec.curTask = rt.arenaAlloc<TaskState>(arena);
    new (exec.curTask) TaskState {};

    task_ptr_atomic.store_release(exec.curTask);
  } else {
    TaskState *cur_task;
    while ((cur_task = task_ptr_atomic.load_relaxed()) == sentinel_ptr) {}
    atomic_thread_fence(sync::acquire);

    exec.curTask = cur_task;
  }

  return exec;
#endif
}

void TaskExec::finish(Runtime &rt)
{
#ifdef BOT_GPU
  i32 num_blocks = gridDim.x;
  __syncthreads();
  if (threadIdx.x == 0) {
    AtomicI32Ref num_finished_atomic(mgr->numFinished);
    i32 prev_num_finished = num_finished_atomic.fetch_add_acq_rel(1);

    if (prev_num_finished + 1 == num_blocks) {
      num_finished_atomic.store_relaxed(0);
      mgr->reset(rt);
    }
  }
#else
  mgr->reset(rt);
#endif
}

bool TaskExec::beginTask(bool dbg)
{
  AtomicRef<TaskState *> next_task_ptr_atomic(curTask->next);

  TaskState *next_task = next_task_ptr_atomic.load_relaxed();

  if (next_task == nullptr) {
    return true;
  } else {
    atomic_thread_fence(sync::acquire);
    curTask = next_task;

    return false;
  }
}

void TaskExec::completeTask(Runtime &rt, bool dbg)
{
  AtomicRef<TaskState *> next_task_ptr_atomic(curTask->next);
  TaskState *next_task = rt.arenaAlloc<TaskState>(mgr->arena);
  new (next_task) TaskState {};
  next_task_ptr_atomic.store_release(next_task);

  curTask = next_task;
}

void TaskExec::waitForTaskCompletion(bool dbg)
{
  AtomicRef<TaskState *> next_task_ptr_atomic(curTask->next);
  TaskState *next_task = nullptr;
  while (next_task == nullptr) {
    next_task = next_task_ptr_atomic.load_relaxed();
  }
  atomic_thread_fence(sync::acquire);

  curTask = next_task;
}

}
