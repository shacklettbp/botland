#include "utils.hpp"
#include "sync.hpp"
#include "log.hpp"

#include <cstddef>
#include <cstdio>
#include <algorithm>

#ifdef BOT_GPU
#include "rt_gpu.hpp"
#include "gpu_prims.hpp"
#endif

namespace bot {

RuntimeState * getRuntimeState(RTStateHandle state_hdl)
{
  return (RuntimeState *)state_hdl.base;
}

char * memHandleToPtr(RTStateHandle state_hdl, MemHandle mem)
{
  if constexpr (GLOBAL_ALLOC_CHECK_OOM_PTR) {
    if (mem.hdl == 0) [[unlikely]] {
      return nullptr;
    }
  }

  char *base_ptr = (char *)state_hdl.base;
  return base_ptr + (u64)GLOBAL_ALLOC_BLOCK_SIZE * (u64)mem.hdl;
}

void * arenaAlloc(RTStateHandle state_hdl,
                  MemArena &arena,
                  u64 num_bytes,
                  u64 alignment)
{
  if (alignment > ARENA_NATIVE_ALIGNMENT) [[unlikely]] {
    num_bytes += alignment - 1;
  }

  num_bytes = roundToAlignment(num_bytes, ARENA_NATIVE_ALIGNMENT);

  AtomicU64Ref atomic_offset(arena.offset);

  u64 alloc_offset = atomic_offset.fetch_add_relaxed(num_bytes);

  u64 end_offset = alloc_offset + num_bytes;

  if (alignment > ARENA_NATIVE_ALIGNMENT) [[unlikely]] {
    alloc_offset = roundToAlignment(alloc_offset, alignment);
  }

  u64 chunk_size = (u64)arena.mem.numBlks * GLOBAL_ALLOC_BLOCK_SIZE;

  if (end_offset <= chunk_size) [[likely]] {
    char *handle_ptr = memHandleToPtr(state_hdl, arena.mem);
    return handle_ptr + alloc_offset;
  }

  void * memArenaGrow(RTStateHandle, MemArena &, u64, u64);
  return memArenaGrow(state_hdl, arena, num_bytes, alignment);
}

void arenaPreAllocate(RTStateHandle state_hdl,
                      MemArena &arena,
                      u32 num_blks)
{
  void * memArenaGrow(RTStateHandle state_hdl, MemArena &, u64, u64);
  memArenaGrow(state_hdl, arena,
      (u64)GLOBAL_ALLOC_BLOCK_SIZE * (u64)num_blks - sizeof(MemHandle),
      alignof(MemHandle));
  arena.offset = sizeof(MemHandle);
}

ArenaRegion arenaBeginRegion(RTStateHandle state_hdl, MemArena &arena)
{
  return ArenaRegion {
    memHandleToPtr(state_hdl, arena.mem) + arena.offset,
  };
}

u32 arenaEndRegion(RTStateHandle state_hdl, MemArena &arena, ArenaRegion region)
{
  char *region_begin_ptr = (char *)region.ptr;

  char *blk_ptr = memHandleToPtr(state_hdl, arena.mem);
  u64 offset = region_begin_ptr - blk_ptr;

  if (offset <= (u64)arena.mem.hdl * (u64)GLOBAL_ALLOC_BLOCK_SIZE) {
    arena.offset = offset;
    return 0;
  }

  u32 memArenaShrink(RTStateHandle state_hdl, MemArena &, char *);
  return memArenaShrink(state_hdl, arena, region_begin_ptr);
}

u32 arenaRelease(RTStateHandle state_hdl, MemArena &arena)
{
  u32 memArenaShrink(RTStateHandle state_hdl, MemArena &, char *);
  return memArenaShrink(state_hdl, arena, nullptr);
}

constexpr TaskKernelConfig TaskKernelConfig::persistentDefault()
{
  return {};
}

constexpr TaskKernelConfig TaskKernelConfig::singleThread()
{
  return {
    .numGPUThreadsPerBlock = 1,
    .numGPUBlocksPerSM = 1,
  };
}

Runtime::Runtime() {}

Runtime::Runtime(BOT_RT_INIT_PARAMS)
{
#ifdef BOT_GPU
  state_hdl_ = { (char *)__cvta_generic_to_global(gpuConsts().gpuBuffer) };
  thread_id_ = threadIdx.x + blockDim.x * blockIdx.x;
#else
  state_hdl_ = _bot_rt_state_hdl;
  thread_id_ = _bot_rt_thread_id;
#endif
}

Runtime::~Runtime()
{
  releaseArena(result_arena_);
  releaseArena(tmp_arena_);
}

RTStateHandle Runtime::stateHandle()
{
  return state_hdl_;
}

RuntimeState * Runtime::state()
{
  return getRuntimeState(state_hdl_);
}

i32 Runtime::threadID() 
{
  return thread_id_;
}

MemArena & Runtime::tmpArena()
{
  return tmp_arena_;
}

MemArena & Runtime::resultArena()
{
  return result_arena_;
}

void * Runtime::tmpAlloc(u64 num_bytes, u64 alignment)
{
  return ::bot::arenaAlloc(state_hdl_, tmp_arena_, num_bytes, alignment);
}

template <typename T>
T * Runtime::tmpAlloc()
{
  return (T *)::bot::arenaAlloc(state_hdl_, tmp_arena_, sizeof(T), alignof(T));
}

template <typename T>
T * Runtime::tmpAllocN(i64 n)
{
  return (T *)::bot::arenaAlloc(state_hdl_, tmp_arena_, sizeof(T) * n, alignof(T));
}

ArenaRegion Runtime::beginTmpRegion()
{
  return ::bot::arenaBeginRegion(state_hdl_, tmp_arena_);
}

void Runtime::endTmpRegion(ArenaRegion region)
{
  ::bot::arenaEndRegion(state_hdl_, tmp_arena_, region);
}

void * Runtime::resultAlloc(u64 num_bytes, u64 alignment)
{
  return ::bot::arenaAlloc(state_hdl_, result_arena_, num_bytes, alignment);
}

template <typename T>
T * Runtime::resultAlloc()
{
  return (T *)::bot::arenaAlloc(state_hdl_, result_arena_, sizeof(T), alignof(T));
}

template <typename T>
T * Runtime::resultAllocN(i64 n)
{
  return (T *)::bot::arenaAlloc(state_hdl_, result_arena_, sizeof(T) * n, alignof(T));
}

ArenaRegion Runtime::beginResultRegion()
{
  return ::bot::arenaBeginRegion(state_hdl_, result_arena_);
}

void Runtime::endResultRegion(ArenaRegion region)
{
  ::bot::arenaEndRegion(state_hdl_, result_arena_, region);
}

ArenaRegion Runtime::beginArenaRegion(MemArena &arena)
{
  return ::bot::arenaBeginRegion(state_hdl_, arena);
}

void Runtime::endArenaRegion(MemArena &arena, ArenaRegion region)
{
  ::bot::arenaEndRegion(state_hdl_, arena, region);
}

void Runtime::releaseArena(MemArena &arena)
{
  ::bot::arenaRelease(state_hdl_, arena);
}

void * Runtime::arenaAlloc(MemArena &arena, u64 num_bytes, u64 alignment)
{
  return ::bot::arenaAlloc(state_hdl_, arena, num_bytes, alignment);
}

template <typename T>
T * Runtime::arenaAlloc(MemArena &arena)
{
  return (T *)::bot::arenaAlloc(state_hdl_, arena, sizeof(T), alignof(T));
}

template <typename T>
T * Runtime::arenaAllocN(MemArena &arena, i64 n)
{
  return (T *)::bot::arenaAlloc(state_hdl_, arena, sizeof(T) * n, alignof(T));
}

bool Runtime::isLeaderLane()
{
  return thread_id_ % 32 == 0;
}

void Runtime::syncLanes()
{
#ifdef BOT_GPU
  __syncwarp(0xFFFF'FFFF);
#endif
}

template <typename Fn, typename... Args>
auto TaskExec::serialTask(Runtime &rt, Fn &&fn, bool dbg, Args &&...args)
{
  using R = decltype(fn(std::forward<Args>(args)...));
  constexpr bool returns = !std::is_void_v<R>;

  if constexpr (returns) {
    static_assert(sizeof(R) <= TaskState::NUM_INLINE_DATA_BYTES &&
                  alignof(R) <= TaskState::INLINE_DATA_ALIGNMENT);
  }

  TaskState *start_task = curTask;

#ifdef BOT_GPU
  bool begin_task_res = gpu_prims::leaderExec(
      [&]() { return beginTask(dbg); });

  if (!begin_task_res) {
    if constexpr (returns) {
      return *(R *)start_task->result;
    } else {
      return;
    }
  }
#else
  if (!beginTask(dbg)) {
    if constexpr (returns) {
      return *(R *)start_task->result;
    } else {
      return;
    }
  }
#endif

  if (rt.isLeaderLane()) {
    AtomicI32Ref start_offset_atomic(curTask->startedOffset);

    i32 prev = start_offset_atomic.exchange<sync::relaxed>(1);

    if (prev == 0) {
      if constexpr (returns) {
        if constexpr (std::is_pointer_v<R>) {
          start_task->result = fn(std::forward<Args>(args)...);
        } else {
          start_task->result = new (rt.arenaAlloc<R>(mgr->arena)) R(
            fn(std::forward<Args>(args)...));
        }
      } else {
        fn(std::forward<Args>(args)...);
      }

      completeTask(rt);
    } else {
      waitForTaskCompletion();
    }
  }

  rt.syncLanes();

#ifdef BOT_GPU
  curTask = (TaskState *)__shfl_sync(0xFFFF'FFFF, (uintptr_t)curTask, 0);
#endif

  if constexpr (returns) {
    if constexpr (std::is_pointer_v<R>) {
      return (R)start_task->result;
    } else {
      return *(R *)start_task->result;
    }
  }
}

#ifdef BOT_GPU
template <typename Fn, typename ...Args>
void TaskExec::warpForEachTask(Runtime &rt, i32 num_invocations,
                               Fn &&fn, Args &&...args)
{
  auto leader_exec = [&](auto fn) {
    decltype(fn()) res;
    if (gpu_prims::isLeader()) {
      res = fn();
    }
    res = __shfl_sync(0xFFFF'FFFF, res, 0);
    __syncwarp();
    return res;
  };

  TaskState *start_task = curTask;

  bool begin_task_res = leader_exec([&]() 
      { return beginTask(); });

  if (!begin_task_res) {
    return;
  }

  AtomicI32Ref start_offset_atomic(start_task->startedOffset);
  AtomicI32Ref finished_offset_atomic(start_task->finishedOffset);

  while (true) {
    i32 idx = leader_exec([&]() 
        { return start_offset_atomic.fetch_add_relaxed(1); });

    if (idx < num_invocations) {
      fn(idx, std::forward<Args>(args)...);

      i32 prev_num_finished = leader_exec([&]() 
          { return finished_offset_atomic.fetch_add_acq_rel(1); });

      if (prev_num_finished == num_invocations - 1) {
        leader_exec([&]() { 
          completeTask(rt);
          return 0; 
        });
        break;
      }
    } else {
      leader_exec([&]() { 
        waitForTaskCompletion(); 
        return 0;
      });
      break;
    }
  }

  curTask = (TaskState *)__shfl_sync(0xFFFF'FFFF, (uintptr_t)curTask, 0);
}

template <typename Fn, typename ...Args>
void TaskExec::blockForEachTask(Runtime &rt, i32 num_invocations,
                                Fn &&fn, Args &&...args)
{
  auto leader_exec = [&](auto fn) {
    using RetType = decltype(fn());

    RetType *res = (RetType *)gpuSMem();
    if (threadIdx.x == 0) {
      *res = fn();
    }
    __syncthreads(); // TODO: Do I need a thread fence or something?
    return *res;
  };

  TaskState *start_task = curTask;

  bool begin_task_res = leader_exec([&]() 
      { return beginTask(); });

  if (!begin_task_res) {
    return;
  }

  AtomicI32Ref start_offset_atomic(start_task->startedOffset);
  AtomicI32Ref finished_offset_atomic(start_task->finishedOffset);

  while (true) {
    i32 idx = leader_exec([&]() 
        { return start_offset_atomic.fetch_add_relaxed(1); });

    if (idx < num_invocations) {
      fn(idx, std::forward<Args>(args)...);

      i32 prev_num_finished = leader_exec([&]() 
          { return finished_offset_atomic.fetch_add_acq_rel(1); });

      if (prev_num_finished == num_invocations - 1) {
        leader_exec([&]() { 
          completeTask(rt);
          return 0; 
        });
        break;
      }
    } else {
      leader_exec([&]() { 
        waitForTaskCompletion(); 
        return 0;
      });
      break;
    }
  }

  TaskState **cur_task = (TaskState **)gpuSMem();
  if (threadIdx.x == 0) {
    *cur_task = curTask;
  }
  __syncthreads();
  curTask = *cur_task;
}
#endif

template <typename Fn, typename... Args>
void TaskExec::forEachTask(Runtime &rt, i32 num_invocations,
                           bool leader_only, Fn &&fn, Args &&...args)
{
  TaskState *start_task = curTask;

#ifdef BOT_GPU
  bool begin_task_res = false;
  if (rt.isLeaderLane()) {
    begin_task_res = beginTask();
  }
  begin_task_res = __shfl_sync(0xFFFF'FFFF, begin_task_res, 0);

  if (!begin_task_res) {
    return;
  }

  AtomicI32Ref start_offset_atomic(start_task->startedOffset);
  AtomicI32Ref finished_offset_atomic(start_task->finishedOffset);

  if (leader_only) {
    if (rt.isLeaderLane()) {
      while (true) {
        i32 idx = start_offset_atomic.fetch_add_relaxed(1);

        if (idx < num_invocations) {
          fn(idx, std::forward<Args>(args)...);

          i32 prev_num_finished = finished_offset_atomic.fetch_add_acq_rel(1);
          if (prev_num_finished == num_invocations - 1) {
            completeTask(rt);
            break;
          }
        } else {
          waitForTaskCompletion();
          break;
        }
      }
    }
  } else {
    while (true) {
      i32 idx;
      if (rt.isLeaderLane()) {
        idx = start_offset_atomic.fetch_add_relaxed(32);
      }

      idx = __shfl_sync(0xFFFF'FFFF, idx, 0);
      idx += rt.threadID() % 32;

      if (idx < num_invocations) {
        fn(idx, std::forward<Args>(args)...);
      }

      rt.syncLanes();

      bool should_break = false;
      if (rt.isLeaderLane()) {
        i32 num_in_bounds = std::max(0, std::min(num_invocations - idx, 32));

        if (num_in_bounds == 0) {
          waitForTaskCompletion();
          should_break = true;
        } else {
          i32 prev_num_finished = finished_offset_atomic.fetch_add_acq_rel(num_in_bounds);
          if (prev_num_finished == num_invocations - num_in_bounds) {
            completeTask(rt);
            should_break = true;
          } 
        }
      }

      should_break = __shfl_sync(0xFFFF'FFFF, should_break, 0);

      if (should_break) {
        break;
      }
    }
  }

  curTask = (TaskState *)__shfl_sync(0xFFFF'FFFF, (uintptr_t)curTask, 0);
#else
  if (!beginTask()) {
    return;
  }

  AtomicI32Ref start_offset_atomic(start_task->startedOffset);
  AtomicI32Ref finished_offset_atomic(start_task->finishedOffset);

  (void)leader_only;
  while (true) {
    i32 idx = start_offset_atomic.fetch_add_relaxed(1);

    if (idx < num_invocations) {
      fn(idx, std::forward<Args>(args)...);

      i32 prev_num_finished = finished_offset_atomic.fetch_add_acq_rel(1);
      if (prev_num_finished == num_invocations - 1) {
        completeTask(rt);
        break;
      }
    } else {
      waitForTaskCompletion();
      break;
    }
  }
#endif
}

template <Reduction reduction, typename Fn, typename... Args>
auto TaskExec::reduceTask(Runtime &rt, i32 num_invocations,
                          bool leader_only, Fn &&fn, Args &&...args)
{
  using R = decltype(fn(0, std::forward<Args>(args)...));
  static_assert(std::is_integral_v<R>);
  static_assert(reduction == Reduction::Sum);

  R *global_result = serialTask(rt,
    [](Runtime &rt, TaskManager *mgr) {
      R init;
      if constexpr (reduction == Reduction::Sum) {
        init = 0;
      }

      return new (rt.arenaAlloc<R>(mgr->arena)) R(init);
    }, rt, mgr);

  AtomicRef<R> global_result_atomic(*global_result);

  TaskState *start_task = curTask;

  if (!beginTask()) {
    return global_result_atomic.load_relaxed();
  }

  AtomicI32Ref start_offset_atomic(start_task->startedOffset);
  AtomicI32Ref finished_offset_atomic(start_task->finishedOffset);

  R local_result;
  if constexpr (reduction == Reduction::Sum) {
    local_result = 0;
  }

  auto reduceLocalToGlobal =
    [&]
  ()
  {
    if constexpr (reduction == Reduction::Sum) {
      if (local_result != 0) {
        global_result_atomic.fetch_add_relaxed(local_result);
      }
    }
  };

#ifdef BOT_GPU
  if (leader_only) {
    if (rt.isLeaderLane()) {
      while (true) {
        i32 idx = start_offset_atomic.fetch_add_relaxed(1);

        if (idx < num_invocations) {
          R r = fn(idx, std::forward<Args>(args)...);

          if constexpr (reduction == Reduction::Sum) {
            local_result += r;
          } else {
            static_assert(false);
          }

          i32 prev_num_finished = finished_offset_atomic.fetch_add_acq_rel(1);
          if (prev_num_finished == num_invocations - 1) {
            reduceLocalToGlobal();
            completeTask(rt);
            break;
          }
        } else {
          reduceLocalToGlobal();
          waitForTaskCompletion();
          break;
        }
      }
    }
  } else {
    while (true) {
      i32 idx;
      if (rt.isLeaderLane()) {
        idx = start_offset_atomic.fetch_add_relaxed(32);
      }

      idx = __shfl_sync(0xFFFF'FFFF, idx, 0);
      idx += rt.threadID() % 32;

      R r;
      if constexpr (reduction == Reduction::Sum) {
        r = 0;
      } else {
        static_assert(false);
      }

      if (idx < num_invocations) {
        r = fn(idx, std::forward<Args>(args)...);
      }

      rt.syncLanes();

      if constexpr (reduction == Reduction::Sum) {
        r = __reduce_add_sync(0xFFFF'FFFF, r);
        local_result += r;
      } else {
        static_assert(false);
      }

      bool should_break = false;
      if (rt.isLeaderLane()) {
        i32 num_in_bounds = std::max(0, std::min(num_invocations - idx, 32));

        if (num_in_bounds == 0) {
          reduceLocalToGlobal();
          waitForTaskCompletion();
          should_break = true;
        } else {
          i32 prev_num_finished = finished_offset_atomic.fetch_add_acq_rel(num_in_bounds);
          if (prev_num_finished == num_invocations - num_in_bounds) {
            reduceLocalToGlobal();
            completeTask(rt);
            should_break = true;
          } 
        }
      }

      should_break = __shfl_sync(0xFFFF'FFFF, should_break, 0);

      if (should_break) {
        break;
      }
    }
  }

  curTask = (TaskState *)__shfl_sync(0xFFFF'FFFF, (uintptr_t)curTask, 0);

#else
  (void)leader_only;

  while (true) {
    i32 idx = start_offset_atomic.fetch_add_relaxed(1);

    if (idx < num_invocations) {
      R r = fn(idx, std::forward<Args>(args)...);

      if constexpr (reduction == Reduction::Sum) {
        local_result += r;
      } else {
        static_assert(false);
      }

      i32 prev_num_finished = finished_offset_atomic.fetch_add_acq_rel(1);
      if (prev_num_finished == num_invocations - 1) {
        reduceLocalToGlobal();
        completeTask(rt);
        break;
      }
    } else {
      reduceLocalToGlobal();
      waitForTaskCompletion();
      break;
    }
  }
#endif

  return global_result_atomic.load_relaxed();
}



}
