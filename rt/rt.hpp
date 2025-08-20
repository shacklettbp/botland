#pragma once

#include "fwd.hpp"
#include "types.hpp"
#include "utils.hpp"
#include "err.hpp"
#include "sync.hpp"

#ifdef BOT_GPU
#include "rt_gpu.hpp"
#endif

namespace bot {

constexpr inline i32 GLOBAL_ALLOC_BLOCK_SIZE = 64 * 1024;
constexpr inline bool GLOBAL_ALLOC_CHECK_OOM_PTR = true;
constexpr inline u32 MAX_SCOPE_MEM_CHUNKS = 16;
constexpr inline u64 ARENA_NATIVE_ALIGNMENT = 16;

class Runtime;

struct RTStateHandle {
  void *base = nullptr;
};

struct RuntimeConfig {
  u32 memPoolCommittedSizeMB = 0;
  u32 memPoolReservedSizeMB = 0;
};

struct ThreadPoolConfig {
  enum : u32 {
    None = 0,
    PinThreads = 1 << 0,
  };

  i32 numOSThreads;
  u32 flags = PinThreads;
};

struct MemHandle {
  u32 hdl = 0;
  u32 numBlks = 0;
};

struct MemArena {
  u64 offset = 0;
  MemHandle mem = {};
};

struct ArenaRegion {
  void *ptr;
};

struct TaskKernelConfig {
  u32 numGPUThreadsPerBlock = 256;
  u32 numGPUBlocksPerSM = 3;

  static inline constexpr TaskKernelConfig persistentDefault();
  static inline constexpr TaskKernelConfig singleThread();
};

#ifdef BOT_GPU
#define BOT_RT_INIT_PARAMS void *
#define BOT_RT_INIT_ARGS nullptr
#define BOT_KERNEL(func_name, cfg, ...) \
extern "C" __global__ void \
__launch_bounds__(cfg.numGPUThreadsPerBlock, \
                  cfg.numGPUBlocksPerSM) \
func_name(__VA_ARGS__)

#else
#define BOT_RT_INIT_PARAMS \
  [[maybe_unused]] RTStateHandle _bot_rt_state_hdl, \
  [[maybe_unused]] i32 _bot_rt_thread_id
#define BOT_RT_INIT_ARGS _bot_rt_state_hdl, _bot_rt_thread_id
#define BOT_KERNEL(func_name, cfg, ...) \
extern "C" void func_name(BOT_RT_INIT_PARAMS, __VA_ARGS__)
#endif

#define BOT_TASK_KERNEL(func_name, ...) BOT_KERNEL( \
  func_name, TaskKernelConfig::persistentDefault(), __VA_ARGS__)

class Runtime {
public:
  inline Runtime();
  inline Runtime(BOT_RT_INIT_PARAMS);
  inline ~Runtime();

  inline RTStateHandle stateHandle();
  inline RuntimeState * state();

  inline i32 threadID(); // Globally unique thread ID

  inline MemArena & tmpArena();
  inline MemArena & resultArena();
     
  inline void * tmpAlloc(u64 num_bytes, u64 alignment = 8);
  template <typename T>
  T * tmpAlloc();
  template <typename T>
  T * tmpAllocN(i64 n);

  inline ArenaRegion beginTmpRegion();
  inline void endTmpRegion(ArenaRegion region);

  inline void * resultAlloc(u64 num_bytes, u64 alignment = 8);
  template <typename T>
  T * resultAlloc();
  template <typename T>
  T * resultAllocN(i64 n);

  inline ArenaRegion beginResultRegion();
  inline void endResultRegion(ArenaRegion region);

  inline ArenaRegion beginArenaRegion(MemArena &arena);
  inline void endArenaRegion(MemArena &arena, ArenaRegion region);
  inline void releaseArena(MemArena &arena);

  inline void * arenaAlloc(MemArena &arena, u64 num_bytes, u64 alignment = 8);
  template <typename T>
  inline T * arenaAlloc(MemArena &arena);
  template <typename T>
  inline T * arenaAllocN(MemArena &arena, i64 n);

  inline bool isLeaderLane();
  inline void syncLanes();

protected:
  RTStateHandle state_hdl_ = {};
  i32 thread_id_ = -1;
  MemArena tmp_arena_ = {};
  MemArena result_arena_ = {};
};

enum class Reduction : u32 {
  Sum = 0,
};

struct TaskState {
  static inline constexpr i32 NUM_INLINE_DATA_BYTES = 32;
  static inline constexpr i32 INLINE_DATA_ALIGNMENT = 16;

  i32 startedOffset = 0;
  i32 finishedOffset = 0;
  void *result = nullptr;

  TaskState *next = nullptr;
};

struct TaskManager;

struct TaskExec {
  TaskManager *mgr = nullptr;
  TaskState *curTask = nullptr;

  template <typename Fn, typename... Args>
  auto serialTask(Runtime &rt, Fn &&fn, bool dbg = false, Args &&...args);

  template <typename Fn, typename... Args>
  void forEachTask(Runtime &rt, i32 num_invocations, bool leader_only,
                   Fn &&fn, Args &&...args);

#ifdef BOT_GPU
  // Each warp gets one piece of work
  template <typename Fn, typename ...Args>
  void warpForEachTask(Runtime &rt, i32 num_invocations,
                       Fn &&fn, Args &&...args);

  template <typename Fn, typename ...Args>
  void blockForEachTask(Runtime &rt, i32 num_invocations,
                        Fn &&fn, Args &&...args);
#endif

  template <Reduction reduction, typename Fn, typename... Args>
  auto reduceTask(Runtime &rt, i32 num_invocations, bool leader_only,
                  Fn &&fn, Args &&...args);

  void finish(Runtime &rt);

private:
  bool beginTask(bool dbg = false);
  void completeTask(Runtime &rt, bool dbg = false);
  void waitForTaskCompletion(bool dbg = false);
};

struct TaskManager {
  MemArena arena = {};
  ArenaRegion tasksRegion = {};
  TaskState *head = nullptr;
  i32 numFinished = 0;

  TaskExec start(Runtime &rt);

private:
  void reset(Runtime &rt);

friend struct TaskExec;
};

RTStateHandle createRuntimeState(const RuntimeConfig &cfg);
void destroyRuntimeState(RTStateHandle state_hdl);
inline RuntimeState * getRuntimeState(RTStateHandle state_hdl);

MemHandle globalAlloc(RTStateHandle state_hdl, u32 num_blocks);
void globalDealloc(RTStateHandle state_hdl, MemHandle chunk);

inline char * memHandleToPtr(RTStateHandle state_hdl, MemHandle mem);

inline void * arenaAlloc(RTStateHandle state_hdl,
                         MemArena &arena,
                         u64 num_bytes,
                         u64 alignment);

inline void arenaPreAllocate(RTStateHandle state_hdl,
                             MemArena &arena,
                             u32 num_blks = 1);

inline ArenaRegion arenaBeginRegion(RTStateHandle state_hdl, MemArena &arena);
inline u32 arenaEndRegion(RTStateHandle state_hdl, MemArena &arena,
                          ArenaRegion region);
inline u32 arenaRelease(RTStateHandle state_hdl, MemArena &arena);

void initGlobalFallbackAllocator(RuntimeState *state);

}

#include "rt.inl"
