#include "rt_impl.hpp"
#include "rt.hpp"
#include "job.hpp"

#include "err.hpp"
#include "os.hpp"
#include "utils.hpp"

#include <cstdio>
#include <thread>

namespace bot {
  
namespace {

inline constexpr u32 VMEM_COMMIT_GROW_BLOCKS = 
    256 * 1024 * 1024 / GLOBAL_ALLOC_BLOCK_SIZE;

inline constexpr bool ENABLE_GLOBAL_DEALLOC_DECOMMIT = true;

}

struct JobQueue {
  static inline constexpr i32 MAX_NUM_QUEUED_JOBS = 32;

  alignas(BOT_CACHE_LINE)
    AtomicU32 head = 0;
  AtomicU32 tail = 0;

  i32 numTotalWorkers;
  u64 prngState;
  JobState jobs[MAX_NUM_QUEUED_JOBS] = {};
};

struct ThreadPool {
  std::thread *osThreads = nullptr;
  i32 numWorkers = 0;
};

RTStateHandle createRuntimeState(const RuntimeConfig &cfg)
{
  VirtualMemProperties vmem_props = virtualMemProperties();
  chk(VMEM_COMMIT_GROW_BLOCKS * GLOBAL_ALLOC_BLOCK_SIZE % vmem_props.pageSize == 0);

  u32 mem_pool_reserved_blocks;
  {
    OSMemStats os_mem_stats = osMemStats();

    u64 reserved_size_mb = cfg.memPoolReservedSizeMB;

    if (reserved_size_mb == 0) {
      reserved_size_mb = os_mem_stats.totalMem / (1024 * 1024);
    }

    u64 mem_pool_reserved_size = roundToAlignment(
      roundToAlignment(reserved_size_mb * 1024 * 1024, (u64)GLOBAL_ALLOC_BLOCK_SIZE),
      vmem_props.pageSize);

    mem_pool_reserved_blocks = u32(mem_pool_reserved_size / GLOBAL_ALLOC_BLOCK_SIZE);

    chk(mem_pool_reserved_blocks <= (u64)GlobalAlloc::MAX_NUM_BLOCKS);
  }

  u32 init_used_blocks = 
      (u32)divideRoundUp(sizeof(RuntimeState), (size_t)GLOBAL_ALLOC_BLOCK_SIZE);

  u32 mem_pool_committed_blocks;
  {
    u64 mem_pool_committed_size =
      roundToAlignment((u64)1024 * 1024 * (u64)cfg.memPoolCommittedSizeMB,
                       (u64)GLOBAL_ALLOC_BLOCK_SIZE);

    mem_pool_committed_size = std::max(mem_pool_committed_size,
        u64(init_used_blocks) * GLOBAL_ALLOC_BLOCK_SIZE);

    mem_pool_committed_size = roundToAlignment(
        mem_pool_committed_size, vmem_props.pageSize);

    mem_pool_committed_blocks = u32(mem_pool_committed_size / GLOBAL_ALLOC_BLOCK_SIZE);
    chk(mem_pool_committed_blocks > 0);
  }

  char *mem_pool = (char *)virtualMemReserveRegion(
      (u64)mem_pool_reserved_blocks * GLOBAL_ALLOC_BLOCK_SIZE);
  virtualMemCommitRegion(
      mem_pool, 0, (u64)mem_pool_committed_blocks * GLOBAL_ALLOC_BLOCK_SIZE);

  RuntimeState *state = new (mem_pool) RuntimeState {};

  state->globalAlloc.init(
      mem_pool, init_used_blocks, mem_pool_committed_blocks);

  state->memPoolReservedBlocks = mem_pool_reserved_blocks;
  state->memPoolCommittedBlocks = mem_pool_committed_blocks;
  state->memPoolMinCommittedBlocks = mem_pool_committed_blocks;

  return { mem_pool };
}

void destroyRuntimeState(RTStateHandle state_hdl)
{
  RuntimeState *state = getRuntimeState(state_hdl);

  void *mem_pool = (void *)state;
  u32 mem_pool_reserved_blocks = state->memPoolReservedBlocks;
  u32 mem_pool_committed_blocks = state->memPoolCommittedBlocks;

  state->~RuntimeState();
  virtualMemDecommitRegion(
      (void *)mem_pool, 0, (u64)mem_pool_committed_blocks * GLOBAL_ALLOC_BLOCK_SIZE);
  virtualMemReleaseRegion(
      (void *)mem_pool, (u64)mem_pool_reserved_blocks * GLOBAL_ALLOC_BLOCK_SIZE);
}

ThreadPool * createThreadPool(
    RTStateHandle state_hdl, MemArena &arena, i32 num_workers,
    void (*fn)(void *, i32), void *data)
{
  chk(num_workers > 0);

  ThreadPool *pool = (ThreadPool *)arenaAlloc(state_hdl, arena,
      sizeof(ThreadPool), alignof(ThreadPool));

  pool->osThreads = (std::thread *)arenaAlloc(state_hdl, arena,
      sizeof(std::thread) * num_workers, alignof(std::thread));
  pool->numWorkers = num_workers;

  for (i32 worker_idx = 0; worker_idx < pool->numWorkers; worker_idx++) {
    new (&pool->osThreads[worker_idx]) std::thread(fn, data, worker_idx);
  }

  return pool;
}

void destroyThreadPool(ThreadPool *pool)
{
  for (i32 worker_idx = 0; worker_idx < pool->numWorkers; worker_idx++) {
    pool->osThreads[worker_idx].join();
    pool->osThreads[worker_idx].~thread();
  }
}

JobQueue * createJobQueue(RTStateHandle state_hdl, MemArena &arena,
                          i32 num_workers)
{
  JobQueue *job_queues = (JobQueue *)arenaAlloc(
      state_hdl, arena, sizeof(JobQueue) * num_workers, alignof(JobQueue));

  for (i32 i = 0; i < num_workers; i++) {
    new (&job_queues[i]) JobQueue {
      .numTotalWorkers = num_workers,
      .prngState = u32Hash((u32)i + 1),
    };
  }

  return job_queues;
}

namespace {

// Chase-Lev Deque

inline bool isQueueEmpty(u32 head, u32 tail)
{
  return head - tail <= (1u << 31u);
}

enum class PopLocalResult : u32 {
  NoJob = 0,
  OneJob = 1,
  MultipleJobs = 2,
};

PopLocalResult popLocalJob(JobQueue &queue, JobState *job_out)
{
  u32 cur_tail = queue.tail.load_relaxed();
  cur_tail -= 1;
  queue.tail.store<sync::seq_cst>(cur_tail);

  u32 cur_head = queue.head.load<sync::seq_cst>();

  u32 job_idx = cur_tail % JobQueue::MAX_NUM_QUEUED_JOBS;

  PopLocalResult result;

  if (cur_head == cur_tail) {
    u32 next_tail = cur_tail + 1;
    if (queue.head.compare_exchange_strong<sync::relaxed, sync::relaxed>(
          cur_head, next_tail)) {
      *job_out = queue.jobs[job_idx];
      result = PopLocalResult::OneJob;
    } else {
      result = PopLocalResult::NoJob;
    }
    queue.tail.store_relaxed(next_tail);
  } else if (isQueueEmpty(cur_head, cur_tail)) {
    queue.tail.store_relaxed(cur_tail + 1);
    result = PopLocalResult::NoJob;
  } else {
    *job_out = queue.jobs[cur_tail];
    result = PopLocalResult::MultipleJobs;
  }

  return result;
}

// Sadly this isn't tsan safe because the read of the job array
// isn't actually fully synchronized. The compare exchange at the end
// guarantees that if a data race occurs the data will not be used, but
// tsan will correctly flag the read as a race even if the data is discarded.
BOT_TSAN_DISABLE bool stealJob(JobQueue &queue, JobState *job_out)
{
  u32 cur_head = queue.head.load_relaxed();
  u32 cur_tail = queue.tail.load<sync::seq_cst>();

  if (isQueueEmpty(cur_head, cur_tail)) {
    return false;
  }

  u32 job_idx = cur_head % JobQueue::MAX_NUM_QUEUED_JOBS;
  *job_out = queue.jobs[job_idx];

  return queue.head.compare_exchange_strong<sync::seq_cst, sync::relaxed>(
      cur_head, cur_head + 1);
}

u32 splitParallelizeJob(JobQueue *job_queue, i32 worker_idx,
                        JobFnPtr fn, void *data,
                        u32 num_invocations, u32 invocation_offset)
{
  u32 b_num_invocations = num_invocations / 2;
  u32 a_num_invocations = num_invocations - b_num_invocations;

  u32 b_offset = invocation_offset + a_num_invocations;

  queueJob(job_queue, worker_idx, fn, data,
           b_num_invocations, b_offset);

  return a_num_invocations;
}

inline i32 randomVictim(JobQueue &queue, i32 worker_idx)
{
  u64 x = queue.prngState;

  // xorwow
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  x *= 0x2545F4914F6CDD1DULL;
  queue.prngState = x;

  i32 random_idx = (i32)u32mulhi(x >> 32, queue.numTotalWorkers - 1);

  if (random_idx >= worker_idx) {
    random_idx += 1;
  }

  return random_idx;
}

}

void queueJob(JobQueue *job_queue, i32 worker_idx,
              JobFnPtr fn, void *data,
              u32 num_invocations, u32 invocation_offset)
{
  chk(num_invocations != 0);

  JobQueue &worker_queue = job_queue[worker_idx];

  u32 cur_tail = worker_queue.tail.load_relaxed();
  u32 cur_head = worker_queue.head.load_acquire();

  if (cur_tail - cur_head == JobQueue::MAX_NUM_QUEUED_JOBS) [[unlikely]] {
    FATAL("Job queue overflow for worker %d", worker_idx);
  }

  u32 job_idx = cur_tail % JobQueue::MAX_NUM_QUEUED_JOBS;

  worker_queue.jobs[job_idx] = JobState {
    .fn = fn,
    .data = data,
    .numInvocationsToRun = (u32)num_invocations,
    .invocationOffset = invocation_offset,
  };

  worker_queue.tail.store_release(cur_tail + 1);
}

bool getJob(JobQueue *job_queue, i32 worker_idx, JobState *job_out)
{
  JobQueue &worker_queue = job_queue[worker_idx];

  JobState run_job;
  PopLocalResult pop_result = popLocalJob(worker_queue, &run_job);
  if (pop_result == PopLocalResult::MultipleJobs) {
    *job_out = run_job;
    return true;
  } else if (pop_result == PopLocalResult::NoJob) {
    i32 victim_idx = randomVictim(worker_queue, worker_idx);

    if (!stealJob(job_queue[victim_idx], &run_job)) {
      return false;
    }
  }

  job_out->fn = run_job.fn;
  job_out->data = run_job.data;

  if (run_job.numInvocationsToRun == 1) {
    job_out->numInvocationsToRun = 1;
  } else {
    // At this point we know the local deque is empty so should definitely split
    job_out->numInvocationsToRun = splitParallelizeJob(
        job_queue, worker_idx, run_job.fn, run_job.data,
        run_job.numInvocationsToRun, run_job.invocationOffset);
  }

  job_out->invocationOffset = run_job.invocationOffset;

  return true;
}

u32 checkParallelizeJob(JobQueue *job_queue, i32 worker_idx,
                        JobFnPtr fn, void *data,
                        u32 num_invocations, u32 invocation_offset)
{
  JobQueue &worker_queue = job_queue[worker_idx];

  u32 cur_head = worker_queue.head.load_relaxed();
  u32 cur_tail = worker_queue.tail.load_relaxed();

  u32 num_jobs = cur_tail - cur_head;

  if (num_jobs > 0) {
    return num_invocations;
  }

  return splitParallelizeJob(job_queue, worker_idx, fn, data,
                             num_invocations, invocation_offset);
}

MemHandle globalAlloc(RTStateHandle state_hdl, u32 num_blocks)
{
  RuntimeState *state = getRuntimeState(state_hdl);

  spinLock(&state->allocLock);
  BOT_DEFER(spinUnlock(&state->allocLock));

  u32 blk = state->globalAlloc.alloc((char *)state, num_blocks);
  if (blk == GLOBAL_ALLOC_OOM) [[unlikely]] {
    u32 old_committed_blocks = state->memPoolCommittedBlocks;

    u32 num_new_blocks = roundToAlignment(
        num_blocks, VMEM_COMMIT_GROW_BLOCKS);

    u32 new_committed_blocks = old_committed_blocks + num_new_blocks;
    if (new_committed_blocks > state->memPoolReservedBlocks) [[unlikely]] {
      return { GLOBAL_ALLOC_OOM, 0 };
    }

    virtualMemCommitRegion((char *)state,
        (u64)old_committed_blocks * GLOBAL_ALLOC_BLOCK_SIZE,
        (u64)num_new_blocks * GLOBAL_ALLOC_BLOCK_SIZE);
    state->memPoolCommittedBlocks = new_committed_blocks;

    blk = old_committed_blocks;

    state->globalAlloc.addNewBlocks(
        (char *)state, blk, num_blocks, num_new_blocks - num_blocks);
  }

  return { blk, num_blocks };
}

void globalDealloc(RTStateHandle state_hdl, MemHandle mem)
{
  RuntimeState *state = getRuntimeState(state_hdl);

  if (mem.hdl == GLOBAL_ALLOC_OOM) [[unlikely]] {
    return;
  }

  spinLock(&state->allocLock);
  BOT_DEFER(spinUnlock(&state->allocLock));

  GlobalAlloc::DeallocStatus dealloc_status = 
      state->globalAlloc.dealloc((char *)state, mem.hdl, mem.numBlks);

  if constexpr (ENABLE_GLOBAL_DEALLOC_DECOMMIT) {
    if (state->memPoolMinCommittedBlocks != state->memPoolReservedBlocks) {
      u32 start_block = dealloc_status.freeRegionStart;
      u32 end_block = start_block + dealloc_status.numFreeBlocks;

      u32 eligible_start_block =
          std::max(start_block, state->memPoolMinCommittedBlocks);

      u32 eligible_free_blocks = end_block - eligible_start_block;

      if (end_block == state->memPoolCommittedBlocks &&
          eligible_free_blocks > 0 &&
          eligible_free_blocks % VMEM_COMMIT_GROW_BLOCKS == 0) {
        state->globalAlloc.removeFreeBlock((char *)state,
            start_block, eligible_start_block - start_block);

        virtualMemDecommitRegion(
            (char *)state, eligible_start_block * GLOBAL_ALLOC_BLOCK_SIZE,
            eligible_free_blocks * GLOBAL_ALLOC_BLOCK_SIZE);
        state->memPoolCommittedBlocks = eligible_start_block;
      }
    }
  }
}

}
