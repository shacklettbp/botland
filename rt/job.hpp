#pragma once

#include "rt.hpp"

#ifdef BOT_GPU
#error "job.hpp not supported on GPU"
#endif

namespace bot {

struct JobQueue;

using JobFnPtr =
  void (*)(RuntimeState *, void *, JobQueue *, i32, i32, void *, i32, i32);

struct JobState {
  JobFnPtr fn;
  void *data;
  u32 numInvocationsToRun;
  u32 invocationOffset;
};

ThreadPool * createThreadPool(
    RuntimeState *state, MemArena &arena, i32 num_workers,
    void (*fn)(void *, i32), void *data);
void destroyThreadPool(ThreadPool *pool);

JobQueue * createJobQueue(RuntimeState *rt_state,
                          MemArena &arena, i32 num_workers);

void queueJob(JobQueue *job_queue, i32 worker_id,
              JobFnPtr fn, void *data, 
              u32 num_invocations, u32 invocation_offset = 0);

bool getJob(JobQueue *job_queue, i32 worker_id, JobState *job_out);

u32 checkParallelizeJob(JobQueue *job_queue, i32 worker_idx,
                        JobFnPtr fn, void *data, 
                        u32 num_invocations, u32 invocation_offset);

}
