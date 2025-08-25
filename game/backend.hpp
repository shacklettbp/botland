#pragma once

#include "rt/rt.hpp"
#include "rt/cuda_cfg.hpp"

#include "sim.hpp"

#ifdef BOT_CUDA_SUPPORT
#include "rt/cuda_utils.hpp"
#endif

namespace bot {

struct BackendConfig {
  RuntimeConfig rt = {};
  CUDAConfig cuda = {};
  SimConfig sim = {};
};

struct Backend;

Backend * backendInit(BackendConfig cfg);
void backendShutdown(Backend *backend);

RuntimeState * backendRuntimeState(Backend *backend);
RTStateHandle backendRTStateHandle(Backend *backend);
void backendSyncStepWorlds(Backend *backend);

int backendGPUID(Backend *backend);

}

#include "backend.inl"
