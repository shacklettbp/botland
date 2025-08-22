#pragma once

#include "rt/rt.hpp"
#include "rt/cuda_cfg.hpp"

#include "sim.hpp"

#ifdef BOT_CUDA_SUPPORT
#include "rt/cuda_utils.hpp"
#endif

#include "bridge.hpp"

namespace bot {

struct BackendConfig {
  RuntimeConfig rt = {};
  CUDAConfig cuda = {};
};

struct Backend;

Backend * backendInit(BackendConfig cfg);
void backendStart(Backend *backend, SimConfig sim_cfg);
void backendShutdown(Backend *backend);

RuntimeState * backendRuntimeState(Backend *backend);
RTStateHandle backendRTStateHandle(Backend *backend);
void backendSyncStepWorlds(Backend *backend);

int backendGPUID(Backend *backend);

}

#include "backend.inl"
