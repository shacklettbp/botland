#pragma once

#include <rt/rt.hpp>
#include <rt/math.hpp>
#include <rt/host_print.hpp>

#include "render.hpp"
#include "physics.hpp"

#ifdef BOT_CUDA_SUPPORT
#include "rt/cuda_comm.hpp"
#endif

namespace bot {

struct MLInterface {
  i32 * episodeDoneEvents = nullptr;
  u32 * episodeCounters = nullptr;
  float * rewards = nullptr;
  bool * dones = nullptr;
  float * actions = nullptr;
  float * observations = nullptr;

  alignas(BOT_CACHE_LINE) i32 numEpisodeDoneEvents = 0;
};

struct World {
  MemArena persistentArena = {};

  u64 worldID = 0;

  PhysicsWorld physics = {};
  RenderWorld render = {};

  BodyID bodyTest = {};
  float moveSpeed = 1.f;
};

struct SimConfig {
  i32 numActiveWorlds = 0;
  i32 numActionsPerAgent = 0;
  i32 numDOFSPerAgent = 0;
  i32 maxNumAgentsPerWorld = 0;
  RenderBridge *renderBridge = nullptr;
  ObjectManager *objManager = nullptr;
};

// For prefix sum test
struct TestData {
  int *bufferIn = nullptr;
  int *bufferOut = nullptr;
#if 0
  void *tmpData = nullptr;
  size_t tmpDataSize = 0;
#endif
};

struct Sim {
  MemArena globalArena = {};
  MemArena stepTmpArena = {};

  TaskManager taskMgr = {};

  World ** activeWorlds = nullptr;
  i32 numActiveWorlds = 0;

  MLInterface ml = {};

  TestData testData = {};
};

class SimRT : public Runtime {
public:
  inline SimRT(BOT_RT_INIT_PARAMS, Sim *sim);

  inline Sim * sim();

  inline World * world();
  inline void setWorld(World *world);

private:
  Sim *sim_;
  World *world_;
};

World * createWorld(SimRT &rt, u64 world_id);
void destroyWorld(SimRT &rt, World *world);

BOT_KERNEL(botInitSim, TaskKernelConfig::singleThread(),
          const SimConfig *cfg, 
          Sim **sim_out);

BOT_TASK_KERNEL(botCreateWorlds, Sim *sim, const SimConfig *cfg);

BOT_TASK_KERNEL(botStepWorlds, Sim *sim);

}

#include "sim.inl"
