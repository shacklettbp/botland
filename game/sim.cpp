#include "sim.hpp"
#include "prims.hpp"
#include <rt/log.hpp>
#include <rt/math.hpp>

namespace bot {

World * createWorld(
    SimRT &rt,
    u64 world_id)
{
  World *world;
  {
    MemArena persistent_arena {};
    world = rt.arenaAlloc<World>(persistent_arena);
    new (world) World {};
    world->persistentArena = persistent_arena;
  }

  world->worldID = world_id;

  return world;
}

void destroyWorld(SimRT &rt, World *world)
{
  rt.releaseArena(world->persistentArena);
}

BOT_KERNEL(botInitSim, TaskKernelConfig::singleThread(),
          const SimConfig *cfg, Sim **sim_out)
{
  Runtime rt(BOT_RT_INIT_ARGS);

  chk(cfg->numDOFSPerAgent > 0);
  chk(cfg->numActiveWorlds > 0);
  chk(cfg->maxNumAgentsPerWorld > 0);

  Sim *sim;
  {
    MemArena arena;
    sim = rt.arenaAlloc<Sim>(arena);
    new (sim) Sim {};
    sim->globalArena = arena;
  }

  sim->numActiveWorlds = cfg->numActiveWorlds;
  sim->activeWorlds = rt.arenaAllocN<World *>(
      sim->globalArena, cfg->numActiveWorlds);

  MLInterface &ml = sim->ml;

  ml.episodeDoneEvents = rt.arenaAllocN<i32>(
     sim->globalArena, cfg->numActiveWorlds);
  ml.episodeCounters = rt.arenaAllocN<u32>(
      sim->globalArena, cfg->numActiveWorlds);
  zeroN<u32>(ml.episodeCounters, cfg->numActiveWorlds);
  ml.numEpisodeDoneEvents = 0;

  i32 max_total_agents = cfg->maxNumAgentsPerWorld * cfg->numActiveWorlds;

  ml.rewards = rt.arenaAllocN<float>(
      sim->globalArena, max_total_agents);

  ml.dones = rt.arenaAllocN<bool>(
      sim->globalArena, max_total_agents);

  ml.actions = rt.arenaAllocN<float>(
      sim->globalArena, max_total_agents * cfg->numActionsPerAgent);

  ml.observations = rt.arenaAllocN<float>(
      sim->globalArena, max_total_agents * cfg->numDOFSPerAgent);

  *sim_out = sim;
}

static void envTasks(SimRT &rt, TaskExec &exec)
{
  Sim *sim = rt.sim();

  exec.forEachTask(
    rt, sim->numActiveWorlds, true,
    [&](i32 idx) {
      World *world = sim->activeWorlds[idx];
    });
}

BOT_TASK_KERNEL(botCreateWorlds, Sim *sim, const SimConfig *cfg)
{
  SimRT rt(BOT_RT_INIT_ARGS, sim);

  TaskExec exec = sim->taskMgr.start(rt);

  exec.forEachTask(
    rt, sim->numActiveWorlds, true,
    [&](i32 idx) {
      sim->activeWorlds[idx] = createWorld(rt, idx);
    });

  exec.finish(rt);
}

BOT_TASK_KERNEL(botStepWorlds, Sim *sim)
{
  SimRT rt(BOT_RT_INIT_ARGS, sim);

  TaskExec exec = sim->taskMgr.start(rt);

  envTasks(rt, exec);

  // Clear temporary arena
  exec.forEachTask(
    rt, sim->numActiveWorlds, true, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];
    });

  exec.serialTask(
    rt, [&]() {
      rt.releaseArena(sim->stepTmpArena);
    });

  exec.finish(rt);
}

}
