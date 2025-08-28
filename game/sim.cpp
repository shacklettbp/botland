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

  world->units.init(rt, world->persistentArena);

  {
    i32 base_spawn_x = GRID_SIZE / 2 - TEAM_SIZE / 2; 

    for (i32 i = 0; i < TEAM_SIZE; i++) {
      i32 spawn_x = base_spawn_x++;
      i32 spawn_y = 0;

      UnitID id = world->units.create(rt, (u32)ActorType::Unit);

      world->playerTeam[i] = id;
      world->grid[spawn_y][spawn_x].actorID = id.toGeneric();

      UnitRef unit = world->units.get(rt, id);

      *unit.pos = { spawn_x, spawn_y };
      *unit.hp = DEFAULT_HP;
      *unit.speed = DEFAULT_SPEED;
      *unit.team = 0;
    }
  }

  {
    i32 base_spawn_x = GRID_SIZE / 2 - TEAM_SIZE / 2; 

    for (i32 i = 0; i < TEAM_SIZE; i++) {
      i32 spawn_x = base_spawn_x++; 
      i32 spawn_y = GRID_SIZE - 1;

      UnitID id = world->units.create(rt, (u32)ActorType::Unit);

      world->enemyTeam[i] = id;
      world->grid[spawn_y][spawn_x].actorID = id.toGeneric();

      UnitRef unit = world->units.get(rt, id);

      *unit.pos = { spawn_x, spawn_y };
      *unit.hp = DEFAULT_HP;
      *unit.speed = DEFAULT_SPEED;
      *unit.team = 1;
    }
  }

  return world;
}

static void ste

void destroyWorld(SimRT &rt, World *world)
{
  rt.releaseArena(world->persistentArena);
}

BOT_KERNEL(botInitSim, TaskKernelConfig::singleThread(),
          const SimConfig *cfg, Sim **sim_out)
{
  Runtime rt(BOT_RT_INIT_ARGS);

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

  // Turn based game.
  i32 max_total_agents = 1;

  ml.rewards = rt.arenaAllocN<float>(
      sim->globalArena, max_total_agents);

  ml.dones = rt.arenaAllocN<bool>(
      sim->globalArena, max_total_agents);

  constexpr i32 num_actions_per_agent = sizeof(UnitAction) / sizeof(i32);
  static_assert(num_actions_per_agent * sizeof(i32) == sizeof(UnitAction));

  ml.actions = rt.arenaAllocN<float>(
      sim->globalArena, max_total_agents * num_actions_per_agent);

  constexpr i32 num_ob_floats_per_agent = sizeof(UnitObservation) / sizeof(f32);
  static_assert(num_ob_floats_per_agent * sizeof(f32) == sizeof(UnitObservation));

  ml.observations = rt.arenaAllocN<float>(
      sim->globalArena, max_total_agents * num_ob_floats_per_agent);

  *sim_out = sim;
}

BOT_TASK_KERNEL(botCreateWorlds, Sim *sim, const SimConfig *)
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

  exec.forEachTask(
    rt, sim->numActiveWorlds, true,
    [&](i32 idx) {
      stepWorld(rt, sim->activeWorlds[idx]);
    });

  exec.finish(rt);
}

}
