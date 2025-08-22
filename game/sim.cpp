#include "sim.hpp"
#include "prims.hpp"
#include <rt/log.hpp>
#include <rt/math.hpp>

namespace bot {

static BodyID createBody(SimRT &rt,
                         World *world,
                         Vector3 pos,
                         Quat rot,
                         Diag3x3 scale,
                         int object_id)
{

  // Create a render instance
  auto inst_id = world->render.instanceStore.create(rt);
  auto inst_ref = world->render.instanceStore.get(rt, inst_id);

  // Create an body
  auto body_id = world->physics.bodyStore.create(rt);
  auto body_ref = world->physics.bodyStore.get(rt, body_id);

  *(body_ref.position) = pos;
  *(body_ref.rotation) = rot;
  *(body_ref.scale) = scale;
  *(body_ref.renderInstance) = inst_ref;
  *(body_ref.linVelocity) = Vector3::zero();
  *(body_ref.angVelocity) = Vector3::zero();
  // TODO: This is not going to necessarily be this.
  *(body_ref.objectID) = object_id;
  *(body_ref.responseType) = ResponseType::Dynamic;
  *(body_ref.leafID) = world->physics.registerBody(body_id, object_id);

  inst_ref.instanceData->position = *(body_ref.position);
  inst_ref.instanceData->rotation = *(body_ref.rotation);
  inst_ref.instanceData->scale = *(body_ref.scale);
  inst_ref.instanceData->objectID = object_id;

  return body_id;
}

World * createWorld(
    SimRT &rt,
    u64 world_id,
    RenderBridge *render_bridge,
    ObjectManager *physics_bridge)
{
  World *world;
  {
    MemArena persistent_arena {};
    world = rt.arenaAlloc<World>(persistent_arena);
    new (world) World {};
    world->persistentArena = persistent_arena;
  }

  world->worldID = world_id;

  world->physics.init(
      rt,
      world->persistentArena,
      rt.sim()->stepTmpArena,
      physics_bridge);

  world->render.init(
      rt,
      world->persistentArena,
      rt.sim()->stepTmpArena,
      render_bridge);

  BodyID body_a = createBody(
      rt, world,
      Vector3 { 0.f, 0.f, 0.f },
      Quat::id(),
      Diag3x3 { 1.f, 1.f, 1.f },
      0);

  BodyID body_b = createBody(
      rt, world,
      Vector3 { 2.f, 0.f, 0.f },
      Quat::angleAxis(PI/4.f, Vector3 { 1.f, 1.f, 1.f }.normalize()),
      Diag3x3 { 1.f, 1.f, 1.f },
      0);
  (void)body_b;

  world->bodyTest = body_a;

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
      BodyRef ref = world->physics.bodyStore.get(rt, world->bodyTest);
      ref.position->x += 0.005f * world->moveSpeed;
    });
}

void prefixSumTest(SimRT &rt, TaskExec &exec)
{
  static constexpr u32 total_num_items = 2048;
  Sim *sim = rt.sim();

  exec.serialTask(rt,
    [&]() {
      sim->testData.bufferIn = rt.arenaAllocN<int>(
          sim->stepTmpArena, total_num_items);
      sim->testData.bufferOut = rt.arenaAllocN<int>(
          sim->stepTmpArena, total_num_items);
      for (int i = 0; i < total_num_items; ++i) {
        sim->testData.bufferIn[i] = 1;
        sim->testData.bufferOut[i] = 0;
      }
    });

  TaskPrimitives::prefixSum(
    rt, exec,
    sim->stepTmpArena,
    sim->testData.bufferIn,
    sim->testData.bufferOut,
    total_num_items);

  exec.serialTask(rt,
    [&]() {
      for (int i = 0; i < total_num_items; ++i) {
        if (i != sim->testData.bufferOut[i]) {
          Log::log("item {} is {}\n",
              i, sim->testData.bufferOut[i]);
        }
        assert(i == sim->testData.bufferOut[i]);
      }
    });
}

BOT_TASK_KERNEL(botCreateWorlds, Sim *sim, const SimConfig *cfg)
{
  SimRT rt(BOT_RT_INIT_ARGS, sim);

  TaskExec exec = sim->taskMgr.start(rt);

  exec.forEachTask(
    rt, sim->numActiveWorlds, true,
    [&](i32 idx) {
      sim->activeWorlds[idx] = createWorld(
          rt, idx, cfg->renderBridge, cfg->objManager);
    });

  exec.finish(rt);
}

BOT_TASK_KERNEL(botStepWorlds, Sim *sim)
{
  SimRT rt(BOT_RT_INIT_ARGS, sim);

  TaskExec exec = sim->taskMgr.start(rt);

  envTasks(rt, exec);
  physicsTasks(rt, exec);
  renderTasks(rt, exec);

  // Clear temporary arena
  exec.forEachTask(
    rt, sim->numActiveWorlds, true, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];
      world->physics.contactStore.clear();
      world->physics.midphaseStore.clear();
      world->physics.narrowphaseStore.clear();
    });

  exec.serialTask(
    rt, [&]() {
      rt.releaseArena(sim->stepTmpArena);
    });

  prefixSumTest(rt, exec);

  exec.finish(rt);
}

}
