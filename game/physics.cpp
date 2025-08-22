#include "sim.hpp"
#include "physics.hpp"
#include "collision.hpp"

namespace bot {

inline constexpr i32 NUM_INIT_BODY_CHUNK_PTRS = 16;
inline constexpr i32 NUM_INIT_CONTACT_CHUNK_PTRS = 16;

void PhysicsWorld::init(Runtime &rt, 
                        MemArena &persistent_arena,
                        MemArena &tmp_arena,
                        ObjectManager *obj_mgr)
{
  bodyStore.init(rt, persistent_arena);
  contactStore.init(rt, tmp_arena);
  midphaseStore.init(rt, tmp_arena);
  narrowphaseStore.init(rt, tmp_arena);

  bvh.init(rt, nullptr, 50, 2.f * deltaT, 
      100.0f * deltaT * deltaT, persistent_arena);
  objMgr = obj_mgr;
}

void bvhTasks(SimRT &rt, TaskExec &exec)
{
  Sim *sim = rt.sim();

  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      world->physics.bodyStore.iterate(rt, [&](BodyID, BodyRef ref) {
          updateLeaf(world, ref);
        });
    });
  
  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];
      world->physics.bvh.updateTree();
    });

  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      world->physics.bodyStore.iterate(rt, [&](BodyID, BodyRef ref) {
          refitTree(world, ref);
        });
    });
}

void preIntegrationTasks(SimRT &rt, TaskExec &exec)
{
  Sim *sim = rt.sim();

  // We want a warp to be working on each item if on GPU.
  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      world->physics.bodyStore.iterate(rt, [&](BodyID id, BodyRef ref) {
          findIntersecting(rt, world, id, ref);
      });
    });

#ifdef BOT_GPU
  exec.warpForEachTask(
    rt, sim->numActiveWorlds, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      world->physics.midphaseStore.iterate(
        [&](MidphaseCandidate *cand) {
            makeNarrowphaseCandidates(rt, world, cand);
        });
    });
#else
  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      world->physics.midphaseStore.iterate(
        [&](MidphaseCandidate *cand) {
            makeNarrowphaseCandidates(rt, world, cand);
        });
    });
#endif
}

void postIntegrationTasks(SimRT &rt, TaskExec &exec)
{
  Sim *sim = rt.sim();

  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      world->physics.bodyStore.iterate(rt, [&](BodyID, BodyRef ref) {
          updateLeaf(world, ref);
        });
    });

  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      world->physics.bodyStore.iterate(rt, [&](BodyID, BodyRef ref) {
          refitTree(world, ref);
        });
    });
}

void narrowphaseTasks(SimRT &rt, TaskExec &exec)
{
  Sim *sim = rt.sim();

#ifdef BOT_GPU
  exec.warpForEachTask(
    rt, sim->numActiveWorlds, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      world->physics.narrowphaseStore.warpIterate(
        [&](NarrowphaseCandidate *cand) {
          const int32_t mwgpu_warp_id = threadIdx.x / 32;
          const int32_t mwgpu_lane_id = threadIdx.x % 32;

          bool lane_active = (cand != nullptr);

          runNarrowphase(rt, world, cand, 
              mwgpu_warp_id, mwgpu_lane_id, lane_active);
        });
    });

#else
  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      world->physics.narrowphaseStore.iterate([&](NarrowphaseCandidate *cand) {
        runNarrowphase(rt, world, cand);
      });
    });
  
#endif
}

LeafID PhysicsWorld::registerBody(BodyID ref, i32 object_id)
{
  return bvh.reserveLeaf(ref, object_id);
}

void physicsTasks(SimRT &rt, TaskExec &exec)
{
  bvhTasks(rt, exec);
  preIntegrationTasks(rt, exec);
  narrowphaseTasks(rt, exec);

  // TODO: Solver and integration

  postIntegrationTasks(rt, exec);
  
  Sim *sim = rt.sim();

  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];
      if (world->physics.contactStore.chunksRange != 0) {
        world->moveSpeed = 0.f;
      }
    });
}

}
