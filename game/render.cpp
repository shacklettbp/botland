#include "render.hpp"
#include "sim.hpp"

namespace bot {

void RenderWorld::init(Runtime &rt,
                       MemArena &persistent_arena,
                       MemArena &tmp_arena,
                       RenderBridge *render_bridge)
{
  (void)tmp_arena;

  bridge = render_bridge;

  instanceStore.init(rt, persistent_arena);
  cameraStore.init(rt, persistent_arena);
  lightStore.init(rt, persistent_arena);
}

void renderTasks(SimRT &rt, TaskExec &exec)
{
  Sim *sim = rt.sim();

  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      world->physics.bodyStore.iterate(
        rt, [&](BodyID, BodyRef ref) {
          ref.renderInstance->instanceData->position = *ref.position;
          ref.renderInstance->instanceData->rotation = *ref.rotation;
        });
    });

  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];
      world->render.numInstances.store_relaxed(0);
      world->render.numCameras.store_relaxed(0);
      world->render.numLights.store_relaxed(0);
    });

  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      RenderBridge *bridge = world->render.bridge;

      if (idx != (i32)bridge->worldID) {
        return;
      }

      world->render.instanceStore.iterate(
        rt, [&](RenderInstanceID, RenderInstanceRef ref) {
          i32 instance_idx = world->render.numInstances.fetch_add_relaxed(1);

          if (instance_idx > (i32)bridge->instances.size) {
            world->render.numInstances.fetch_add_relaxed(-1);
            return;
          }

          bridge->instances[instance_idx] = *ref.instanceData;
        });

      world->render.cameraStore.iterate(
        rt, [&](RenderCameraID, RenderCameraRef ref) {
          i32 camera_idx = world->render.numCameras.fetch_add_relaxed(1);

          if (camera_idx > (i32)bridge->cameras.size) {
            world->render.numCameras.fetch_add_relaxed(-1);
            return;
          }

          bridge->cameras[camera_idx] = *ref.cameraData;
        });

      world->render.lightStore.iterate(
        rt, [&](RenderLightID, RenderLightRef ref) {
          i32 light_idx = world->render.numLights.fetch_add_relaxed(1);

          if (light_idx > (i32)bridge->lights.size) {
            world->render.numLights.fetch_add_relaxed(-1);
            return;
          }

          bridge->lights[light_idx] = *ref.lightData;
        });

      world->render.bridge->numInstances =
        world->render.numInstances.load_relaxed();
      world->render.bridge->numCameras =
        world->render.numCameras.load_relaxed();
      world->render.bridge->numLights =
        world->render.numLights.load_relaxed();
    });

#if 0
  exec.forEachTask(
    rt, sim->numActiveWorlds, false, [&](i32 idx) {
      World *world = sim->activeWorlds[idx];

      RenderBridge *bridge = world->render.bridge;

      if (idx != bridge->worldID) {
        return;
      }

      world->render.instanceStore.iterateParallel(
        [&](RenderInstanceRef ref) {
          i32 instance_idx = world->render.numInstances.fetch_add_relaxed(1);

          if (instance_idx > bridge->maxNumInstances) {
            world->render.numInstances.fetch_add_relaxed(-1);
            return;
          }

          bridge->instances[instance_idx] = *ref.instanceData;
        });

      world->render.cameraStore.iterateParallel(
        [&](RenderCameraRef ref) {
          i32 camera_idx = world->render.numCameras.fetch_add_relaxed(1);

          if (camera_idx > bridge->maxNumCameras) {
            world->render.numCameras.fetch_add_relaxed(-1);
            return;
          }

          bridge->cameras[camera_idx] = *ref.cameraData;
        });

      world->render.lightStore.iterateParallel(
        [&](RenderLightRef ref) {
          i32 light_idx = world->render.numLights.fetch_add_relaxed(1);

          if (light_idx > bridge->maxNumLights) {
            world->render.numLights.fetch_add_relaxed(-1);
            return;
          }

          bridge->lights[light_idx] = *ref.lightData;
        });
    });
#endif
}

}
