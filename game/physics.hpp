#pragma once

#include "rt/rt.hpp"
#include "rt/math.hpp"
#include "rt/store.hpp"

#include <scene/geo.hpp>

#include "bvh.hpp"
#include "render.hpp"
#include "physics_obj.hpp"
#include "physics_store.hpp"

namespace bot {

inline constexpr float deltaT = 0.001f;

struct PhysicsWorld {
  BodyStore bodyStore = {};
  ContactStore contactStore = {};
  MidphaseStore midphaseStore = {};
  NarrowphaseStore narrowphaseStore = {};

  BVH bvh = {};
  ObjectManager *objMgr = nullptr;

  void init(Runtime &rt,
            MemArena &persistent_arena,
            MemArena &tmp_arena,
            ObjectManager *obj_mgr);

  LeafID registerBody(BodyID ref, i32 object_id);
};

struct World;

void bvhTasks(SimRT &rt, TaskExec &exec);
void preIntegrationTasks(SimRT &rt, TaskExec &exec);
void postIntegrationTasks(SimRT &rt, TaskExec &exec);
void narrowphaseTasks(SimRT &rt, TaskExec &exec);
void physicsTasks(SimRT &rt, TaskExec &exec);

}
