#include "sim.hpp"
#include "prims.hpp"
#include <rt/log.hpp>
#include <rt/math.hpp>
#include <cstdio>

namespace bot {

static inline void unlinkUnitFromTurnOrder(World *world, UnitID id)
{
  UnitPtr u = world->units.get(id);
  TurnListLinkedList &turnListItem = u->turnListItem;

  UnitID prev_id = turnListItem.prev;
  UnitID next_id = turnListItem.next;

  if (prev_id) {
    UnitPtr prev = world->units.get(prev_id);
    prev->turnListItem.next = next_id;
  }
  
  if (next_id) {
    UnitPtr next = world->units.get(next_id);
    next->turnListItem.prev = prev_id;
  }

  u->turnListItem.prev = UnitID::none();
  u->turnListItem.next = UnitID::none();
}

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

      UnitPtr u = world->units.create((u32)ActorType::Unit);

      world->playerTeam[i] = u->id;
      world->grid[spawn_y][spawn_x].actorID = u->id.toGeneric();

      u->pos = { spawn_x, spawn_y };
      u->hp = DEFAULT_HP;
      u->speed = DEFAULT_SPEED;
      u->team = 0;
      u->attackType = AttackType::Melee;
      snprintf(u->name.data, sizeof(u->name.data), "R%d", i);
    }
  }

  {
    i32 base_spawn_x = GRID_SIZE / 2 - TEAM_SIZE / 2; 

    for (i32 i = 0; i < TEAM_SIZE; i++) {
      i32 spawn_x = base_spawn_x++; 
      i32 spawn_y = GRID_SIZE - 1;

      UnitPtr u = world->units.create((u32)ActorType::Unit);

      world->enemyTeam[i] = u->id;
      world->grid[spawn_y][spawn_x].actorID = u->id.toGeneric();

      u->pos = { spawn_x, spawn_y };
      u->hp = DEFAULT_HP;
      u->speed = DEFAULT_SPEED;
      u->team = 1;
      u->attackType = AttackType::Melee;
      snprintf(u->name.data, sizeof(u->name.data), "B%d", i + TEAM_SIZE);
    }
  }

  { // Initialize turn order (intrusive doubly linked list by descending speed)
    world->numAliveUnits = TEAM_SIZE * 2;
    
    auto tmp_region = rt.beginTmpRegion();
    BOT_DEFER(rt.endTmpRegion(tmp_region));
    
    struct SortData {
      UnitID id;
      i32 speed;
    };

    SortData *sort_tmp = rt.tmpAllocN<SortData>(TEAM_SIZE * 2);

    i32 idx = 0;
    for (i32 i = 0; i < TEAM_SIZE; i++) {
      UnitPtr u = world->units.get(world->playerTeam[i]);
      sort_tmp[idx++] = { u->id, u->speed };
    }
    for (i32 i = 0; i < TEAM_SIZE; i++) {
      UnitPtr u = world->units.get(world->enemyTeam[i]);
      sort_tmp[idx++] = { u->id, u->speed };
    }

    // Simple bubble sort by speed descending
    for (i32 i = 0; i < world->numAliveUnits - 1; i++) {
      for (i32 j = 0; j < world->numAliveUnits - i - 1; j++) {
        if (sort_tmp[j].speed < sort_tmp[j + 1].speed) {
          SortData t = sort_tmp[j];
          sort_tmp[j] = sort_tmp[j + 1];
          sort_tmp[j + 1] = t;
        }
      }
    }

    // Link them
    world->turnHead = sort_tmp[0].id;
    world->turnCur = world->turnHead;
    for (i32 i = 0; i < world->numAliveUnits; i++) {
      UnitPtr u = world->units.get(sort_tmp[i].id);
      UnitID prev = (i == 0) ? sort_tmp[world->numAliveUnits - 1].id : sort_tmp[i - 1].id;
      UnitID next = (i == world->numAliveUnits - 1) ? sort_tmp[0].id : sort_tmp[i + 1].id;
      u->turnListItem.prev = prev;
      u->turnListItem.next = next;
    }
  }

  return world;
}

void stepWorld(SimRT &rt, World *world, UnitAction action)
{
  UnitID cur_unit_id = world->turnCur;
  UnitPtr unit = world->units.get(cur_unit_id);
  assert(unit && unit->hp > 0);

  // Calculate movement direction offsets
  i32 dx = 0, dy = 0;
  switch (action.move) {
    case MoveAction::Wait:
      // No movement
      break;
    case MoveAction::Left:
      dx = -1;
      break;
    case MoveAction::Right:
      dx = 1;
      break;
    case MoveAction::Up:
      dy = -1;
      break;
    case MoveAction::Down:
      dy = 1;
      break;
    default:
      break;
  }

  // Only process if there's an actual movement
  if (dx != 0 || dy != 0) {
    i32 cur_x = unit->pos.x;
    i32 cur_y = unit->pos.y;
    
    // Determine attack and movement targets based on attack type
    i32 attack_x = cur_x, attack_y = cur_y;
    i32 move_x = cur_x + dx, move_y = cur_y + dy;
    
    if (unit->attackType == AttackType::Melee) {
      // Melee attacks the adjacent tile
      attack_x = move_x;
      attack_y = move_y;
    } else if (unit->attackType == AttackType::RangedGapOne) {
      // RangedGapOne skips adjacent tile and attacks two tiles away
      attack_x = cur_x + 2 * dx;
      attack_y = cur_y + 2 * dy;
    }
    
    // Check if we can attack an enemy at the attack position
    bool attacked = false;
    if (attack_x >= 0 && attack_x < GRID_SIZE && attack_y >= 0 && attack_y < GRID_SIZE) {
      GenericID target_gen_id = world->grid[attack_y][attack_x].actorID;
      if (target_gen_id && target_gen_id.type == (i32)ActorType::Unit) {
        UnitID target_id = UnitID::fromGeneric(target_gen_id);
        UnitPtr target_unit = world->units.get(target_id);
        assert(target_unit);
        if (target_unit->team != unit->team) {
          // Attack the enemy unit
          target_unit->hp -= 1;
          attacked = true;
          
          if (target_unit->hp <= 0) {
            world->units.destroy(target_id);

            // Remove dead unit from grid and turn order
            world->grid[attack_y][attack_x].actorID = GenericID::none();
            unlinkUnitFromTurnOrder(world, target_id);
            world->numAliveUnits--;
          }
        }
      }
    }
    
    // If we didn't attack, try to move
    if (!attacked) {
      if (move_x >= 0 && move_x < GRID_SIZE && move_y >= 0 && move_y < GRID_SIZE) {
        GenericID move_target_id = world->grid[move_y][move_x].actorID;
        // Only move if the target tile is empty
        if (!move_target_id) {
          // Clear current position
          world->grid[cur_y][cur_x].actorID = GenericID::none();
          // Move to new position
          world->grid[move_y][move_x].actorID = unit->id.toGeneric();
          unit->pos.x = move_x;
          unit->pos.y = move_y;
        }
      }
    }
  }

  {
    // Advance to next alive unit
    world->turnCur = unit->turnListItem.next ? unit->turnListItem.next : world->turnHead;

    rt.releaseArena(world->tmpArena);
  }
}

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

  ml.actions = rt.arenaAllocN<i32>(
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
  //exec.forEachTask(
  //  rt, sim->numActiveWorlds, true,
  //  [&](i32 idx) {
  //    stepWorld(rt, sim->activeWorlds[idx]);
  //  });

  exec.finish(rt);
}

}
