#include "sim.hpp"
#include "prims.hpp"
#include <rt/log.hpp>
#include <rt/math.hpp>
#include <cstdio>

namespace bot {

static void logEvent(SimRT &rt, const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);

  World *world = rt.world();

  EventLog *prev_log = world->eventLogTail;
  EventLog *log = rt.arenaAlloc<EventLog>(world->persistentArena);
  prev_log->next = log;
  world->eventLogTail = log;
  
  int num_chars = vsnprintf(nullptr, 0, fmt, args);
  log->text = rt.arenaAllocN<char>(world->persistentArena, num_chars + 1);
  int num_written = vsnprintf(log->text, num_chars + 1, fmt, args);
  assert(num_written == num_chars);

  log->next = nullptr;
}

// Destroy unit and unlink from turn order
static inline void killUnit(SimRT &rt,  World *world, UnitID id)
{
  UnitPtr u = world->units.get(id);
  
  logEvent(rt, "Unit %s killed", u->name.data);

  TurnListLinkedList &turnListItem = u->turnListItem;
  
  UnitID prev_id = turnListItem.prev;
  UnitID next_id = turnListItem.next;
  
  if (world->turnHead == id) {
    world->turnHead = next_id;
  }
  
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
  
  world->grid[u->pos.y][u->pos.x].actorID = GenericID::none();
  
  world->killUnitJobs.add(KillUnitJob { .unitID = id });
  world->numAliveUnits--;
}

static inline constexpr auto ALL_ATTACK_PROPERTIES = std::to_array<AttackProperties>({
  { .range = 1, .damage = 2, .effect = AttackEffect::None },
  { .range = 2, .damage = 1, .effect = AttackEffect::None },
  { .range = 1, .damage = 1, .effect = AttackEffect::PoisonSpread },
  { .range = 1, .damage = 0, .effect = AttackEffect::HealingBloom },
  { .range = 1, .damage = 1, .effect = AttackEffect::VampiricBite },
  { .range = 1, .damage = 1, .effect = AttackEffect::Push },
});

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
  
  world->rng = RNG(
    rand::split_i(rt.sim()->baseRND, u32(world_id >> 32), u32(world_id)));

  world->worldID = world_id;

  world->units.init(rt, world->persistentArena);
  world->locationEffects.init(rt, world->persistentArena);
  world->obstacles.init(rt, world->persistentArena);

  world->killUnitJobs.init(rt, world->tmpArena);
  
  auto initializeUnit = [&](i32 team, i32 team_offset, i32 spawn_x, i32 spawn_y) {
    UnitPtr u = world->units.create((u32)ActorType::Unit);
    world->grid[spawn_y][spawn_x].actorID = u->id.toGeneric();

    u->pos = { spawn_x, spawn_y };
    u->hp = world->rng.sampleI32(1, DEFAULT_HP);
    u->speed = world->rng.sampleI32(1, DEFAULT_SPEED);
    u->team = team;

    snprintf(u->name.data, sizeof(u->name.data), "%s%d", team == 0 ? "R" : "B", team_offset);
    
    u->attackProp =
      ALL_ATTACK_PROPERTIES[world->rng.sampleI32(0, ALL_ATTACK_PROPERTIES.size())];

    u->passiveAbility = 
      (PassiveAbility)world->rng.sampleI32(0, (i32)PassiveAbility::NUM_PASSIVE_ABILITIES);

    return u->id;
  };

  {
    i32 base_spawn_y = GRID_SIZE / 2 - TEAM_SIZE / 2; 

    for (i32 i = 0; i < TEAM_SIZE; i++) {
      i32 spawn_x = 0;
      i32 spawn_y = base_spawn_y++;
      world->playerTeam[i] = initializeUnit(0, i, spawn_x, spawn_y);
    }
  }

  {
    i32 base_spawn_y = GRID_SIZE / 2 - TEAM_SIZE / 2; 

    for (i32 i = 0; i < TEAM_SIZE; i++) {
      i32 spawn_x = GRID_SIZE - 1;
      i32 spawn_y = base_spawn_y++;
      world->enemyTeam[i] = initializeUnit(1, i, spawn_x, spawn_y);
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
  
  {
    i32 wall_config = world->rng.sampleI32(0, 3);

    auto spawn2x2Walls = [&](i32 base_x, i32 base_y) {
      for (i32 i = 0; i < 2; i++) {
        for (i32 j = 0; j < 2; j++) {
          ObstaclePtr obstacle = world->obstacles.create(u32(ActorType::Obstacle));
          obstacle->pos.x = base_x + i;
          obstacle->pos.y = base_y+ j;
          obstacle->type = ObstacleType::Wall;  
          
          world->grid[obstacle->pos.y][obstacle->pos.x].actorID = obstacle->id.toGeneric();
        }
      }
    };

    switch (wall_config) {
      case 0: break;
      case 1: {
        i32 center_x = GRID_SIZE / 2 - 1;
        i32 center_y = GRID_SIZE / 2 - 1;
        
        spawn2x2Walls(center_x, center_y);
      } break;
      case 2: {
        i32 center_x = GRID_SIZE / 2 - 1;
        spawn2x2Walls(center_x, 0);
        spawn2x2Walls(center_x, GRID_SIZE - 2);
      } break;
    }
  }

  return world;
}

void stepWorld(SimRT &rt, World *world, UnitAction action)
{
  rt.setWorld(world);

  UnitID cur_unit_id = world->turnCur;
  UnitPtr unit = world->units.get(cur_unit_id);
  
  if (!unit) { // Everyone is dead
    return;
  }

  assert(unit->hp > 0);
  
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
      dy = 1;
      break;
    case MoveAction::Down:
      dy = -1;
      break;
    default:
      break;
  }

  // Only process if there's an actual movement
  if (dx != 0 || dy != 0) {
    i32 cur_x = unit->pos.x;
    i32 cur_y = unit->pos.y;
    
    // Determine attack and movement targets based on attack type
    i32 move_x = cur_x + dx, move_y = cur_y + dy;
    
    i32 attack_x = cur_x + dx * unit->attackProp.range;
    i32 attack_y = cur_y + dy * unit->attackProp.range;
    
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
          target_unit->hp -= unit->attackProp.damage;
          attacked = true;
          
          if (target_unit->hp <= 0) {
            killUnit(rt, world, target_id);
            
            if (unit->attackProp.effect == AttackEffect::VampiricBite) {
              unit->hp += unit->attackProp.damage;
            }
          } else {
            if (unit->attackProp.effect == AttackEffect::Push) {
              i32 push_x = attack_x + dx;
              i32 push_y = attack_y + dy;
              if (push_x >= 0 && push_x < GRID_SIZE && push_y >= 0 && push_y < GRID_SIZE) {
                GenericID push_target_id = world->grid[push_y][push_x].actorID;
                if (!push_target_id) {
                  world->grid[push_y][push_x].actorID = target_id.toGeneric();
                  world->grid[attack_y][attack_x].actorID = GenericID::none();
                  target_unit->pos.x = push_x;
                  target_unit->pos.y = push_y;
                }
              }
            }
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
          
          switch (unit->attackProp.effect) {
            case AttackEffect::PoisonSpread: {
              LocationEffectPtr effect =
                world->locationEffects.create((u32)ActorType::LocationEffect);
              world->grid[cur_y][cur_x].effectID = effect->id;
              effect->pos.x = cur_x;
              effect->pos.y = cur_y;
              effect->duration = 16;
              effect->type = LocationEffectType::Poison;
            } break;
            case AttackEffect::HealingBloom: {
              LocationEffectPtr effect =
                world->locationEffects.create((u32)ActorType::LocationEffect);
              world->grid[cur_y][cur_x].effectID = effect->id;
              effect->pos.x = cur_x;
              effect->pos.y = cur_y;
              effect->duration = 2;
              effect->type = LocationEffectType::Healing;
            } break;
            default: break;
          }
        }
      }
    }
  }
  
  for (auto effect : world->locationEffects) {
    Cell &cell = world->grid[effect->pos.y][effect->pos.x];
    
    if (cell.actorID && cell.actorID.type == (i32)ActorType::Unit) {
      UnitID unit_id = UnitID::fromGeneric(cell.actorID);
      UnitPtr affected_unit = world->units.get(unit_id);
      assert(affected_unit);

      switch (effect->type) {
        case LocationEffectType::Poison: {
          affected_unit->hp -= 1;
          
          if (affected_unit->hp <= 0) {
            killUnit(rt, world, unit_id);
          }
        } break;
        case LocationEffectType::Healing: {
          affected_unit->hp += 1;
          effect->duration--;
        } break;
        default: break;
      }
    }

    if (effect->type == LocationEffectType::Poison) {
      effect->duration--;
    }

    if (effect->duration <= 0) {
      world->locationEffects.destroy(cell.effectID);
      cell.effectID = LocationEffectID::none();
    }
  }

  {
    // Advance to next alive unit
    world->turnCur = unit->turnListItem.next ? unit->turnListItem.next : world->turnHead;
  }
  
  for (auto job : world->killUnitJobs) {
    world->units.destroy(job.unitID);
  }
  
  world->killUnitJobs.clear();

  rt.releaseArena(world->tmpArena);
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
  exec.forEachTask(
    rt, sim->numActiveWorlds, true,
    [&](i32 idx) {
      stepWorld(rt, sim->activeWorlds[idx], UnitAction {
        .move = (MoveAction)sim->ml.actions[idx],
      });
    });

  exec.finish(rt);
}

}
