#include "sim.hpp"
#include "prims.hpp"
#include <rt/log.hpp>
#include <rt/math.hpp>

namespace bot {

static inline void unlinkUnitFromTurnOrder(SimRT &rt, World *world, UnitID id)
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

#if 0
void curUnitMove(SimRT &rt, World *world, MoveAction action)
{
  UnitID cur_unit_id = world->turnCur;

  UnitRef unit = world->units.get(cur_unit_id);
  assert(unit && *unit.hp > 0);
  
  // Calculate target position
  GridPos currentPos = *unit.pos;
  GridPos targetPos = {
    currentPos.x + action.deltaX,
    currentPos.y + action.deltaY,
  };
  
  // Check if target position is within grid bounds
  if (targetPos.x >= 0 && targetPos.x < GRID_SIZE &&
      targetPos.y >= 0 && targetPos.y < GRID_SIZE) {
    
    // BFS to verify a path exists through empty/teammate cells only,
    // and that the destination cell is empty.
    int dist[GRID_SIZE][GRID_SIZE];
    for (int yy = 0; yy < GRID_SIZE; ++yy) {
      for (int xx = 0; xx < GRID_SIZE; ++xx) {
        dist[yy][xx] = -1;
      }
    }
    
    // Destination must be empty to end move
    bool destinationEmpty = (world->grid[targetPos.y][targetPos.x].actorID == GenericID::none());
    
    int qx[GRID_SIZE * GRID_SIZE];
    int qy[GRID_SIZE * GRID_SIZE];
    int qh = 0, qt = 0;
    
    qx[qt] = currentPos.x; qy[qt] = currentPos.y; qt++;
    dist[currentPos.y][currentPos.x] = 0;
    
    auto isEnemyAt = [&](int x, int y) {
      GenericID aid = world->grid[y][x].actorID;
      if (aid == GenericID::none()) return false;
      UnitID oid = UnitID::fromGeneric(aid);
      UnitRef o = world->units.get(oid);
      return (o && *o.team != *unit.team);
    };
    
    auto isPassable = [&](int x, int y) {
      if (x < 0 || x >= GRID_SIZE || y < 0 || y >= GRID_SIZE) return false;
      return !isEnemyAt(x, y);
    };
    
    while (qh < qt) {
      int cx = qx[qh];
      int cy = qy[qh];
      qh++;
      
      // Early exit if we've already exceeded speed
      if (dist[cy][cx] >= *unit.speed) continue;
      
      const int dx[4] = { 1, -1, 0, 0 };
      const int dy[4] = { 0, 0, 1, -1 };
      
      for (int dir = 0; dir < 4; ++dir) {
        int nx = cx + dx[dir];
        int ny = cy + dy[dir];
        if (!isPassable(nx, ny)) continue;
        if (dist[ny][nx] != -1) continue;
        dist[ny][nx] = dist[cy][cx] + 1;
        qx[qt] = nx; qy[qt] = ny; qt++;
      }
    }
    
    bool reachable = destinationEmpty && dist[targetPos.y][targetPos.x] != -1 &&
                     dist[targetPos.y][targetPos.x] <= *unit.speed;
    
    if (reachable) {
      if (targetPos.x != currentPos.x || targetPos.y != currentPos.y) {
        world->grid[currentPos.y][currentPos.x].actorID = GenericID::none();
        *unit.pos = targetPos;
        world->grid[targetPos.y][targetPos.x].actorID = cur_unit_id.toGeneric();
      }
    }
    // If not reachable, unit stays in place
  }
  // If target position is out of bounds, unit stays in place

}

void curUnitAttack(SimRT &rt, World *world, AttackAction action)
{
  UnitID cur_unit_id = world->turnCur;
  UnitRef unit = world->units.get(cur_unit_id);
  assert(unit && *unit.hp > 0);

  // Delta-based attack: (0,0) is noop
  if (action.deltaX == 0 && action.deltaY == 0) {
    return;
  }

  // Target position from delta
  i32 tx = unit.pos->x + action.deltaX;
  i32 ty = unit.pos->y + action.deltaY;

  // Check bounds
  if (tx >= 0 && tx < GRID_SIZE && ty >= 0 && ty < GRID_SIZE) {
    GenericID target_gen_id = world->grid[ty][tx].actorID;
    if (target_gen_id) {
      UnitID target_id = UnitID::fromGeneric(target_gen_id);
      UnitRef target_unit = world->units.get(target_id);
      if (target_unit && *target_unit.hp > 0 && *target_unit.team != *unit.team) {
        // Attack: for now, just decrement HP by 1
        *target_unit.hp -= 1;
        if (*target_unit.hp <= 0) {
          // Remove from grid and turn order if dead
          world->grid[ty][tx].actorID = GenericID::none();
          unlinkUnitFromTurnOrder(rt, world, target_id);
        }
      }
    }
  }
}
#endif

void stepWorld(SimRT &rt, World *world, UnitAction action)
{

  {
    // Advance to next alive unit
    if (world->turnCur) {
      UnitPtr cur = world->units.get(world->turnCur);
      UnitID next_id = cur->turnListItem.next ? cur->turnListItem.next : world->turnHead;

      while (next_id) {
        UnitPtr next = world->units.get(next_id);
        next_id = next->turnListItem.next ? next->turnListItem.next : world->turnHead;
      }
    }

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
