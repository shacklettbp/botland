#pragma once

#include <rt/rt.hpp>
#include <rt/store.hpp>
#include <rt/math.hpp>

#ifdef BOT_CUDA_SUPPORT
#include "rt/cuda_comm.hpp"
#endif

namespace bot {

constexpr inline i32 GRID_SIZE = 8;
constexpr inline i32 TEAM_SIZE = 4;
constexpr inline i32 DEFAULT_HP = 10;
constexpr inline i32 DEFAULT_SPEED = 10;

struct GridPos {
  i32 x;
  i32 y;
};

enum class ActorType : u32 {
  None = 0,
  Unit,
  Static,
};

enum class MoveAction : u32 {
  Wait = 0,
  Left,
  Up,
  Right,
  Down,
  NUM_MOVE_ACTIONS,
};

enum class AttackType : u32 {
  Melee,
  RangedGapOne,
};

struct UnitAction {
  MoveAction move = MoveAction::Wait;
};

struct GridCellOb {
  f32 v[3];
};

struct UnitObservation {
  GridCellOb grid[GRID_SIZE][GRID_SIZE];
};

BOT_PERSISTENT_ID(UnitID)

struct TurnListLinkedList {
  UnitID prev = {};
  UnitID next = {};
};

#define UNIT_FIELDS(F) \
  F(AttackType, attackType) \
  F(i32, speed) \
  F(GridPos, pos) \
  F(i32, hp) \
  F(i32, team) \
  F(TurnListLinkedList, turnListItem)

BOT_PERSISTENT_STORE(Unit, UnitID, 64, UNIT_FIELDS)

#undef UNIT_FIELDS

struct MLInterface {
  i32 * episodeDoneEvents = nullptr;
  u32 * episodeCounters = nullptr;
  float * rewards = nullptr;
  bool * dones = nullptr;
  i32 * actions = nullptr;
  float * observations = nullptr;

  alignas(BOT_CACHE_LINE) i32 numEpisodeDoneEvents = 0;
};

struct Cell {
  GenericID actorID = {};
};

struct World {
  MemArena persistentArena = {};
  MemArena tmpArena = {};

  u64 worldID = 0;

  UnitStore units;

  UnitID playerTeam[TEAM_SIZE];
  UnitID enemyTeam[TEAM_SIZE];

  Cell grid[GRID_SIZE][GRID_SIZE] = {};

  UnitID turnHead = UnitID::none();
  UnitID turnCur = UnitID::none();
  i32 numAliveUnits = 0;
};

struct SimConfig {
  i32 numActiveWorlds = 0;
  i32 maxNumAgentsPerWorld = TEAM_SIZE * 2;
};

struct Sim {
  MemArena globalArena = {};

  TaskManager taskMgr = {};

  World ** activeWorlds = nullptr;
  i32 numActiveWorlds = 0;

  MLInterface ml = {};
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

void stepWorld(SimRT &rt, World *world, UnitAction action);

BOT_KERNEL(botInitSim, TaskKernelConfig::singleThread(),
           const SimConfig *cfg, 
           Sim **sim_out);

BOT_TASK_KERNEL(botCreateWorlds, Sim *sim, const SimConfig *cfg);

BOT_TASK_KERNEL(botStepWorlds, Sim *sim);

}

#include "sim.inl"
