#pragma once

#include <rt/rt.hpp>
#include <rt/store.hpp>
#include <rt/math.hpp>
#include <rt/rand.hpp>

#ifdef BOT_CUDA_SUPPORT
#include "rt/cuda_comm.hpp"
#endif

namespace bot {

constexpr inline i32 GRID_SIZE = 6;
constexpr inline i32 TEAM_SIZE = 4;
constexpr inline i32 DEFAULT_HP = 5;
constexpr inline i32 DEFAULT_SPEED = 10;

struct GridPos {
  i32 x;
  i32 y;
};

enum class ActorType : u32 {
  None = 0,
  Unit,
  Obstacle,
  LocationEffect,
};

enum class MoveAction : u32 {
  Wait = 0,
  Left,
  Up,
  Right,
  Down,
  NUM_MOVE_ACTIONS,
};

enum class LocationEffectType : u32 {
  Poison,
  Healing,
  NUM_LOCATION_EFFECT_TYPES,
};

enum class AttackEffect : u32 {
  None,
  PoisonSpread,
  HealingBloom,
  VampiricBite,
  Push,
  NUM_ATTACK_EFFECTS,
};

struct AttackProperties {
  i32 range = 0;
  i32 damage = 0;
  AttackEffect effect = AttackEffect::None;
};

enum class PassiveAbility : u32 {
  None,
  HolyAura,
  NUM_PASSIVE_ABILITIES,
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

struct UnitName {
  char data[16];
};

#define UNIT_FIELDS(F) \
  F(AttackProperties, attackProp) \
  F(PassiveAbility, passiveAbility) \
  F(i32, speed) \
  F(GridPos, pos) \
  F(i32, hp) \
  F(i32, team) \
  F(TurnListLinkedList, turnListItem) \
  F(UnitName, name)

BOT_PERSISTENT_STORE(Unit, UnitID, 64, UNIT_FIELDS)

#undef UNIT_FIELDS
  
BOT_PERSISTENT_ID(LocationEffectID)
  
#define LOCATION_EFFECT_FIELDS(F) \
  F(GridPos, pos) \
  F(LocationEffectType, type) \
  F(i32, duration)
  
BOT_PERSISTENT_STORE(LocationEffect, LocationEffectID, 64, LOCATION_EFFECT_FIELDS)
  
#undef LOCATION_EFFECT_FIELDS
  
enum class ObstacleType : u32 {
  None,
  Wall,
  NUM_OBSTACLE_TYPES,
};

BOT_PERSISTENT_ID(ObstacleID)
  
#define OBSTACLE_FIELDS(F) \
  F(GridPos, pos) \
  F(ObstacleType, type)
  
BOT_PERSISTENT_STORE(Obstacle, ObstacleID, 64, OBSTACLE_FIELDS)
  
#undef OBSTACLE_FIELDS

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
  LocationEffectID effectID = {};
};

struct KillUnitJob {
  UnitID unitID = {};
};

enum class EventType : u32 {
  UnitKilled,
  UnitAction,
  UnitAttacked,
  UnitBiteHealed,
  UnitPushed,
  UnitMoveBlocked,
  UnitMoved,
  EffectReplaced,
  PoisonDropped,
  HealingDropped,
  UnitMovedOutOfBounds,
  UnitPoisoned,
  UnitHealed,
  UnitHealedAtMaxHP,
  EffectExpired,
};

struct Event {
  EventType type;
  
  struct UnitKilled { 
    UnitName unit; 
  };
  struct UnitAction {
    UnitName unit;
    MoveAction action;
  };
  struct UnitAttacked {
    UnitName attacker;
    UnitName target;
    i32 damage;
  };
  struct UnitBiteHealed {
    UnitName attacker;
    UnitName target;
    i32 healAmount;
  };
  struct UnitPushed {
    UnitName pusher;
    UnitName pushed;
    i32 fromX;
    i32 fromY;
    i32 toX;
    i32 toY;
  };
  struct UnitMoveBlocked {
    UnitName unit;
    i32 fromX;
    i32 fromY;
    i32 toX;
    i32 toY;
  };
  struct UnitMoved {
    UnitName unit;
    i32 fromX;
    i32 fromY;
    i32 toX;
    i32 toY;
  };
  struct EffectReplaced {
    i32 x;
    i32 y;
  };
  struct PoisonDropped {
    UnitName unit;
    i32 x; 
    i32 y;
  };
  struct HealingDropped {
    UnitName unit;
    i32 x;
    i32 y;
  };
  struct UnitMovedOutOfBounds {
    UnitName unit;
  };
  struct UnitPoisoned {
    UnitName unit;
    i32 x;
    i32 y;
  };
  struct UnitHealed {
    UnitName unit;
    i32 x;
    i32 y;
  };
  struct UnitHealedAtMaxHP {
    UnitName unit;
    i32 x;
    i32 y;
  };
  struct EffectExpired {
    LocationEffectType effectType;
    i32 x;
    i32 y;
  };

  union {
    UnitKilled unitKilled;
    UnitAction unitAction;
    UnitAttacked unitAttacked;
    UnitBiteHealed unitBiteHealed;
    UnitPushed unitPushed;
    UnitMoveBlocked unitMoveBlocked;
    UnitMoved unitMoved;
    EffectReplaced effectReplaced;
    PoisonDropped poisonDropped;
    HealingDropped healingDropped;
    UnitMovedOutOfBounds unitMovedOutOfBounds;
    UnitPoisoned unitPoisoned;
    UnitHealed unitHealed;
    UnitHealedAtMaxHP unitHealedAtMaxHP;
    EffectExpired effectExpired;
  };
};

struct EventLog {
  Event data = {};
  EventLog *next = nullptr;
};

struct World {
  MemArena persistentArena = {};
  MemArena tmpArena = {};
  
  RNG rng = {};

  u64 worldID = 0;

  UnitStore units = {};
  LocationEffectStore locationEffects = {};
  ObstacleStore obstacles = {};
  
  TemporaryStore<KillUnitJob, 32> killUnitJobs = {};
  
  UnitID playerTeam[TEAM_SIZE];
  UnitID enemyTeam[TEAM_SIZE];

  Cell grid[GRID_SIZE][GRID_SIZE] = {};

  UnitID turnHead = UnitID::none();
  UnitID turnCur = UnitID::none();
  i32 numAliveUnits = 0;
  
  EventLog eventLogDummy = {};
  EventLog *eventLogHead = &eventLogDummy;
  EventLog *eventLogTail = &eventLogDummy;
};

struct SimConfig {
  i32 numActiveWorlds = 0;
  i32 maxNumAgentsPerWorld = TEAM_SIZE * 2;
  RandKey baseRND = rand::initKey(0);
};

struct Sim {
  MemArena globalArena = {};

  TaskManager taskMgr = {};

  World ** activeWorlds = nullptr;
  i32 numActiveWorlds = 0;

  MLInterface ml = {};
  
  RandKey baseRND = {};
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
