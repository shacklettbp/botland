namespace bot {

SimRT::SimRT(BOT_RT_INIT_PARAMS, Sim *sim)
  : Runtime(BOT_RT_INIT_ARGS),
    sim_(sim),
    world_(nullptr)
{}

Sim * SimRT::sim()
{
  return sim_;
}

World * SimRT::world()
{
  return world_;
}

void SimRT::setWorld(World *world)
{
  world_ = world;
}

}
