#pragma once

namespace bot {

void updateLeaf(World *world, BodyRef ref);
void refitTree(World *world, BodyRef ref);
void findIntersecting(SimRT &rt, World *world, BodyID id, BodyRef ref);
void makeNarrowphaseCandidates(SimRT &rt, World *world, MidphaseCandidate *cand);
  
void runNarrowphase(
    SimRT &rt,
    World *world,
    NarrowphaseCandidate *cand
    BOT_GPU_COND(,
      const int32_t mwgpu_warp_id,
      const int32_t mwgpu_lane_id,
      bool lane_active));

}
