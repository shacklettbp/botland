#pragma once

#include <scene/geo.hpp>

namespace bot {

struct CollisionPrimitive {
  enum class Type : uint32_t {
    Sphere = 1 << 0,
    Hull = 1 << 1,
    Plane = 1 << 2,
    Capsule = 1 << 3,
    Box = 1 << 4,
  };

  struct Sphere {
    float radius;
  };

  struct Hull {
    HalfEdgeMesh halfEdgeMesh;
  };

  struct Plane {};

  struct Box {
    Vector3 dim;
  };

  struct Capsule {
    float radius;
    float cylinderHeight;
  };

  Type type;
  union {
    Sphere sphere;
    Box box;
    Plane plane;
    Hull hull;
    Capsule capsule;
  };
};

struct ObjectManager {
  CollisionPrimitive *collisionPrimitives;
  AABB *primitiveAABBs;
  AABB *rigidBodyAABBs;
  uint32_t *rigidBodyPrimitiveOffsets;
  uint32_t *rigidBodyPrimitiveCounts;
};

  
}
