#pragma once

#include "rt/math.hpp"

namespace bot {

struct alignas(16) PerspectiveCameraData {
  Vector3 position;
  Quat rotation;
  float xScale;
  float yScale;
  float zNear;
  int32_t worldIDX;
  uint32_t pad;
};

struct alignas(16) InstanceData {
  Vector3 position;
  Quat rotation;
  Diag3x3 scale;

  // If this is -1, we just use whatever default material the model
  // has defined for it.
  //
  // If this is -2, we use the color at the end of this struct.
  int32_t matID;

  int32_t objectID;
  int32_t worldIDX;

  uint32_t color;
};

struct alignas(16) LightData {
  enum Type : bool {
    Directional = true,
    Spotlight = false
  };

  Type type;

  bool castShadow;

  // Only affects the spotlight (defaults to 0 0 0).
  Vector3 position;

  // Affects both directional/spotlight.
  Vector3 direction;

  // Angle for the spotlight (default to pi/4).
  float cutoff;

  // Intensity of the light. (1.f is default)
  float intensity;

  // Gives ability to turn light on or off.
  bool active;
};
  
}
