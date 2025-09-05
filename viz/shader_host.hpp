#pragma once

#include <rt/types.hpp>
#include <rt/math.hpp>

#include <array>

namespace bot {

struct float3x4 {
  Vector4 rows[3];
};

struct float3x3 {
  Vector3 rows[3];
};

namespace shader {
using float4 = Vector4;
using float3 = Vector3;
using float2 = Vector2;

using uint2 = std::array<uint32_t, 2>;
using uint3 = std::array<uint32_t, 3>;
using uint4 = std::array<uint32_t, 4>;
using uint = uint32_t;

#define BOT_SHADER_HOST_CPP_INCLUDE
#include "shader_host.slang"
#undef BOT_SHADER_HOST_CPP_INCLUDE
}

using shader::RenderVertex;
using shader::GlobalPassData;
using shader::ViewData;
using shader::NonUniformScaleObjectTransform;
using shader::UnitsPerDraw;
using shader::BoardDrawData;
using shader::HealthBarPerDraw;

}
