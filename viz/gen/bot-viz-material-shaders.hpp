#pragma once
#include <gas/gas.hpp>

namespace bot {

enum class MaterialShaderID : u32 {
  Board = 0,
  Units = 1,
  HealthBar = 2,
};

struct VizMaterialShaders : gas::CompiledShadersBlob {
  inline gas::ShaderByteCode getByteCode(MaterialShaderID id) const
  {
    return gas::CompiledShadersBlob::getByteCode((u32)id);
  }
};

}
