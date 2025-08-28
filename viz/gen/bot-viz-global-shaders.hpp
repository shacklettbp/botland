#pragma once
#include <gas/gas.hpp>

namespace bot {

enum class GlobalShaderID : u32 {
  Tonemap = 0,
};

struct VizGlobalShaders : gas::CompiledShadersBlob {
  inline gas::ShaderByteCode getByteCode(GlobalShaderID id) const
  {
    return gas::CompiledShadersBlob::getByteCode((u32)id);
  }
};

}
