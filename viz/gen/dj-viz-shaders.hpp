#pragma once
#include <gas/gas.hpp>

namespace bot {

enum class ShaderID : u32 {
  Objects = 0,
  Tonemap = 1,
};

struct VizShaders : gas::CompiledShadersBlob {
  inline gas::ShaderByteCode getByteCode(ShaderID id) const
  {
    return gas::CompiledShadersBlob::getByteCode((u32)id);
  }
};

}
