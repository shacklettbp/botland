#pragma once
#include <gas/gas.hpp>

namespace bot::gas {

enum class ImGuiShaderID : u32 {
  Render = 0,
};

struct ImGuiShaders : gas::CompiledShadersBlob {
  inline gas::ShaderByteCode getByteCode(ImGuiShaderID id) const
  {
    return gas::CompiledShadersBlob::getByteCode((u32)id);
  }
};

}
