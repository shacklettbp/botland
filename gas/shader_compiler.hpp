#pragma once

#include "gas.hpp"

#include <rt/span.hpp>
#include <rt/stack_alloc.hpp>

namespace bot::gas {

struct ShaderMacroDefinition {
  const char *name;
  const char *value;
};

struct ShaderCompileArgs {
  const char *path;
  const char *str = nullptr; 
  Span<const char *const> includeDirs = {};
  Span<const ShaderMacroDefinition> macroDefinitions = {};

  static inline constexpr auto allTargets = std::to_array({
    ShaderByteCodeType::SPIRV,
    ShaderByteCodeType::WGSL,
  });
  Span<const ShaderByteCodeType> targets = allTargets;
};

struct ShaderParamBlockReflectionResult {
  Span<const ParamBlockTypeInit> spirv;
  Span<const ParamBlockTypeInit> mtl;
  Span<const ParamBlockTypeInit> dxil;
  Span<const ParamBlockTypeInit> wgsl;

  Span<const char> diagnostics;
  bool success;

  inline Span<const ParamBlockTypeInit> getParamBlocksForBackend(
      ShaderByteCodeType bytecode_type);
};

struct ShaderCompileResult {
  ShaderByteCode spirv;
  ShaderByteCode mtl;
  ShaderByteCode dxil;
  ShaderByteCode wgsl;

  Span<const char> diagnostics;
  Span<const char *> dependencies;
  bool success;

  inline ShaderByteCode getByteCodeForBackend(
      ShaderByteCodeType bytecode_type);
};

class ShaderCompiler {
public:
  virtual ~ShaderCompiler() = 0;

  virtual ShaderParamBlockReflectionResult paramBlockReflection(
      StackAlloc &alloc, ShaderCompileArgs args) = 0;

  virtual ShaderCompileResult compileShader(
      StackAlloc &alloc, ShaderCompileArgs args) = 0;
};

}

extern "C" {

#ifdef gas_shader_compiler_EXPORTS
#define GAS_SHADER_COMPILER_VIS BOT_EXPORT
#else
#define GAS_SHADER_COMPILER_VIS BOT_IMPORT
#endif
GAS_SHADER_COMPILER_VIS ::bot::gas::ShaderCompiler *
    gasCreateShaderCompiler();

GAS_SHADER_COMPILER_VIS void
    gasDestroyShaderCompiler(::bot::gas::ShaderCompiler *);

GAS_SHADER_COMPILER_VIS void gasStartupShaderCompilerLib();
GAS_SHADER_COMPILER_VIS void gasShutdownShaderCompilerLib();

#undef GAS_SHADER_COMPILER_VIS

}

#include "shader_compiler.inl"
