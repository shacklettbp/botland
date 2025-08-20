#include "wgpu_shader_compiler.hpp"

#include <tint/tint.h>
#include <src/tint/lang/core/ir/module.h>
#include <src/tint/lang/spirv/reader/reader.h>
#include <src/tint/lang/wgsl/writer/writer.h>

#include <cassert>
#include <cstdlib>
#include <vector>

#ifdef gas_dawn_tint_EXPORTS
#define GAS_TINT_VIZ BOT_EXPORT
#else
#define GAS_TINT_VIZ BOT_IMPORT
#endif

namespace bot::gas::webgpu {

GAS_TINT_VIZ void tintInit()
{
  tint::Initialize();
}

GAS_TINT_VIZ void tintShutdown()
{
  tint::Shutdown();
}

GAS_TINT_VIZ TintConvertStatus tintConvertSPIRVToWGSL(
    void *spirv_bytecode, int64_t num_bytes,
    void *(*alloc_fn)(void *alloc_data, int64_t num_bytes), void *alloc_data,
    char **out_wgsl, int64_t *out_num_bytes, char **out_diagnostics)
{
  assert(num_bytes % 4 == 0);
  std::vector<uint32_t> tint_input(num_bytes / 4);
  memcpy(tint_input.data(), spirv_bytecode, num_bytes);
  
  tint::Program tint_prog = tint::spirv::reader::Read(tint_input);

  auto writeWGSLProgToDiagnostics = [
      out_diagnostics, &tint_prog, alloc_fn, alloc_data]
  ()
  {
    std::string wgsl_prog_str = tint::Program::printer(tint_prog);
    std::string diag_str = tint_prog.Diagnostics().Str();
    size_t num_bytes = wgsl_prog_str.size() + diag_str.size() + 3;

    *out_diagnostics = (char *)alloc_fn(alloc_data, (int64_t)num_bytes);
    char *cur_out_diagnostics = *out_diagnostics;

    memcpy(cur_out_diagnostics, wgsl_prog_str.data(), wgsl_prog_str.size() - 1);
    cur_out_diagnostics += wgsl_prog_str.size() - 1;
    *cur_out_diagnostics++ = '\n';
    *cur_out_diagnostics++ = '\n';
    memcpy(cur_out_diagnostics, diag_str.data(), diag_str.size());
  };
  
  if (tint_prog.Diagnostics().ContainsErrors()) {
    writeWGSLProgToDiagnostics();
    return TintConvertStatus::SPIRVConvertError;
  }
  
  auto tint_wgsl = tint::wgsl::writer::Generate(
      tint_prog, tint::wgsl::writer::Options {});
  
  if (tint_wgsl != tint::Success) {
    writeWGSLProgToDiagnostics();
    return TintConvertStatus::WGSLOutputError;
  }
  
  *out_num_bytes = (int64_t)tint_wgsl->wgsl.size() + 1;
  *out_wgsl = (char *)alloc_fn(alloc_data, *out_num_bytes);
  *out_diagnostics = nullptr;

  memcpy(*out_wgsl, tint_wgsl->wgsl.data(), *out_num_bytes);

  return TintConvertStatus::Success;
}

}
