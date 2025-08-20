#include "rt.hpp"
#include "os.hpp"
#include "cuda_utils.hpp"

#include <nvrtc.h>
#include <nvJitLink.h>

#include <fstream>

namespace bot {

[[noreturn]] void nvrtcError(
        nvrtcResult err, const char *file,
        int line, const char *funcname)
{
  fatal(file, line, funcname, "%s", nvrtcGetErrorString(err));
}

[[noreturn]] void nvJitLinkError(
        nvJitLinkResult err, const char *file,
        int line, const char *funcname)
{
  const char *err_str;
  switch(err) {
  case NVJITLINK_SUCCESS: {
      FATAL("Passed NVJITLINK_SUCCESS to nvJitLinkError");
  } break;
  case NVJITLINK_ERROR_UNRECOGNIZED_OPTION: {
      err_str = "Unrecognized option";
  } break;
  case NVJITLINK_ERROR_MISSING_ARCH: {
      err_str = "Need to specify -arch=sm_NN";
  } break;
  case NVJITLINK_ERROR_INVALID_INPUT: {
      err_str = "Invalid Input";
  } break;
  case NVJITLINK_ERROR_PTX_COMPILE: {
      err_str = "PTX compilation error";
  } break;
  case NVJITLINK_ERROR_NVVM_COMPILE: {
      err_str = "NVVM compilation error";
  } break;
  case NVJITLINK_ERROR_INTERNAL: {
      err_str = "Internal error";
  } break;
  default: {
      err_str = "Unknown error";
  } break;
  }

  fatal(file, line, funcname, "nvJitLink error: %s", err_str);
}

inline void checkNVRTC(nvrtcResult res, const char *file,
                       int line, const char *funcname)
{
  if (res != NVRTC_SUCCESS) {
    nvrtcError(res, file, line, funcname);
  }
}

inline void checknvJitLink(nvJitLinkResult res, const char *file,
                           int line, const char *funcname)
{
  if (res != NVJITLINK_SUCCESS) {
    nvJitLinkError(res, file, line, funcname);
  }
}

#define ERR_NVRTC(err) \
    ::bot::nvrtcError((err), __FILE__, __LINE__,\
                              BOT_COMPILER_FUNCTION_NAME)
#define ERR_NVJITLINK(err) \
    ::bot::nvJitLinkError((err), __FILE__, __LINE__,\
                                  BOT_COMPILER_FUNCTION_NAME)

#define REQ_NVRTC(expr) \
    ::bot::checkNVRTC((expr), __FILE__, __LINE__,\
                              BOT_COMPILER_FUNCTION_NAME)
#define REQ_NVJITLINK(expr) \
    ::bot::checknvJitLink((expr), __FILE__, __LINE__,\
                                  BOT_COMPILER_FUNCTION_NAME)

}

using namespace bot;

namespace {

enum class FrontendMode : u32 {
  CC,
  LD,
  LTO_CC,
  LTO_LD,
  Error,
};

struct CompileObj {
  char *data;
  u64 numBytes;
  const char *path;
};

Span<char> compileCPPFile(Runtime &rt, const char *arch_str,
                          const char **flags, i32 num_flags,
                          const char *src, const char *src_path,
                          bool lto_compile)
{
  auto print_compile_log = [&](nvrtcProgram prog) {
    // Retrieve log output
    size_t log_size = 0;
    REQ_NVRTC(nvrtcGetProgramLogSize(prog, &log_size));

    if (log_size > 1) {
      ArenaRegion tmp_region = rt.beginTmpRegion();

      char *nvrtc_log = rt.tmpAllocN<char>(log_size);
      REQ_NVRTC(nvrtcGetProgramLog(prog, nvrtc_log));
      fprintf(stderr, "%s\n\n", nvrtc_log);

      rt.endTmpRegion(tmp_region);
    }
  };

  nvrtcProgram prog;
  REQ_NVRTC(nvrtcCreateProgram(&prog, src, src_path, 0, nullptr, nullptr));

  ArenaRegion tmp_region = rt.beginTmpRegion();

  const char **flags_with_arch = rt.tmpAllocN<const char *>(num_flags + 2);

  copyN<const char *>(flags_with_arch, flags, num_flags);
  flags_with_arch[num_flags] = "-arch";
  flags_with_arch[num_flags + 1] = arch_str;

  nvrtcResult res = nvrtcCompileProgram(prog, num_flags + 2, flags_with_arch);

  rt.endTmpRegion(tmp_region);

  print_compile_log(prog);

  if (res != NVRTC_SUCCESS) {
    ERR_NVRTC(res);
  }

  size_t num_code_bytes_out;
  char *code_out;

  if (lto_compile) {
    REQ_NVRTC(nvrtcGetLTOIRSize(prog, &num_code_bytes_out));
    code_out  = rt.resultAllocN<char>(num_code_bytes_out);
    REQ_NVRTC(nvrtcGetLTOIR(prog, code_out));
  } else {
    REQ_NVRTC(nvrtcGetPTXSize(prog, &num_code_bytes_out));
    code_out  = rt.resultAllocN<char>(num_code_bytes_out);
    REQ_NVRTC(nvrtcGetPTX(prog, code_out));
  }

  REQ_NVRTC(nvrtcDestroyProgram(&prog));

  return Span<char>(code_out, num_code_bytes_out);
}

Span<char> link(Runtime &rt, const char *arch_str,
                const char **flags, i32 num_flags,
                CompileObj *objs, i32 num_objs,
                bool ltoir_input)
{
  const char **flags_with_arch = rt.tmpAllocN<const char *>(num_flags + 1);
  copyN<const char *>(flags_with_arch, flags, num_flags);

  const char *arch_arg_prefix = "-arch=";
  i32 arch_arg_len = strlen(arch_arg_prefix) + strlen(arch_str) + 1;
  char *arch_arg = rt.tmpAllocN<char>(arch_arg_len);
  snprintf(arch_arg, arch_arg_len, "%s%s", arch_arg_prefix, arch_str);
  flags_with_arch[num_flags] = arch_arg;

  nvJitLinkHandle linker;
  REQ_NVJITLINK(nvJitLinkCreate(&linker, num_flags + 1, flags_with_arch));

  nvJitLinkInputType data_type;
  if (ltoir_input) {
    data_type = NVJITLINK_INPUT_LTOIR;
  } else {
    data_type = NVJITLINK_INPUT_PTX;
  }

  auto printLinkerLogs = [&]() {
    size_t info_log_size, err_log_size;
    REQ_NVJITLINK(
        nvJitLinkGetInfoLogSize(linker, &info_log_size));
    REQ_NVJITLINK(
        nvJitLinkGetErrorLogSize(linker, &err_log_size));

    if (info_log_size > 0) {
      char *info_log = rt.tmpAllocN<char>(info_log_size);

      REQ_NVJITLINK(nvJitLinkGetInfoLog(linker, info_log));

      fprintf(stderr, "%s\n", info_log);
    }

    if (err_log_size > 0) {
      char *err_log = rt.tmpAllocN<char>(err_log_size);
      REQ_NVJITLINK(nvJitLinkGetErrorLog(linker, err_log));

      fprintf(stderr, "%s\n", err_log);
    }
  };

  auto checkLinker = [&](nvJitLinkResult res) {
    if (res != NVJITLINK_SUCCESS) {
      fprintf(stderr, "CUDA linking Failed!\n");

      printLinkerLogs();

      fprintf(stderr, "\n");

      ERR_NVJITLINK(res);
    }
  };

  for (i32 i = 0; i < num_objs; i++) {
    CompileObj obj = objs[i];

    checkLinker(nvJitLinkAddData(linker, data_type,
        obj.data, obj.numBytes, obj.path));
  }

  checkLinker(nvJitLinkComplete(linker));

  size_t cubin_size;
  REQ_NVJITLINK(nvJitLinkGetLinkedCubinSize(linker, &cubin_size));
  char *cubin = rt.resultAllocN<char>(cubin_size);
  REQ_NVJITLINK(nvJitLinkGetLinkedCubin(linker, cubin));

  REQ_NVJITLINK(nvJitLinkDestroy(&linker));

  return Span<char>(cubin, cubin_size);
}

char * getCurrentGPUArch(Runtime &rt, MemArena &arena)
{
  int num_gpus = 0;
  REQ_CUDA(cudaGetDeviceCount(&num_gpus));

  if (num_gpus == 0) {
    FATAL("Used 'native' for arch but no CUDA GPU in system");
  }

  int max_major = -1;
  int max_minor = -1;

  for (int i = 0; i < num_gpus; i++) {
    cudaDeviceProp props;
    REQ_CUDA(cudaGetDeviceProperties(&props, i));

    if (props.major > max_major || 
        (props.major == max_major && props.minor > max_minor)) {
      max_major = props.major;
      max_minor = props.minor;
    }
  }

  chk(max_major > 0 && max_minor >= 0);

  u32 num_digits = 3 + u32NumDigits(max_major) + u32NumDigits(max_minor);

  char *arch_str = rt.arenaAllocN<char>(arena, num_digits + 1);

  snprintf(arch_str, num_digits + 1, "sm_%d%d", max_major, max_minor);

  return arch_str;
}

}

int main(int argc, const char *argv[])
{
  RTStateHandle rt_state_hdl = createRuntimeState({});
  BOT_DEFER(destroyRuntimeState(rt_state_hdl));
  Runtime rt(rt_state_hdl, 0);

  auto usageErr = [&]()
  {
    fprintf(stderr,
      "USAGE: %s MODE ARCH OUT_FILE IN_FILE [IN_FILES...] -- [ARGS...]\n", argv[0]);
    exit(EXIT_FAILURE);
  };

  i32 arg_idx = 1;

  auto nextArg = [&]()
  {
    if (arg_idx == argc) {
      usageErr();
    }

    return argv[arg_idx++];
  };

  const char *mode_arg = nextArg();
  FrontendMode mode;
  if (!strcmp(mode_arg, "cc")) {
    mode = FrontendMode::CC;
  } else if (!strcmp(mode_arg, "ld")) {
    mode = FrontendMode::LD;
  } else if (!strcmp(mode_arg, "lto-cc")) {
    mode = FrontendMode::LTO_CC;
  } else if (!strcmp(mode_arg, "lto-ld")) {
    mode = FrontendMode::LTO_LD;
  } else {
    mode = FrontendMode::Error;
    usageErr();
  }

  if (mode == FrontendMode::CC || mode == FrontendMode::LTO_CC) {
    const char *arch_str = nextArg();
    if (!strcmp(arch_str, "native")) {
      arch_str = getCurrentGPUArch(rt, rt.tmpArena());
    }

    const char *out_path = nextArg();
    const char *in_path = nextArg();
    
    const char **flags;
    {
      const char *delim = nextArg();
      if (strcmp(delim, "--") != 0) {
        usageErr();
      }

      flags = argv + arg_idx;
    }

    i32 num_flags = argc - arg_idx;

    u64 src_num_bytes;
    char *src = readFileAsString(rt, rt.tmpArena(), in_path, &src_num_bytes);
    if (!src) {
      FATAL("%s: Failed to read %s", argv[0], in_path);
    }

    Span<char> compiled = compileCPPFile(
        rt, arch_str, flags, num_flags, src, in_path,
        mode == FrontendMode::LTO_CC);

    if (!writeFile(out_path, compiled.data(), compiled.size())) {
      FATAL("%s: Failed to write %s", argv[0], out_path);
    }
  } else if (mode == FrontendMode::LD || mode == FrontendMode::LTO_LD) {
    const char *arch_str = nextArg();
    if (!strcmp(arch_str, "native")) {
      arch_str = getCurrentGPUArch(rt, rt.tmpArena());
    }

    const char *out_path = nextArg();

    CompileObj *objs = rt.tmpAllocN<CompileObj>(argc - arg_idx);
    i32 num_objs = 0;
    while (true) {
      const char *arg = nextArg();
      if (!strcmp(arg, "--")) {
        break;
      }

      CompileObj &obj = objs[num_objs++];
      obj.data = readFile(rt, rt.tmpArena(), arg, &obj.numBytes);
      obj.path = arg;

      if (!obj.data) {
        FATAL("%s: Failed to read %s", argv[0], arg);
      }
    }

    const char **flags = argv + arg_idx;
    i32 num_flags = argc - arg_idx;

    Span<char> linked = link(
      rt, arch_str, flags, num_flags, objs, num_objs, mode == FrontendMode::LTO_LD);

    if (!writeFile(out_path, linked.data(), linked.size())) {
      FATAL("%s: Failed to write %s", argv[0], out_path);
    }
  } 
}
