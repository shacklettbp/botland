namespace bot {

template <typename... Ts>
void ** CUDAManager::packArgs(Runtime &rt, MemArena &arena, Ts &&...args)
{
  if constexpr (sizeof...(args) == 0) {
    void **arg_buffer = rt.arenaAllocN<void *>(arena, 6);

    arg_buffer[0] = (void *)uintptr_t(0);
    arg_buffer[1] = CU_LAUNCH_PARAM_BUFFER_POINTER;
    arg_buffer[2] = nullptr;
    arg_buffer[3] = CU_LAUNCH_PARAM_BUFFER_SIZE;
    arg_buffer[4] = &arg_buffer[0];
    arg_buffer[5] = CU_LAUNCH_PARAM_END;

    return arg_buffer;
  } else {
    u32 num_arg_bytes = 0;
    auto incrementArgSize = [&num_arg_bytes](auto v) {
      using T = decltype(v);
      num_arg_bytes = roundToAlignment(num_arg_bytes, (u32)alignof(T));
      num_arg_bytes += sizeof(T);
    };

    ( incrementArgSize(std::forward<Ts>(args)), ... );

    auto getArg0Align = [](auto arg0, auto ...) {
      return std::alignment_of_v<decltype(arg0)>;
    };

    u32 arg0_alignment = getArg0Align(args...);

    u32 total_buf_size = sizeof(void *) * 5;
    u32 arg_size_ptr_offset = roundToAlignment(
      total_buf_size, (u32)alignof(size_t));

    total_buf_size = arg_size_ptr_offset + sizeof(size_t);

    u32 arg_ptr_offset = roundToAlignment(total_buf_size, arg0_alignment);

    total_buf_size = arg_ptr_offset + num_arg_bytes;

    void **arg_buffer = rt.arenaAllocN<void *>(
      arena, divideRoundUp(total_buf_size, (u32)sizeof(void *)));

    size_t *arg_size_start = (size_t *)(
        (char *)arg_buffer + arg_size_ptr_offset);

    new (arg_size_start) size_t(num_arg_bytes);

    void *arg_start = (char *)arg_buffer + arg_ptr_offset;

    u32 cur_arg_offset = 0;
    auto copyArgs = [arg_start, &cur_arg_offset](auto v) {
      using T = decltype(v);

      cur_arg_offset = roundToAlignment(cur_arg_offset, (u32)alignof(T));

      memcpy((char *)arg_start + cur_arg_offset, &v, sizeof(T));

      cur_arg_offset += sizeof(T);
    };

    ( copyArgs(std::forward<Ts>(args)), ... );

    arg_buffer[0] = CU_LAUNCH_PARAM_BUFFER_POINTER;
    arg_buffer[1] = arg_start;
    arg_buffer[2] = CU_LAUNCH_PARAM_BUFFER_SIZE;
    arg_buffer[3] = arg_size_start;
    arg_buffer[4] = CU_LAUNCH_PARAM_END;

    return arg_buffer;
  }
}

template <typename... Ts>
void CUDAManager::launchOneOff(Runtime &rt, const char *func_name,
                               CUDALaunchConfig cfg, Ts &&...args)
{
  launch(findFn(func_name), cfg, packArgs(
      rt, rt.tmpArena(),std::forward<Ts>(args)...));
}


template <typename... Ts>
CUgraphNode CUDAManager::addLaunchGraphNode(Runtime &rt, CUgraph graph,
  const char *func_name, CUDALaunchConfig cfg, Ts &&...args)
{
  CUfunction fn = findFn(func_name);

  REQ_CU(cuFuncSetAttribute(fn,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, cfg.numSMemBytes));

  CUDA_KERNEL_NODE_PARAMS kernel_params {
    .func = fn,
    .gridDimX = cfg.numBlocksX,
    .gridDimY = cfg.numBlocksY,
    .gridDimZ = cfg.numBlocksZ,
    .blockDimX = cfg.blockSizeX,
    .blockDimY = cfg.blockSizeY,
    .blockDimZ = cfg.blockSizeZ,
    .sharedMemBytes = cfg.numSMemBytes,
    .kernelParams = nullptr,
    .extra = packArgs(rt, rt.tmpArena(), std::forward<Ts>(args)...),
    .kern = nullptr,
    .ctx = nullptr,
  };

  CUgraphNode kernel_node;
  REQ_CU(cuGraphAddKernelNode(
    &kernel_node, graph, nullptr, 0, &kernel_params));

  return kernel_node;
}

}
