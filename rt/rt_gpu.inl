namespace bot {

#ifdef BOT_GPU

GPUThreadInfo gpuThreadInfo()
{
  return { 
    .warpID = (i32)threadIdx.x / 32, 
    .laneID = (i32)threadIdx.x % 32,
  };
}

#endif

}
