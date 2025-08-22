#pragma once

namespace bot {

struct TaskPrimitives {
  static void prefixSum(
      Runtime &rt,
      TaskExec &exec,
      MemArena &step_tmp_arena,
      i32 *values_in,
      i32 *values_out,
      u32 num_values);
};
  
}
