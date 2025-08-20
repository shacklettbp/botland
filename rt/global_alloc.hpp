#pragma once

#include "types.hpp"

namespace bot {

constexpr inline u32 GLOBAL_ALLOC_OOM = 0;

class GlobalAlloc {
public:
  static constexpr inline u32 MAX_NUM_BLOCKS = 16 * 1024 * 1024;

  struct DeallocStatus {
    u32 freeRegionStart;
    u32 numFreeBlocks;
  };

  void init(char *mem_pool, u32 init_used_blocks, u32 num_blocks);

  u32 alloc(char *mem_pool, u32 num_blocks);
  DeallocStatus dealloc(char *mem_pool, u32 block_idx, u32 num_blocks);

  void addNewBlocks(char *mem_pool, u32 block_idx, u32 num_used_blocks,
                    u32 num_free_blocks);

  void removeFreeBlock(char *mem_pool, u32 block_idx, u32 num_retain_blocks);

private:
  static constexpr inline u32 NUM_TOP_BINS = 32;
  static constexpr inline u32 BIN_FLT_MANTISSA_BITS = 3;
  static constexpr inline u32 BIN_FLT_MANTISSA_IMPLICIT = 
      1 << BIN_FLT_MANTISSA_BITS;
  static constexpr inline u32 BIN_FLT_MANTISSA_MASK = 
      BIN_FLT_MANTISSA_IMPLICIT - 1;
  static constexpr inline u32 BINS_PER_LEAF = BIN_FLT_MANTISSA_IMPLICIT;
  static constexpr inline u32 LEAF_BINS_IDX_MASK = BIN_FLT_MANTISSA_MASK;
  static constexpr inline u32 NUM_LEAF_BINS = NUM_TOP_BINS * BINS_PER_LEAF;

  static constexpr inline u32 NUM_FREE_BITFIELDS = MAX_NUM_BLOCKS / 32;

  struct FreeBlock {
    u32 size;
    u32 freeListPrev;
    u32 freeListNext;
  };

  u32 free_top_bins_ = 0;
  u8 free_leaf_bins_[NUM_TOP_BINS];
  u32 bin_free_heads_[NUM_LEAF_BINS];
  u32 block_free_states_[NUM_FREE_BITFIELDS];

  void addFreeNode(char *mem_pool, u32 block_idx, u32 num_blocks);
  void removeFreeNode(char *mem_pool, u32 node_idx);

  inline FreeBlock & getBlock(char *mem_pool, u32 block_idx);

  inline u32 numBlocksToBinRoundUp(u32 num_blocks);
  inline u32 numBlocksToBinRoundDown(u32 num_blocks);
  inline u32 numBlocksToBin(u32 num_blocks);
};

}
