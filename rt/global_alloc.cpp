#include "global_alloc.hpp"
#include "rt.hpp"
#include "err.hpp"

#include <algorithm>
#include <cstdio>

namespace bot {

namespace {

inline u32 lzcntNonZero(u32 v)
{
#ifdef _MSC_VER
  unsigned long ret_val;
  _BitScanReverse(&ret_val, v);
  return 31 - ret_val;
#else
  return __builtin_clz(v);
#endif
}

inline u32 tzcntNonZero(u32 v)
{
#ifdef _MSC_VER
  unsigned long ret_val;
  _BitScanForward(&ret_val, v);
  return ret_val;
#else
  return __builtin_ctz(v);
#endif
}

// Utility functions
u32 findLowestSetBitAfter(u32 bit_mask, u32 startBitIndex)
{
  u32 maskBeforeStartIndex = (1 << startBitIndex) - 1;
  u32 maskAfterStartIndex = ~maskBeforeStartIndex;

  u32 bitsAfter = bit_mask & maskAfterStartIndex;
  if (bitsAfter == 0) {
    return GLOBAL_ALLOC_OOM;
  }

  return tzcntNonZero(bitsAfter);
}

}

void GlobalAlloc::init(char *mem_pool,
                       u32 init_used_blocks,
                       u32 num_blocks)
{
  free_top_bins_ = 0;
  for (u32 i = 0; i < NUM_TOP_BINS; i++) {
    free_leaf_bins_[i] = 0;
  }

  for (u32 i = 0; i < NUM_LEAF_BINS; i++) {
    bin_free_heads_[i] = GLOBAL_ALLOC_OOM;
  }

  chk(init_used_blocks > 0);
  chk(num_blocks <= MAX_NUM_BLOCKS);

  u32 num_free_blocks = num_blocks - init_used_blocks;
  addNewBlocks(mem_pool, 0, init_used_blocks, num_free_blocks);
}

u32 GlobalAlloc::alloc(char *mem_pool, u32 num_blocks)
{
  // Out of allocations?
  if (free_top_bins_ == 0) {
    return GLOBAL_ALLOC_OOM;
  }

  u32 free_top_bitmask = (u32)free_top_bins_;

  u32 top_bin_idx;
  u32 leaf_bin_idx;
  {
    // Round up to bin index to ensure that alloc >= bin
    // Gives us min bin index that fits the size
    u32 min_bin_idx = numBlocksToBinRoundUp(num_blocks);
    min_bin_idx = std::min(min_bin_idx, NUM_LEAF_BINS - 1);

    u32 min_top_bin_idx = min_bin_idx >> BIN_FLT_MANTISSA_BITS;
    u32 min_leaf_bin_idx = min_bin_idx & LEAF_BINS_IDX_MASK;

    top_bin_idx = min_top_bin_idx;
    leaf_bin_idx = GLOBAL_ALLOC_OOM;

    // If top bin exists, scan its leaf bin. This can fail, because the
    // properly sized leaves may actually be empty (GLOBAL_ALLOC_OOM).
    if (free_top_bitmask & (1 << top_bin_idx)) {
      leaf_bin_idx = findLowestSetBitAfter(
        free_leaf_bins_[top_bin_idx], min_leaf_bin_idx);
    }

    // If we didn't find space in top bin, we search top bin from +1
    if (leaf_bin_idx == GLOBAL_ALLOC_OOM) {
      top_bin_idx = findLowestSetBitAfter(
        free_top_bitmask, min_top_bin_idx + 1);

      // Out of space?
      if (top_bin_idx == GLOBAL_ALLOC_OOM) {
        return GLOBAL_ALLOC_OOM;
      }

      // All leaf bins here fit the alloc, since the top bin was rounded up.
      // Start leaf search from bit 0.
      // NOTE: This search can't fail since at least one leaf bit was set
      // because the top bit was set.
      leaf_bin_idx = tzcntNonZero(free_leaf_bins_[top_bin_idx]);
    }
  }

  u32 bin_idx = (top_bin_idx << BIN_FLT_MANTISSA_BITS) | leaf_bin_idx;
  chk(bin_idx < NUM_LEAF_BINS);

  // Pop the top block of the bin.
  u32 block_idx = (u32)bin_free_heads_[bin_idx];

  FreeBlock &block = getBlock(mem_pool, block_idx);
  u32 block_orig_size = (u32)block.size;

  u16 new_bin_head = block.freeListNext;
  bin_free_heads_[bin_idx] = new_bin_head;

  if (new_bin_head != GLOBAL_ALLOC_OOM) { // Bin still has free blocks
    getBlock(mem_pool, new_bin_head).freeListPrev = GLOBAL_ALLOC_OOM;
  } else { // Bin empty?
    // Remove a leaf bin mask bit
    free_leaf_bins_[top_bin_idx] &= ~(1 << leaf_bin_idx);

    // All leaf bins empty?
    if (free_leaf_bins_[top_bin_idx] == 0) {
      // Remove a top bin mask bit
      free_top_bitmask &= ~(1 << top_bin_idx);
      free_top_bins_ = free_top_bitmask;
    }
  }

  // Mark block as allocated
  {
    u32 free_bit_idx = block_idx / 32;
    u32 free_bit_offset = block_idx % 32;
    block_free_states_[free_bit_idx] |= (1 << free_bit_offset);
  }

  // Mark this new region tail as allocated
  if (num_blocks > 1) {
    u32 tail_idx = block_idx + num_blocks - 1;
    u32 free_bit_idx = tail_idx / 32;
    u32 free_bit_offset = tail_idx % 32;
    block_free_states_[free_bit_idx] |= (1 << free_bit_offset);
  }

  // Push back remainder N elements to a lower bin
  u32 block_remainder = block_orig_size - num_blocks;
  if (block_remainder > 0) {
    u32 new_block_idx = block_idx + num_blocks;
    addFreeNode(mem_pool, new_block_idx, block_remainder);
  }

  return block_idx;
}

GlobalAlloc::DeallocStatus GlobalAlloc::dealloc(
    char *mem_pool, u32 block_idx, u32 num_blocks)
{
  auto isNodeFree = [this](u32 block_idx) {
    // Note that this check catches accesses off both ends of the array
    // due to the usage of u32.
    if (block_idx >= MAX_NUM_BLOCKS) {
      return false;
    }

    u32 free_bit_idx = block_idx / 32;
    u32 free_bit_offset = block_idx % 32;

    return (block_free_states_[free_bit_idx] & (1 << free_bit_offset)) == 0;
  };

  // Double free check
  chk(!isNodeFree(block_idx));

  // Merge with neighbors...
  if (isNodeFree(block_idx - 1)) {
    FreeBlock &prev_neighbor_tail = getBlock(mem_pool, block_idx - 1);
    u32 prev_neighbor = block_idx - prev_neighbor_tail.size;

    removeFreeNode(mem_pool, prev_neighbor);

    num_blocks += prev_neighbor_tail.size;
    block_idx = prev_neighbor;
  }

  if (isNodeFree(block_idx + num_blocks)) {
    u32 next_neighbor = block_idx + num_blocks;

    removeFreeNode(mem_pool, next_neighbor);

    num_blocks += getBlock(mem_pool, next_neighbor).size;
  }

  addFreeNode(mem_pool, block_idx, num_blocks);

  return {
    .freeRegionStart = block_idx,
    .numFreeBlocks = num_blocks,
  };
}

void GlobalAlloc::addNewBlocks(
    char *mem_pool, u32 block_idx, u32 num_used_blocks, u32 num_free_blocks)
{
  u32 free_start = block_idx + num_used_blocks;
  for (; block_idx != free_start; block_idx++) {
    // Mark initial blocks as allocated
    u32 free_bit_idx = block_idx / 32;
    u32 free_bit_offset = block_idx % 32;
    block_free_states_[free_bit_idx] |= (1 << free_bit_offset);
  }

  if (num_free_blocks > 0) {
    addFreeNode(mem_pool, block_idx, num_free_blocks);
  }
}

void GlobalAlloc::removeFreeBlock(
    char *mem_pool, u32 block_idx, u32 num_retain_blocks)
{
  removeFreeNode(mem_pool, block_idx);
  addFreeNode(mem_pool, block_idx, num_retain_blocks);
}

void GlobalAlloc::addFreeNode(
    char *mem_pool, u32 block_idx, u32 num_blocks)
{
  FreeBlock &block = getBlock(mem_pool, block_idx);

  // Mark the head of this block's region as free
  {
    u32 free_bit_idx = block_idx / 32;
    u32 free_bit_offset = block_idx % 32;
    block_free_states_[free_bit_idx] &= ~(1 << free_bit_offset);

    block.size = num_blocks;
  }

  // Mark the tail of this block's region as free
  if (num_blocks > 1) {
    u32 tail_idx = block_idx + num_blocks - 1;
    u32 free_bit_idx = tail_idx / 32;
    u32 free_bit_offset = tail_idx % 32;
    block_free_states_[free_bit_idx] &= ~(1 << free_bit_offset);

    getBlock(mem_pool, tail_idx).size = num_blocks;
  }

  // Round down to bin index to ensure that bin >= alloc
  u32 bin_idx = numBlocksToBinRoundDown(num_blocks);
  bin_idx = std::min(bin_idx, NUM_LEAF_BINS - 1);

  u32 top_bin_idx = bin_idx >> BIN_FLT_MANTISSA_BITS;
  u32 leaf_bin_idx = bin_idx & LEAF_BINS_IDX_MASK;

  // Take a freelist block and insert on top of the bin linked list
  u16 old_bin_head = bin_free_heads_[bin_idx];

  // Bin was empty before?
  if (old_bin_head == GLOBAL_ALLOC_OOM) {
    // Set bin mask bits
    free_leaf_bins_[top_bin_idx] |= 1 << leaf_bin_idx;
    free_top_bins_ |= 1 << top_bin_idx;
  }

#ifdef DEBUG_VERBOSE
  printf("Getting block %u from freelist[%u]\n",
         block_idx, free_offset_ + 1);
#endif

  // (next = old free head)
  block.freeListPrev = GLOBAL_ALLOC_OOM;
  block.freeListNext = old_bin_head;

  if (old_bin_head != GLOBAL_ALLOC_OOM) {
    getBlock(mem_pool, old_bin_head).freeListPrev = block_idx;
  }

  bin_free_heads_[bin_idx] = block_idx;
}

void GlobalAlloc::removeFreeNode(char *mem_pool, u32 block_idx)
{
  FreeBlock &block = getBlock(mem_pool, block_idx);

  if (block.freeListPrev != GLOBAL_ALLOC_OOM) {
    // Easy case: We have previous block. Just remove this block
    // from the middle of the list.
    getBlock(mem_pool, block.freeListPrev).freeListNext = block.freeListNext;
    if (block.freeListNext != GLOBAL_ALLOC_OOM) {
      getBlock(mem_pool, block.freeListNext).freeListPrev = block.freeListPrev;
    }
  } else {
    // Hard case: We are the first block in a bin. Find the bin.

    // Round down to bin index to ensure that bin >= alloc
    u32 bin_idx = numBlocksToBinRoundDown(block.size);
    bin_idx = std::min(bin_idx, NUM_LEAF_BINS - 1);

    u32 top_bin_idx = bin_idx >> BIN_FLT_MANTISSA_BITS;
    u32 leaf_bin_idx = bin_idx & LEAF_BINS_IDX_MASK;

    u16 new_bin_head = block.freeListNext;
    bin_free_heads_[bin_idx] = new_bin_head;

    if (new_bin_head != GLOBAL_ALLOC_OOM) { // Bin still has free blocks
      getBlock(mem_pool, new_bin_head).freeListPrev = GLOBAL_ALLOC_OOM;
    } else { // Bin empty?
      // Remove a leaf bin mask bit
      free_leaf_bins_[top_bin_idx] &= ~(1 << leaf_bin_idx);

      // All leaf bins empty?
      if (free_leaf_bins_[top_bin_idx] == 0) {
        // Remove a top bin mask bit
        u32 free_top_bitmask = (u32)free_top_bins_;
        free_top_bitmask &= ~(1 << top_bin_idx);
        free_top_bins_ = free_top_bitmask;
      }
    }
  }
}

GlobalAlloc::FreeBlock & GlobalAlloc::getBlock(
    char *mem_pool, u32 block_idx)
{
  return *(FreeBlock *)(mem_pool + (u64)GLOBAL_ALLOC_BLOCK_SIZE * (u64)block_idx);
}

// Bin sizes follow floating point (exponent + mantissa) distribution
// (piecewise linear log approx)
// This ensures that for each size class, the average overhead percentage
// stays the same
u32 GlobalAlloc::numBlocksToBinRoundUp(u32 num_blocks)
{
  // Modeled after asfloat with a small float with
  // BIN_FLT_MANTISSA_BITS mantissa size

  u32 exp = 0;
  u32 mantissa = 0;

  if (num_blocks < BIN_FLT_MANTISSA_IMPLICIT) {
    // Denorm: 0..(BIN_FLT_MANTISSA_IMPLICIT - 1)
    mantissa = num_blocks;
  } else {
    // Normalized: Hidden high bit always 1. Not stored. Just like float.
    u32 leading_zeros = lzcntNonZero(num_blocks);
    u32 highest_set_bit = 31 - leading_zeros;

    u32 mantissa_start_bit = highest_set_bit - BIN_FLT_MANTISSA_BITS;
    exp = mantissa_start_bit + 1;
    mantissa = (num_blocks >> mantissa_start_bit) & BIN_FLT_MANTISSA_MASK;

    u32 low_bits_mask = (1 << mantissa_start_bit) - 1;

    // Round up!
    if ((num_blocks & low_bits_mask) != 0) {
      mantissa++;
    }
  }

  // + allows mantissa->exp overflow for round up
  return (exp << BIN_FLT_MANTISSA_BITS) + mantissa;
}

u32 GlobalAlloc::numBlocksToBinRoundDown(u32 num_blocks)
{
  // Modeled after asfloat with a small float with mantissa bits = 3
  //
  u32 exp = 0;
  u32 mantissa = 0;

  if (num_blocks < BIN_FLT_MANTISSA_IMPLICIT) {
    // Denorm: 0..(BIN_FLT_MANTISSA_IMPLICIT - 1)
    mantissa = num_blocks;
  } else {
    // Normalized: Hidden high bit always 1. Not stored. Just like float.
    u32 leading_zeros = lzcntNonZero(num_blocks);
    u32 highest_set_bit = 31 - leading_zeros;

    u32 mantissa_start_bit = highest_set_bit - BIN_FLT_MANTISSA_BITS;
    exp = mantissa_start_bit + 1;
    mantissa = (num_blocks >> mantissa_start_bit) & BIN_FLT_MANTISSA_MASK;
  }

  return (exp << BIN_FLT_MANTISSA_BITS) | mantissa;
}

u32 GlobalAlloc::numBlocksToBin(u32 num_blocks)
{
  u32 exponent = num_blocks >> BIN_FLT_MANTISSA_BITS;
  u32 mantissa = num_blocks & BIN_FLT_MANTISSA_MASK;
  if (exponent == 0) {
    // Denorms
    return mantissa;
  } else {
    return (mantissa | BIN_FLT_MANTISSA_IMPLICIT) << (exponent - 1);
  }
}

}
