#include "sim.hpp"

#ifdef BOT_GPU

#define CUB_NAMESPACE
#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/grid/grid_queue.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>

#include <cuda/std/type_traits>

#include <iterator>

using namespace cub;
using namespace cub::detail;

template <
  int NOMINAL_BLOCK_THREADS_4B,
  int NOMINAL_ITEMS_PER_THREAD_4B,
  typename ComputeT,
  BlockLoadAlgorithm _LOAD_ALGORITHM,
  CacheLoadModifier _LOAD_MODIFIER,
  BlockStoreAlgorithm _STORE_ALGORITHM,
  BlockScanAlgorithm _SCAN_ALGORITHM,
  typename ScalingType       = detail::MemBoundScaling<NOMINAL_BLOCK_THREADS_4B, NOMINAL_ITEMS_PER_THREAD_4B, ComputeT>,
  typename DelayConstructorT = detail::default_delay_constructor_t<ComputeT>>
struct AgentScanPolicy : ScalingType
{
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM   = _LOAD_ALGORITHM;
  static constexpr CacheLoadModifier LOAD_MODIFIER     = _LOAD_MODIFIER;
  static constexpr BlockStoreAlgorithm STORE_ALGORITHM = _STORE_ALGORITHM;
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM   = _SCAN_ALGORITHM;

  struct detail
  {
    using delay_constructor_t = DelayConstructorT;
  };
};

static inline constexpr uint32_t NUM_ITEMS_PER_THREAD = 15;

struct AgentScanCustom
{
  // This is inspired by the default policy
  using AgentScanPolicyT = AgentScanPolicy<
    bot::GPU_DEFAULT_SM_CFG::NUM_THREADS_PER_BLOCK,
    NUM_ITEMS_PER_THREAD,
    int32_t,
    BLOCK_LOAD_WARP_TRANSPOSE,
    LOAD_CA,
    BLOCK_STORE_WARP_TRANSPOSE,
    BLOCK_SCAN_WARP_SCANS
  >;

  using InputIteratorT = int32_t *;
  using OutputIteratorT = int32_t *;
  using ScanOpT = ::cuda::std::plus<>;
  using InitValueT = int32_t;
  using OffsetT = uint32_t;
  using AccumT = int32_t;
  static inline constexpr bool ForceInclusive = false;

  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  // The input value type
  using InputT = cub::detail::value_t<InputIteratorT>;

  // Tile status descriptor interface type
  using ScanTileStateT = ScanTileState<AccumT>;

  // Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using WrappedInputIteratorT =
    ::cuda::std::_If<std::is_pointer<InputIteratorT>::value,
                     CacheModifiedInputIterator<AgentScanPolicyT::LOAD_MODIFIER, InputT, OffsetT>,
                     InputIteratorT>;

  // Constants
  enum
  {
    // Inclusive scan if no init_value type is provided
    HAS_INIT     = !std::is_same<InitValueT, NullType>::value,
    IS_INCLUSIVE = ForceInclusive || !HAS_INIT, // We are relying on either initial value not being `NullType`
                                                // or the ForceInclusive tag to be true for inclusive scan
                                                // to get picked up.
    BLOCK_THREADS    = AgentScanPolicyT::BLOCK_THREADS,
    ITEMS_PER_THREAD = AgentScanPolicyT::ITEMS_PER_THREAD,
    TILE_ITEMS       = BLOCK_THREADS * ITEMS_PER_THREAD,
  };

  // Parameterized BlockLoad type
  using BlockLoadT =
    BlockLoad<AccumT,
              AgentScanPolicyT::BLOCK_THREADS,
              AgentScanPolicyT::ITEMS_PER_THREAD,
              AgentScanPolicyT::LOAD_ALGORITHM>;

  // Parameterized BlockStore type
  using BlockStoreT =
    BlockStore<AccumT,
               AgentScanPolicyT::BLOCK_THREADS,
               AgentScanPolicyT::ITEMS_PER_THREAD,
               AgentScanPolicyT::STORE_ALGORITHM>;

  // Parameterized BlockScan type
  using BlockScanT = BlockScan<AccumT, AgentScanPolicyT::BLOCK_THREADS, AgentScanPolicyT::SCAN_ALGORITHM>;

  // Callback type for obtaining tile prefix during block scan
  using DelayConstructorT     = typename AgentScanPolicyT::detail::delay_constructor_t;
  using TilePrefixCallbackOpT = TilePrefixCallbackOp<AccumT, ScanOpT, ScanTileStateT, 0 /* PTX */, DelayConstructorT>;

  // Stateful BlockScan prefix callback type for managing a running total while
  // scanning consecutive tiles
  using RunningPrefixCallbackOp = BlockScanRunningPrefixOp<AccumT, ScanOpT>;

  // Shared memory type for this thread block
  union _TempStorage
  {
    // Smem needed for tile loading
    typename BlockLoadT::TempStorage load;

    // Smem needed for tile storing
    typename BlockStoreT::TempStorage store;

    struct ScanStorage
    {
      // Smem needed for cooperative prefix callback
      typename TilePrefixCallbackOpT::TempStorage prefix;

      // Smem needed for tile scanning
      typename BlockScanT::TempStorage scan;
    } scan_storage;
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------

  _TempStorage& temp_storage; ///< Reference to temp_storage
  WrappedInputIteratorT d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  ScanOpT scan_op; ///< Binary scan operator
  InitValueT init_value; ///< The init_value element for ScanOpT

  //---------------------------------------------------------------------
  // Block scan utility methods
  //---------------------------------------------------------------------

  /**
   * Exclusive scan specialization (first tile)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScanTile(
    AccumT (&items)[ITEMS_PER_THREAD],
    AccumT init_value,
    ScanOpT scan_op,
    AccumT& block_aggregate,
    ::cuda::std::false_type /*is_inclusive*/)
  {
    BlockScanT(temp_storage.scan_storage.scan).ExclusiveScan(items, items, init_value, scan_op, block_aggregate);
    block_aggregate = scan_op(init_value, block_aggregate);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void ScanTileInclusive(
    AccumT (&items)[ITEMS_PER_THREAD],
    AccumT init_value,
    ScanOpT scan_op,
    AccumT& block_aggregate,
    ::cuda::std::true_type /*has_init*/)
  {
    BlockScanT(temp_storage.scan_storage.scan).InclusiveScan(items, items, init_value, scan_op, block_aggregate);
    block_aggregate = scan_op(init_value, block_aggregate);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void ScanTileInclusive(
    AccumT (&items)[ITEMS_PER_THREAD],
    InitValueT /*init_value*/,
    ScanOpT scan_op,
    AccumT& block_aggregate,
    ::cuda::std::false_type /*has_init*/)

  {
    BlockScanT(temp_storage.scan_storage.scan).InclusiveScan(items, items, scan_op, block_aggregate);
  }

  /**
   * Inclusive scan specialization (first tile)
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void ScanTile(
    AccumT (&items)[ITEMS_PER_THREAD],
    InitValueT init_value,
    ScanOpT scan_op,
    AccumT& block_aggregate,
    ::cuda::std::true_type /*is_inclusive*/)
  {
    ScanTileInclusive(items, init_value, scan_op, block_aggregate, ::cuda::std::bool_constant<HAS_INIT>());
  }

  /**
   * Exclusive scan specialization (subsequent tiles)
   */
  template <typename PrefixCallback>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ScanTile(AccumT (&items)[ITEMS_PER_THREAD],
           ScanOpT scan_op,
           PrefixCallback& prefix_op,
           ::cuda::std::false_type /*is_inclusive*/)
  {
    BlockScanT(temp_storage.scan_storage.scan).ExclusiveScan(items, items, scan_op, prefix_op);
  }

  /**
   * Inclusive scan specialization (subsequent tiles)
   */
  template <typename PrefixCallback>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ScanTile(AccumT (&items)[ITEMS_PER_THREAD],
           ScanOpT scan_op,
           PrefixCallback& prefix_op,
           ::cuda::std::true_type /*is_inclusive*/)
  {
    BlockScanT(temp_storage.scan_storage.scan).InclusiveScan(items, items, scan_op, prefix_op);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE AgentScanCustom(
    TempStorage& temp_storage, InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, InitValueT init_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , init_value(init_value)
  {}

  template <bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeTile(OffsetT num_remaining, int tile_idx, OffsetT tile_offset, ScanTileStateT& tile_state)
  {
    // Load items
    AccumT items[ITEMS_PER_THREAD];

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element because collectives are
      // not suffix guarded.
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items, num_remaining, *(d_in + tile_offset));
    }
    else
    {
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
    }

    __syncthreads();

    // Perform tile scan
    if (tile_idx == 0)
    {
      // Scan first tile
      AccumT block_aggregate;
      ScanTile(items, init_value, scan_op, block_aggregate, ::cuda::std::bool_constant<IS_INCLUSIVE>());

      if ((!IS_LAST_TILE) && (threadIdx.x == 0))
      {
        tile_state.SetInclusive(0, block_aggregate);
      }
    }
    else
    {
      // Scan non-first tile
      TilePrefixCallbackOpT prefix_op(tile_state, temp_storage.scan_storage.prefix, scan_op, tile_idx);
      ScanTile(items, scan_op, prefix_op, ::cuda::std::bool_constant<IS_INCLUSIVE>());
    }

    __syncthreads();

    // Store items
    if (IS_LAST_TILE)
    {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items, num_remaining);
    }
    else
    {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeRange(OffsetT num_items, ScanTileStateT& tile_state, int start_tile)
  {
    // Blocks are launched in increasing order, so just assign one tile per
    // block

    // Current tile index
    int tile_idx = start_tile + blockIdx.x;

    // Global offset for the current tile
    OffsetT tile_offset = OffsetT(TILE_ITEMS) * tile_idx;

    // Remaining items (including this tile)
    OffsetT num_remaining = num_items - tile_offset;

    if (num_remaining > TILE_ITEMS)
    {
      // Not last tile
      ConsumeTile<false>(num_remaining, tile_idx, tile_offset, tile_state);
    }
    else if (num_remaining > 0)
    {
      // Last tile
      ConsumeTile<true>(num_remaining, tile_idx, tile_offset, tile_state);
    }
  }

  template <bool IS_FIRST_TILE, bool IS_LAST_TILE>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ConsumeTile(OffsetT tile_offset, RunningPrefixCallbackOp& prefix_op, int valid_items = TILE_ITEMS)
  {
    // Load items
    AccumT items[ITEMS_PER_THREAD];

    if (IS_LAST_TILE)
    {
      // Fill last element with the first element because collectives are
      // not suffix guarded.
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items, valid_items, *(d_in + tile_offset));
    }
    else
    {
      BlockLoadT(temp_storage.load).Load(d_in + tile_offset, items);
    }

    __syncthreads();

    // Block scan
    if (IS_FIRST_TILE)
    {
      AccumT block_aggregate;
      ScanTile(items, init_value, scan_op, block_aggregate, ::cuda::std::bool_constant<IS_INCLUSIVE>());
      prefix_op.running_total = block_aggregate;
    }
    else
    {
      ScanTile(items, scan_op, prefix_op, ::cuda::std::bool_constant<IS_INCLUSIVE>());
    }

    __syncthreads();

    // Store items
    if (IS_LAST_TILE)
    {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items, valid_items);
    }
    else
    {
      BlockStoreT(temp_storage.store).Store(d_out + tile_offset, items);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeRange(OffsetT range_offset, OffsetT range_end)
  {
    BlockScanRunningPrefixOp<AccumT, ScanOpT> prefix_op(scan_op);

    if (range_offset + TILE_ITEMS <= range_end)
    {
      // Consume first tile of input (full)
      ConsumeTile<true, true>(range_offset, prefix_op);
      range_offset += TILE_ITEMS;

      // Consume subsequent full tiles of input
      while (range_offset + TILE_ITEMS <= range_end)
      {
        ConsumeTile<false, true>(range_offset, prefix_op);
        range_offset += TILE_ITEMS;
      }

      // Consume a partially-full tile
      if (range_offset < range_end)
      {
        int valid_items = range_end - range_offset;
        ConsumeTile<false, false>(range_offset, prefix_op, valid_items);
      }
    }
    else
    {
      // Consume the first tile of input (partially-full)
      int valid_items = range_end - range_offset;
      ConsumeTile<true, false>(range_offset, prefix_op, valid_items);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeRange(OffsetT range_offset, OffsetT range_end, AccumT prefix)
  {
    BlockScanRunningPrefixOp<AccumT, ScanOpT> prefix_op(prefix, scan_op);

    // Consume full tiles of input
    while (range_offset + TILE_ITEMS <= range_end)
    {
      ConsumeTile<true, false>(range_offset, prefix_op);
      range_offset += TILE_ITEMS;
    }

    // Consume a partially-full tile
    if (range_offset < range_end)
    {
      int valid_items = range_end - range_offset;
      ConsumeTile<false, false>(range_offset, prefix_op, valid_items);
    }
  }
};

namespace bot {

struct PrefixSumTmp {
  i32 *finishedCounter;
  void *data;
  size_t numBytes;
};

void TaskPrimitives::prefixSum(
  Runtime &rt, TaskExec &exec,
  MemArena &step_tmp_arena,
  i32 *values_in,
  i32 *values_out,
  u32 num_values)
{
  u32 items_per_block = AgentScanCustom::TILE_ITEMS;
  u32 num_tiles = divideRoundUp(num_values, items_per_block);

  PrefixSumTmp pf_tmp = exec.serialTask(rt,
    [&]() -> PrefixSumTmp {
      size_t num_tmp_bytes = 0;
      AgentScanCustom::ScanTileStateT::AllocationSize(
          num_tiles,
          num_tmp_bytes);

      void *d_temp_storage = rt.arenaAlloc(step_tmp_arena, 
          num_tmp_bytes + sizeof(i32));
      *(i32 *)d_temp_storage = 0;

      return {
        .finishedCounter = (i32 *)d_temp_storage,
        .data = (void *)((char *)d_temp_storage + sizeof(i32)),
        .numBytes = num_tmp_bytes,
      };
    });

  AgentScanCustom::ScanTileStateT tile_state;
  tile_state.Init(num_tiles, 
      pf_tmp.data,
      pf_tmp.numBytes);
  tile_state.InitializeStatus(num_tiles);

  // Insert this as a synchronization point
  exec.serialTask(rt, [&]() {});

  if (blockIdx.x < num_tiles) {
    AgentScanCustom::TempStorage *tmp = 
        (AgentScanCustom::TempStorage *)gpuSMemUser();

    AgentScanCustom agent(
        *tmp,
        values_in,
        values_out,
        ::cuda::std::plus<>{},
        0);

    agent.ConsumeRange(num_values, tile_state, 0);

    if (threadIdx.x == 0) {
      AtomicI32Ref finished_counter(*pf_tmp.finishedCounter);
      finished_counter.fetch_add_release(1);
    }

    __syncthreads();
  }

  if (threadIdx.x % 32 == 0) {
    AtomicI32Ref finished_counter(*pf_tmp.finishedCounter);
    i32 num_finished = 0;
    while (num_finished < num_tiles) {
      num_finished = finished_counter.load_relaxed();
    }
    atomic_thread_fence(sync::acquire);
  }

  __syncwarp();
}

}

#else
namespace bot {

void TaskPrimitives::prefixSum(
  Runtime &rt, TaskExec &exec,
  MemArena &step_tmp_arena,
  i32 *values_in,
  i32 *values_out,
  u32 num_values)
{
  exec.serialTask(
    rt, [&]() {
      for (u32 i = 0; i < num_values; ++i) {
        values_out[i] = values_in[i];
      }

      for (u32 i = 1; i < num_values; ++i) {
        values_out[i] += values_out[i - 1];
      }
    });
}

}
#endif
