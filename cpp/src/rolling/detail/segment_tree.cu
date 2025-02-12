/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cooperative_groups.h>
#include <cuda/std/functional>

#include <cstddef>

namespace cg = cooperative_groups;

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Combine values across a warp using the binop advertised by `Monoid`.
 *
 * @tparam Monoid a device binary operator type with an associative binary operator and an identity
 * @tparam T the type of the values being combined.
 * @tparam WarpSize the number of threads participating in the warp
 * @param tile Cooperative groups thread block tile with `WarpSize` threads
 * @param tile_id The rank of the warp in the tile
 * @param binop Instance of the monoidal binary operator
 * @param value The value on this thread to combine
 * @param output_offset Offset into the output array where results should be written
 * @param output Output array
 *
 * @return the combined value, only valid on thread zero in the tile.
 *
 * @note This does the downsweep of a Brent-Kung scan, writing the
 * intermediate results at each level to a contiguous dense block in
 * the output array, starting at output_offset.
 * At each level of the scan, the output offset is divided by two.
 *
 * Suppose that the warp size is 4 and the initial output offset is
 * 256. Write `value_i` for the value of the input on thread `i` in
 * the warp. Then this function will perform:
 *
 * ```
 * output[256] = inter_0 = binop(value_0, value_1)
 * output[257] = inter_1 = binop(value_2, value_3)
 * output[128] = result_0 = binop(inter_0, inter_1)
 *
 * And return `result_0` (only valid on thread 0). Moreover, the
 * `output_offset` will modified as well and updated to be `64` (i.e.
 * the provided output offset will, after the call to this function,
 * be updated as divided by the warp size.
 */
template <typename Monoid, typename T, unsigned int WarpSize>
static inline T __device__ full_warp_combine(cg::thread_block_tile<WarpSize, cg::thread_block> tile,
                                             int const tile_id,
                                             Monoid binop,
                                             T value,
                                             int& output_offset,
                                             T* output)
{
  static_assert(
    WarpSize > 0 && WarpSize <= cudf::detail::warp_size && (WarpSize & (WarpSize - 1)) == 0,
    "Templated warp size must be at most as large as the physical warp size and a power of two");
  auto const rank{tile.thread_rank()};
  auto logical_rank{rank};
  auto logical_size{WarpSize};
#pragma unroll
  for (auto i = 1; i < WarpSize;) {
    value = binop(value, tile.shfl_down(value, i));
    logical_rank >>= 1;
    logical_size >>= 1;
    i <<= 1;
    if ((rank & (i - 1)) == 0) {
      output[output_offset + logical_size * tile_id + logical_rank] = value;
    }
    output_offset >>= 1;
  }
  return value;
}

/**
 * @brief Combine values across a warp using the binop advertised by `Monoid`.
 *
 * @tparam Monoid a device binary operator type with an associative binary operator and an identity
 * @tparam T the type of the values being combined.
 * @param tile Cooperative groups thread block tile with 32 threads
 * @param tile_id The rank of the warp in the tile
 * @param binop Instance of the monoidal binary operator
 * @param to_process The number of valid values this tile is processing.
 * @param value The value on this thread to combine
 * @param output_offset Offset into the output array where results should be written
 * @param output Output array
 *
 * @return the combined value, only valid on thread zero in the tile.
 *
 * @note Writes only occur if the logical output rank is a "valid"
 * one, indicated by the number of values that the combine should
 * process.
 *
 * See also `full_warp_combine`.
 */
template <typename Monoid, typename T>
static inline T __device__
partial_warp_combine(cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> tile,
                     int const tile_id,
                     Monoid binop,
                     int to_process,
                     T value,
                     int& output_offset,
                     T* output)
{
  auto const rank{tile.thread_rank()};
  unsigned int delta = 1;
  auto logical_rank{rank};
  auto logical_size{tile.size()};
  while (to_process > 1) {
    value = binop(value, tile.shfl_down(value, delta));
    delta <<= 1;
    logical_size >>= 1;
    logical_rank >>= 1;
    to_process >>= 1;
    if (logical_rank < to_process && (rank & (delta - 1)) == 0) {
      output[output_offset + logical_size * tile_id + logical_rank] = value;
    }
    output_offset >>= 1;
  }
  return value;
}

template <typename Monoid, typename T, unsigned int BlockSize>
__global__ void initialize_segment_tree(cudf::device_span<T> const input,
                                        cudf::device_span<T> segtree)
{
  constexpr auto warps_per_block{BlockSize / cudf::detail::warp_size};
  static_assert(warps_per_block <= cudf::detail::warp_size);
  static_assert(BlockSize % warp_size == 0);
  cg::grid_group grid = cg::this_grid();
  cg::thread_block tb = cg::this_thread_block();
  cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> warp_tile{
    cg::tiled_partition<cudf::detail::warp_size>(tb)};

  constexpr Monoid binop{};
  int const tile_id = warp_tile.meta_group_rank();
  int const lane_id = warp_tile.thread_rank();

  __shared__ T smem[cudf::detail::warp_size];
  for (cudf::thread_index_type i = blockIdx.x * blockDim.x; i < input.size();
       i += blockIdx.x * gridDim.x) {
    segtree[input.size() + i + threadIdx.x] = input[i + threadIdx.x];
  }
  grid.sync();

  //
  int prev_pow2                 = sizeof(long long) * CHAR_BIT - __clzll(segtree.size() - 1) - 1;
  cudf::thread_index_type start = static_cast<cudf::thread_index_type>(1) << prev_pow2;
  cudf::thread_index_type end   = segtree.size();
  if (start != input.size()) {
    // Input is not a power of two size, do a single level combination of the "remainder"
    // Since the largest the input can be is cudf::size_type::max
    // entries, and the segtree is 2*input.size() long, this always
    // fits in size_type.
    cudf::size_type N = end - start;
    for (cudf::thread_index_type i = blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
      // Each block reads blocksize elements
      cudf::thread_index_type const read_offset = start + i;
      cudf::size_type const write_offset        = read_offset >> 1;
      bool const is_valid                       = threadIdx.x + read_offset < end;
      T value = is_valid ? segtree[read_offset + threadIdx.x] : Monoid::identity();
      // After this shuffle down, only even lanes have valid data
      value = binop(value, __shfl_down_sync(0xffff'ffff, value, 1, warp_size));
      if (is_valid && lane_id % 2 == 0) {
        segtree[write_offset + warp_size / 2 * tile_id + lane_id / 2] = value;
      }
    }
    grid.sync();
    end = start;
    start >>= 1;
  }
  while (start > 1) {
    cudf::size_type const N = end - start;
    for (cudf::thread_index_type i = blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
      int const block_remainder = (N - i) % BlockSize;
      int const read_offset     = start + i;
      int write_offset          = read_offset >> 1;
      if (block_remainder == 0) {
        // Full block, use unrolled warp_combine
        T value = full_warp_combine(
          warp_tile, tile_id, binop, segtree[read_offset + threadIdx.x], write_offset, segtree);
        tb.sync();
        if (lane_id == 0) { smem[tile_id] = value; }
        tb.sync();
        auto combining_tile = cg::tiled_partition<BlockSize / cudf::detail::warp_size>(tb);
        if (combining_tile.meta_group_rank() == 0) {
          value = full_warp_combine(
            combining_tile, 0, binop, smem[combining_tile.thread_rank()], write_offset, segtree);
        }
      } else {
        // Remainder block. Could peel this as a postamble, but probably not worth it.
        int const to_process =
          cuda::std::max(cuda::std::min(cudf::detail::warp_size,
                                        block_remainder - tile_id * cudf::detail::warp_size),
                         0);
        T value = tb.thread_rank() < block_remainder ? segtree[read_offset + threadIdx.x]
                                                     : Monoid::identity();
        value =
          partial_warp_combine(warp_tile, tile_id, binop, to_process, value, write_offset, segtree);
        tb.sync();
        if (lane_id == 0) { smem[tile_id] = value; }
        tb.sync();
        if (tile_id == 0) {
          value = partial_warp_combine(warp_tile,
                                       tile_id,
                                       binop,
                                       block_remainder / cudf::detail::warp_size,
                                       smem[lane_id],
                                       write_offset,
                                       segtree);
        }
      }
      grid.sync();
      end   = start / (BlockSize / 2);
      start = start / BlockSize;
    }
  }
}

template <typename Monoid, typename T>
static rmm::device_uvector<T> make_segment_tree(cudf::column_view const& values,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  constexpr unsigned int blocksize{512};
  auto kernel = initialize_segment_tree<Monoid, T, blocksize>;
  rmm::device_uvector<T> segtree(static_cast<std::size_t>(values.size()) * 2, stream);
  cudf::device_span<T> d_segtree{segtree};
  cudf::device_span<T> d_values{values};
  void* kernel_args[] = {static_cast<void*>(&d_values), static_cast<void*>(&d_segtree)};
  dim3 block_dim(blocksize, 1, 1);
  cudaDeviceProp prop;
  int num_blocks_per_sm;
  CUDF_CUDA_TRY(cudaGetDeviceProperties(&prop, rmm::get_current_cuda_device().value()));
  CUDF_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel, blocksize, 0));
  // TODO device prop
  dim3 grid_dim(prop.multiProcessorCount * num_blocks_per_sm, 1, 1);

  CUDF_CUDA_TRY(cudaLaunchCooperativeKernel(
    (void*)kernel, grid_dim, block_dim, kernel_args, 0, stream.value()));
  return segtree;
}
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
