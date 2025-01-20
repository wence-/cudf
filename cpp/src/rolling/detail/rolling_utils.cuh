/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#pragma once

#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>
#include <cuda/std/type_traits>

#include <cstddef>

namespace CUDF_EXPORT cudf {

namespace detail::rolling {

/**
 * @brief A group descriptor for an ungrouped rolling window.
 *
 * @param num_rows The number of rows to be rolled over.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct ungrouped {
  cudf::size_type num_rows_;

  static constexpr bool has_nulls{false};
  /**
   * @brief Return information about the current row.
   *
   * @param i The row
   * @returns Tuple of `(null_count, group_start, group_end, null_start,
   * null_end, non_null_start, non_null_end)`
   */
  [[nodiscard]] __device__ constexpr cuda::std::
    tuple<size_type, size_type, size_type, size_type, size_type, size_type, size_type>
    row_info(size_type i) const noexcept
  {
    return {0, 0, num_rows_, 0, 0, 0, num_rows_};
  }
};

/**
 * @brief A group descriptor for a grouped rolling window.
 *
 * @param labels The group labels, mapping from input rows to group.
 * @param offsets The group offsets providing the endpoints of each group.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct grouped {
  cudf::size_type const* labels_;
  cudf::size_type const* offsets_;

  static constexpr bool has_nulls{false};
  /**
   * @copydoc ungrouped::row_info
   */
  [[nodiscard]] __device__ constexpr cuda::std::
    tuple<size_type, size_type, size_type, size_type, size_type, size_type, size_type>
    row_info(size_type i) const noexcept
  {
    auto const label       = labels_[i];
    auto const group_start = offsets_[label];
    auto const group_end   = offsets_[label + 1];
    return {0, group_start, group_end, group_start, group_start, group_start, group_end};
  }
};

enum class direction : bool {
  PRECEDING,
  FOLLOWING,
};

template <typename Grouping, direction Direction>
struct fixed_window_clamper {
  Grouping groups;
  cudf::size_type delta;
  static_assert(cuda::std::is_same_v<Grouping, ungrouped> ||
                  cuda::std::is_same_v<Grouping, grouped>,
                "Invalid grouping descriptor");

  [[nodiscard]] __device__ constexpr cudf::size_type operator()(cudf::size_type i) const
  {
    auto const info  = groups.row_info(i);
    auto const start = cuda::std::get<1>(info);
    auto const end   = cuda::std::get<2>(info);
    if constexpr (Direction == direction::PRECEDING) {
      return cuda::std::min(i + 1 - start, cuda::std::max(delta, i + 1 - end));
    } else {
      return cuda::std::max(start - i - 1, cuda::std::min(delta, end - i - 1));
    }
  }
};

/**
 * @brief Construct a clamped counting iterator for a row-based window offset
 *
 * @tparam Direction the direction of the window `PRECEDING` or `FOLLOWING`.
 * @tparam Grouping the group specification.
 * @param delta the window offset.
 * @param grouper the grouping object.
 *
 * @return An iterator suitable for passing to `cudf::detail::rolling_window`
 */
template <direction Direction, typename Grouping>
[[nodiscard]] auto inline make_clamped_window_iterator(cudf::size_type delta, Grouping grouper)
{
  return cudf::detail::make_counting_transform_iterator(
    cudf::size_type{0}, fixed_window_clamper<Grouping, Direction>{grouper, delta});
}
}  // namespace detail::rolling
}  // namespace CUDF_EXPORT cudf
