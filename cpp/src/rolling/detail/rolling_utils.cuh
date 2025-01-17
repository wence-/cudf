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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>

#include <cstddef>

namespace cudf {

namespace detail::rolling {

struct no_null_mixin {
  [[nodiscard]] __device__ constexpr cudf::size_type null_count(
    cudf::size_type label) const noexcept
  {
    return 0;
  }

  [[nodiscard]] __device__ constexpr bool is_null(cudf::size_type i,
                                                  cudf::size_type null_count) const noexcept
  {
    return false;
  }

  [[nodiscard]] __device__ constexpr bool null_start(cudf::size_type start,
                                                     cudf::size_type end,
                                                     cudf::size_type null_count) const noexcept
  {
    return start;
  }

  [[nodiscard]] __device__ constexpr bool null_end(cudf::size_type start,
                                                   cudf::size_type end,
                                                   cudf::size_type null_count) const noexcept
  {
    return start;
  }

  [[nodiscard]] __device__ constexpr bool non_null_start(cudf::size_type start,
                                                         cudf::size_type end,
                                                         cudf::size_type null_count) const noexcept
  {
    return start;
  }

  [[nodiscard]] __device__ constexpr bool non_null_end(cudf::size_type start,
                                                       cudf::size_type end,
                                                       cudf::size_type null_count) const noexcept
  {
    return end;
  }
};

/**
 * @brief A group descriptor for an ungrouped rolling window.
 *
 * @param num_rows The number of rows to be rolled over.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct ungrouped : no_null_mixin {
  cudf::size_type num_rows_;

  ungrouped(cudf::size_type num_rows) : num_rows_{num_rows}, no_null_mixin{} {}

  [[nodiscard]] __device__ constexpr cudf::size_type group_label(cudf::size_type) const noexcept
  {
    return 0;
  }

  [[nodiscard]] __device__ constexpr cudf::size_type group_start(cudf::size_type) const noexcept
  {
    return 0;
  }

  [[nodiscard]] __device__ constexpr cudf::size_type group_end(cudf::size_type) const noexcept
  {
    return num_rows_;
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
struct grouped : no_null_mixin {
  cudf::size_type const* labels_;
  cudf::size_type const* offsets_;

  grouped(cudf::size_type const* labels, cudf::size_type const* offsets)
    : labels_{labels}, offsets_{offsets}, no_null_mixin{}
  {
  }

  [[nodiscard]] __device__ constexpr cudf::size_type group_label(cudf::size_type i) const noexcept
  {
    return labels_[i];
  }

  [[nodiscard]] __device__ constexpr cudf::size_type group_start(
    cudf::size_type label) const noexcept
  {
    return offsets_[label];
  }

  [[nodiscard]] __device__ constexpr cudf::size_type group_end(cudf::size_type label) const noexcept
  {
    return offsets_[label + 1];
  }
};

struct nulls_mixin {
  bool nulls_at_start_;

  [[nodiscard]] __device__ constexpr bool null_start(cudf::size_type start,
                                                     cudf::size_type end,
                                                     cudf::size_type null_count) const noexcept
  {
    return nulls_at_start_ ? start : end - null_count;
  }

  [[nodiscard]] __device__ constexpr bool null_end(cudf::size_type start,
                                                   cudf::size_type end,
                                                   cudf::size_type null_count) const noexcept
  {
    return nulls_at_start_ ? start + null_count : end;
  }

  [[nodiscard]] __device__ constexpr bool non_null_start(cudf::size_type start,
                                                         cudf::size_type end,
                                                         cudf::size_type null_count) const noexcept
  {
    return nulls_at_start_ ? start + null_count : start;
  }

  [[nodiscard]] __device__ constexpr bool non_null_end(cudf::size_type start,
                                                       cudf::size_type end,
                                                       cudf::size_type null_count) const noexcept
  {
    return nulls_at_start_ ? end : end - null_count;
  }
};

/**
 * @brief A group descriptor for an ungrouped rolling window.
 *
 * @param num_rows The number of rows to be rolled over.
 * @param nulls_at_start Are the nulls at the start or end?
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct ungrouped_with_nulls : nulls_mixin {
  cudf::size_type num_rows_;
  cudf::size_type null_count_;

  [[nodiscard]] __device__ constexpr cudf::size_type group_label(cudf::size_type i) const noexcept
  {
    return 0;
  }
  [[nodiscard]] __device__ constexpr cudf::size_type group_start(
    cudf::size_type label) const noexcept
  {
    return 0;
  }
  [[nodiscard]] __device__ constexpr cudf::size_type group_end(cudf::size_type label) const noexcept
  {
    return num_rows_;
  }

  [[nodiscard]] __device__ constexpr cudf::size_type null_count(
    cudf::size_type label) const noexcept
  {
    return null_count_;
  }

  [[nodiscard]] __device__ constexpr bool is_null(cudf::size_type i,
                                                  cudf::size_type null_count) const noexcept
  {
    return (nulls_at_start_ && i < null_count) || (!nulls_at_start_ && i >= num_rows_ - null_count);
  }
};

/**
 * @brief A group descriptor for a grouped rolling window with nulls
 *
 * @param labels The group labels, mapping from input rows to group.
 * @param offsets The group offsets providing the endpoints of each group.
 * @param num_groups The number of groups.
 * @param orderby The orderby columns, sorted groupwise.
 * @param nulls_at_start Are the nulls at the start of each group?
 * @param stream CUDA stream used for device memory operations and kernel
 * launches.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped
 * iterator construction.
 */
struct grouped_with_nulls : nulls_mixin {
  cudf::size_type const* labels_;
  cudf::size_type const* offsets_;
  cudf::size_type const* null_counts_;
  column_device_view const orderby_;

  struct is_null_kernel {
    column_device_view const orderby_;
    [[nodiscard]] __device__ cudf::size_type operator()(cudf::size_type i) const noexcept
    {
      return static_cast<cudf::size_type>(orderby_.is_null_nocheck(i));
    }
  };

  [[nodiscard]] static rmm::device_uvector<cudf::size_type> nulls_per_group(
    std::size_t num_groups,
    cudf::size_type const* offsets,
    column_device_view const orderby,
    rmm::cuda_stream_view stream)
  {
    std::size_t bytes{0};
    auto is_null_it =
      cudf::detail::make_counting_transform_iterator(cudf::size_type{0}, is_null_kernel{orderby});
    rmm::device_uvector<cudf::size_type> null_counts{num_groups, stream};
    cub::DeviceSegmentedReduce::Sum(nullptr,
                                    bytes,
                                    is_null_it,
                                    null_counts.begin(),
                                    num_groups,
                                    offsets,
                                    offsets + 1,
                                    stream.value());
    auto tmp = rmm::device_buffer(bytes, stream);
    cub::DeviceSegmentedReduce::Sum(tmp.data(),
                                    bytes,
                                    is_null_it,
                                    null_counts.begin(),
                                    num_groups,
                                    offsets,
                                    offsets + 1,
                                    stream.value());
    return null_counts;
  }

  [[nodiscard]] __device__ constexpr cudf::size_type group_label(cudf::size_type i) const noexcept
  {
    return labels_[i];
  }
  [[nodiscard]] __device__ constexpr cudf::size_type group_start(
    cudf::size_type label) const noexcept
  {
    return offsets_[label];
  }
  [[nodiscard]] __device__ constexpr cudf::size_type group_end(cudf::size_type label) const noexcept
  {
    return offsets_[label + 1];
  }

  [[nodiscard]] __device__ cudf::size_type null_count(cudf::size_type label) const noexcept
  {
    return null_counts_[label];
  }

  [[nodiscard]] __device__ bool is_null(cudf::size_type i,
                                        cudf::size_type null_count) const noexcept
  {
    return orderby_.is_null_nocheck(i);
  }
};

enum class direction : bool {
  PRECEDING,
  FOLLOWING,
};

enum class window_type : std::int8_t {
  BOUNDED_OPEN,
  BOUNDED_CLOSED,
  UNBOUNDED,
  CURRENT_ROW,
};

template <typename Grouping, direction Direction>
struct fixed_window_clamper {
  Grouping groups;
  cudf::size_type delta;
  [[nodiscard]] __device__ constexpr cudf::size_type operator()(cudf::size_type i) const
  {
    auto label = groups.group_label(i);
    auto start = groups.group_start(label);
    auto end   = groups.group_end(label);
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
}  // namespace cudf
