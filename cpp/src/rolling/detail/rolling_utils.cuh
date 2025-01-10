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

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_segmented_reduce.cuh>
#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/partition.h>

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
 * @note This is used for uniformity of interface between grouped and ungrouped iterator
 * construction.
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
 * @note This is used for uniformity of interface between grouped and ungrouped iterator
 * construction.
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
 * @note This is used for uniformity of interface between grouped and ungrouped iterator
 * construction.
 */
struct ungrouped_with_nulls : nulls_mixin {
  cudf::size_type num_rows_;
  cudf::size_type null_count_;

  ungrouped_with_nulls(cudf::size_type num_rows, cudf::size_type null_count, bool nulls_at_start)
    : num_rows_{num_rows}, null_count_{null_count}, nulls_mixin{nulls_at_start}
  {
  }

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
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @note This is used for uniformity of interface between grouped and ungrouped iterator
 * construction.
 */
struct grouped_with_nulls : nulls_mixin {
  cudf::size_type const* labels_;
  cudf::size_type const* offsets_;
  column_device_view const& orderby_;
  rmm::device_uvector<cudf::size_type> null_counts_;

  struct is_null_kernel {
    column_device_view const& orderby_;
    [[nodiscard]] __device__ cudf::size_type operator()(cudf::size_type i) const noexcept
    {
      return static_cast<cudf::size_type>(orderby_.is_null_nocheck(i));
    }
  };

  grouped_with_nulls(cudf::size_type const* labels,
                     cudf::size_type const* offsets,
                     std::size_t num_groups,
                     column_device_view const& orderby,
                     bool nulls_at_start,
                     rmm::cuda_stream_view stream)
    : labels_{labels},
      offsets_{offsets},
      orderby_{orderby},
      null_counts_{num_groups, stream},
      nulls_mixin{nulls_at_start}
  {
    std::size_t bytes{0};
    auto is_null_it =
      cudf::detail::make_counting_transform_iterator(cudf::size_type{0}, is_null_kernel{orderby});
    cub::DeviceSegmentedReduce::Sum(nullptr,
                                    bytes,
                                    is_null_it,
                                    null_counts_.begin(),
                                    num_groups,
                                    offsets,
                                    offsets + 1,
                                    stream.value());
    auto tmp = rmm::device_buffer(bytes, stream);
    cub::DeviceSegmentedReduce::Sum(tmp.data(),
                                    bytes,
                                    is_null_it,
                                    null_counts_.begin(),
                                    num_groups,
                                    offsets,
                                    offsets + 1,
                                    stream.value());
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

  [[discard]] __device__ cudf::size_type null_count(cudf::size_type label) const noexcept
  {
    return *(null_counts_.begin() + label);
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

/*
 * Select the appropriate ordering comparator for the window type.
 */
template <window_type Type>
struct op_impl {
  using op     = void;
  using rev_op = void;
};

template <>
struct op_impl<window_type::BOUNDED_CLOSED> {
  using op     = thrust::less<>;
  using rev_op = thrust::greater<>;
};
template <>
struct op_impl<window_type::BOUNDED_OPEN> {
  using op     = thrust::less_equal<>;
  using rev_op = thrust::greater_equal<>;
};
template <>
struct op_impl<window_type::CURRENT_ROW> {
  using op     = thrust::less<>;
  using rev_op = thrust::greater<>;
};

template <window_type Type, order Order>
using op_t = std::conditional_t<Order == order::ASCENDING,
                                typename op_impl<Type>::op,
                                typename op_impl<Type>::rev_op>;

/**
 * @brief Compute `x + y` saturating at the numeric bounds rather than
 * overflowing.
 *
 * @tparam T the type of the result and left operand.
 * @tparam V the type of the right operand.
 * @param x The left operand.
 * @param y The right operand.
 *
 * @returns x + y, saturated at the numeric limits for the type of
 * `x`, without overflowing or invoking undefined behaviour.
 *
 * @note If `T` is a numeric type we must have `std::is_same_v<T,
 * V>`. If `T` is a timestamp type, `V` must be a duration type and
 * `std::is_same_v<typename T::duration, V>`. Note in particular that the
 * usual integral promotion rules are not applied.
 */
template <typename T, typename V>
[[nodiscard]] __host__ __device__ constexpr T add_sat(T x, V y) noexcept
{
  if constexpr (cudf::is_timestamp_t<T>()) {
    static_assert(cudf::is_duration_t<V>(), "Can only add durations to timestamps");
    static_assert(cuda::std::is_same_v<typename T::duration, V>,
                  "Duration resolution must match timestamp resolution");
    return T{add_sat(x.time_since_epoch(), y)};
  } else if constexpr (cudf::is_duration_t<T>()) {
    static_assert(cuda::std::is_same_v<T, V>, "Cannot add mismatching types");
    return T{add_sat(x.count(), y.count())};
  } else if constexpr (cudf::is_fixed_point<T>()) {
    // Requirement, not checked, x and y have the same scale.
    static_assert(cudf::is_fixed_point<V>(), "Must add fixed point to fixed point.");
    using Rep = typename T::rep;
    return T{numeric::scaled_integer<Rep>{add_sat(x.value(), y.value()), x.scale()}};
  } else {
    static_assert(cuda::std::is_same_v<T, V>, "Cannot add mismatching types");
    if constexpr (cuda::std::is_signed_v<T>) {
      using U  = cuda::std::make_unsigned_t<T>;
      U ux     = static_cast<U>(x);
      U uy     = static_cast<U>(y);
      U result = ux + uy;
      ux       = (ux >> cuda::std::numeric_limits<T>::digits) +
           static_cast<U>(cuda::std::numeric_limits<T>::max());
      // Note: this cast is implementation defined (until C++20) but all
      // the platforms we care about do the twos-complement thing.
      return static_cast<T>((ux ^ uy) | ~(uy ^ result)) >= 0 ? ux : result;
    } else if constexpr (cuda::std::is_unsigned_v<T>) {
      T result = x + y;
      // Only way we can overflow is in the positive direction
      // in which case result will be less than both of x and y.
      // To saturate, we bit-or with (T)-1 in this case
      return result | (-static_cast<T>(result < x));
    } else if constexpr (cudf::is_duration_t<T>()) {
      return T{add_sat(x.count(), y.count())};
    } else if constexpr (cuda::std::is_floating_point_v<T>) {
      // Question: should adding a finite y to a finite x saturate at
      // numeric_limits::lowest()/max()?
      return x + y;
    } else {
      static_assert(std::integral_constant<T, false>(),
                    "Saturating addition only for signed and unsigned integers, floats, "
                    "durations, or timestamps.");
    }
  }
}

template <window_type WindowType, direction Direction, cudf::order Order>
struct range_window_clamper {
  template <typename Grouping, typename OrderbyT, typename DeltaT>
  struct distance_kernel {
    Grouping groups;
    // Delta from current row that defines the interval endpoint.
    // The endpoint is always current_row_value + row_delta, saturated
    // at the datatype bounds.
    // Note that these are always value-wise, so if you have a
    // descending ordered column you often want row_delta to be
    // negative for the following window.
    DeltaT const* row_delta;
    column_device_view::const_iterator<OrderbyT> begin;
    column_device_view::const_iterator<OrderbyT> end;

    [[nodiscard]] __device__ size_type operator()(size_type i) const
    {
      using Comp             = op_t<WindowType, Order>;
      auto const label       = groups.group_label(i);
      auto const group_start = groups.group_start(label);
      auto const group_end   = groups.group_end(label);
      auto const null_count  = groups.null_count(label);
      auto const start       = groups.non_null_start(group_start, group_end, label);
      auto const end         = groups.non_null_end(group_start, group_end, label);
      if constexpr (Direction == direction::PRECEDING) {
        if constexpr (WindowType == window_type::UNBOUNDED) { return i - group_start + 1; }
        if (groups.is_null(i, null_count)) {
          return i - groups.null_start(group_start, group_end, null_count) + 1;
        }
        if constexpr (WindowType == window_type::CURRENT_ROW) {
          return 1 +
                 thrust::distance(
                   thrust::lower_bound(thrust::seq, begin + start, begin + i, *(begin + i), Comp{}),
                   begin + i);
        } else if constexpr (WindowType != window_type::UNBOUNDED) {
          return 1 + thrust::distance(thrust::lower_bound(thrust::seq,
                                                          begin + start,
                                                          begin + end,
                                                          add_sat(*(begin + i), *row_delta),
                                                          Comp{}),
                                      begin + i);
        }
      } else {
        if constexpr (WindowType == window_type::UNBOUNDED) { return group_end - i - 1; }
        if (groups.is_null(i, null_count)) {
          return groups.null_end(group_start, group_end, null_count) - i - 1;
        }
        if constexpr (WindowType == window_type::CURRENT_ROW) {
          return thrust::distance(
                   begin + i,
                   thrust::upper_bound(thrust::seq, begin + i, begin + end, *(begin + i), Comp{})) -
                 1;
        } else if constexpr (WindowType != window_type::UNBOUNDED) {
          return thrust::distance(begin + i,
                                  thrust::upper_bound(thrust::seq,
                                                      begin + start,
                                                      begin + end,
                                                      add_sat(*(begin + i), *row_delta),
                                                      Comp{})) -
                 1;
        }
      }
    }
  };

  template <typename OrderbyT, typename DeltaT>
  [[nodiscard]] std::unique_ptr<column> window_bounds(table_view const& group_keys,
                                                      column_view const& orderby,
                                                      bool nulls_at_start,
                                                      scalar const* row_delta,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) const
  {
    using ScalarT = cudf::scalar_type_t<DeltaT>;
    auto result   = make_numeric_column(
      data_type(type_to_id<size_type>()), orderby.size(), mask_state::UNALLOCATED, stream, mr);
    auto const d_orderby   = column_device_view::create(orderby, stream);
    auto d_begin           = d_orderby->begin<OrderbyT>();
    auto d_end             = d_orderby->end<OrderbyT>();
    auto const d_row_delta = dynamic_cast<ScalarT const&>(*row_delta).data();
    // auto copy_n            = [&](auto&& kernel) {
    //   thrust::copy_n(rmm::exec_policy_nosync(stream),
    //                  cudf::detail::make_counting_transform_iterator(0, kernel),
    //                  orderby.size(),
    //                  result->mutable_view().begin<size_type>());
    // };
    if (group_keys.num_columns() == 0) {
      if (orderby.has_nulls()) {
        thrust::copy_n(
          rmm::exec_policy_nosync(stream),
          cudf::detail::make_counting_transform_iterator(
            0,
            distance_kernel<ungrouped_with_nulls, OrderbyT, DeltaT>{
              ungrouped_with_nulls{orderby.size(), orderby.null_count(), nulls_at_start},
              d_row_delta,
              d_begin,
              d_end}),
          orderby.size(),
          result->mutable_view().begin<size_type>());
      } else {
        CUDF_FAIL("BARF");
        // copy_n(distance_kernel<ungrouped, OrderbyT, DeltaT>{
        //   ungrouped{orderby.size()}, d_row_delta, d_begin, d_end});
      }
    } else {
      CUDF_FAIL("BARF");
      // CUDF_EXPECTS(orderby.size() == group_keys.num_rows(),
      //              "group_keys and orderby must have the same number of rows",
      //              std::invalid_argument);
      // using sort_helper = cudf::groupby::detail::sort::sort_groupby_helper;
      // sort_helper helper{group_keys, null_policy::INCLUDE, sorted::YES, {}};
      // auto const& labels  = helper.group_labels(stream);
      // auto const& offsets = helper.group_offsets(stream);
      // if (orderby.has_nulls()) {
      //   copy_n(distance_kernel<grouped_with_nulls, OrderbyT, DeltaT>{
      //     grouped_with_nulls{
      //       labels.data(), offsets.data(), labels.size(), *d_orderby, nulls_at_start, stream},
      //     d_row_delta,
      //     d_begin,
      //     d_end});
      // } else {
      //   copy_n(distance_kernel<grouped, OrderbyT, DeltaT>{
      //     grouped{labels.data(), offsets.data()}, d_row_delta, d_begin, d_end});
      // }
    }
    return result;
  }

  // template <typename OrderbyT, CUDF_ENABLE_IF(cudf::is_timestamp<OrderbyT>())>
  // [[nodiscard]] std::unique_ptr<column> operator()(table_view const& group_keys,
  //                                                  column_view const& orderby,
  //                                                  bool nulls_at_start,
  //                                                  scalar const* row_delta,
  //                                                  rmm::cuda_stream_view stream,
  //                                                  rmm::device_async_resource_ref mr) const
  // {
  //   using DiffT = typename OrderbyT::duration;
  //   CUDF_EXPECTS(cudf::is_duration(row_delta->type()),
  //                "Row delta must be a duration type.",
  //                cudf::data_type_error);
  //   CUDF_EXPECTS(row_delta->type().id() == type_to_id<DiffT>(),
  //                "Row delta must have same the resolution as orderby.",
  //                cudf::data_type_error);
  //   return window_bounds<OrderbyT, DiffT>(
  //     group_keys, orderby, nulls_at_start, row_delta, stream, mr);
  // }

  template <typename OrderbyT,
            CUDF_ENABLE_IF((cudf::is_index_type<OrderbyT>() && !cudf::is_unsigned<OrderbyT>()))>
  [[nodiscard]] std::unique_ptr<column> operator()(table_view const& group_keys,
                                                   column_view const& orderby,
                                                   bool nulls_at_start,
                                                   scalar const* row_delta,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr) const
  {
    using DiffT = OrderbyT;
    CUDF_EXPECTS(cudf::have_same_types(orderby, *row_delta),
                 "Orderby column and row_delta must have the same type.",
                 cudf::data_type_error);
    return window_bounds<OrderbyT, DiffT>(
      group_keys, orderby, nulls_at_start, row_delta, stream, mr);
  }

  template <typename OrderbyT,
            CUDF_ENABLE_IF(!((cudf::is_index_type<OrderbyT>() && !cudf::is_unsigned<OrderbyT>())))>
  std::unique_ptr<column> operator()(table_view const&,
                                     column_view const&,
                                     bool,
                                     scalar const*,
                                     rmm::cuda_stream_view,
                                     rmm::device_async_resource_ref) const
  {
    CUDF_FAIL("Unsupported rolling window type.", cudf::data_type_error);
  }
};

/**
 * @brief Make a column representing the window offsets for a range-based window
 *
 * @tparam Direction Is this a preceding window or a following one.
 *
 * @param group_keys Table defining grouping of the windows. May be empty. If non-empty, group keys
 * must be sorted.
 * @param orderby Column use to define window ranges. If @p group_keys is empty, must be sorted. If
 * @p group_keys is non-empty, must be sorted within each group. As well as being sorted, must be
 * sorted consistently with the @p order and @p null_order parameters.
 * @param order The sort order of the @p orderby column.
 * @param null_order The sort order of nulls in the @p orderby column.
 * @param row_delta Pointer to scalar providing the delta for the window range. May be null, but
 * only if the @p window_type is @p CURRENT_ROW or @p UNBOUNDED. Note that @p row_delta is always
 * added to the current row value.
 * @param window_type The type of window we are computing bounds for.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used for allocations.
 */
template <direction Direction>
[[nodiscard]] std::unique_ptr<column> inline make_range_window_bounds(
  table_view const& group_keys,
  column_view const& orderby,
  order order,
  null_order null_order,
  scalar const* row_delta,
  window_type window,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(
    window == window_type::UNBOUNDED || window == window_type::CURRENT_ROW || row_delta != nullptr,
    "For bounded windows, row_delta must be non-null.");
  bool const nulls_at_start = (order == order::ASCENDING && null_order == null_order::BEFORE) ||
                              (order == order::DESCENDING && null_order == null_order::AFTER);

  if (window == window_type::UNBOUNDED && order == order::ASCENDING) {
    return type_dispatcher(
      orderby.type(),
      range_window_clamper<window_type::UNBOUNDED, Direction, order::ASCENDING>{},
      group_keys,
      orderby,
      nulls_at_start,
      row_delta,
      stream,
      mr);
  } else if (window == window_type::UNBOUNDED && order == order::DESCENDING) {
    return type_dispatcher(
      orderby.type(),
      range_window_clamper<window_type::UNBOUNDED, Direction, order::DESCENDING>{},
      group_keys,
      orderby,
      nulls_at_start,
      row_delta,
      stream,
      mr);
  } else if (window == window_type::CURRENT_ROW && order == order::ASCENDING) {
    return type_dispatcher(
      orderby.type(),
      range_window_clamper<window_type::CURRENT_ROW, Direction, order::ASCENDING>{},
      group_keys,
      orderby,
      nulls_at_start,
      row_delta,
      stream,
      mr);
  } else if (window == window_type::CURRENT_ROW && order == order::DESCENDING) {
    return type_dispatcher(
      orderby.type(),
      range_window_clamper<window_type::CURRENT_ROW, Direction, order::DESCENDING>{},
      group_keys,
      orderby,
      nulls_at_start,
      row_delta,
      stream,
      mr);
  } else if (window == window_type::BOUNDED_OPEN && order == order::ASCENDING) {
    return type_dispatcher(
      orderby.type(),
      range_window_clamper<window_type::BOUNDED_OPEN, Direction, order::ASCENDING>{},
      group_keys,
      orderby,
      nulls_at_start,
      row_delta,
      stream,
      mr);
  } else if (window == window_type::BOUNDED_OPEN && order == order::DESCENDING) {
    return type_dispatcher(
      orderby.type(),
      range_window_clamper<window_type::BOUNDED_OPEN, Direction, order::DESCENDING>{},
      group_keys,
      orderby,
      nulls_at_start,
      row_delta,
      stream,
      mr);
  } else if (window == window_type::CURRENT_ROW && order == order::ASCENDING) {
    return type_dispatcher(
      orderby.type(),
      range_window_clamper<window_type::CURRENT_ROW, Direction, order::ASCENDING>{},
      group_keys,
      orderby,
      nulls_at_start,
      row_delta,
      stream,
      mr);
  } else if (window == window_type::CURRENT_ROW && order == order::DESCENDING) {
    return type_dispatcher(
      orderby.type(),
      range_window_clamper<window_type::CURRENT_ROW, Direction, order::DESCENDING>{},
      group_keys,
      orderby,
      nulls_at_start,
      row_delta,
      stream,
      mr);
  } else {
    CUDF_FAIL("Unsupported window type and sorted order combination for range window bounds");
  }
}

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
