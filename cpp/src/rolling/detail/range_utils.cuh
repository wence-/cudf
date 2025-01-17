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

#include "rolling_utils.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
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

namespace cudf {

namespace detail::rolling {

/**
 * @brief Select the appropriate ordering comparator for the window type.
 * @tparam Type The type of the window.
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

/**
 * @brief Select the appropriate ordering comparator for the window type.
 * @tparam Type The type of the window
 * @tparam Order The sort order of the column used to define the windows.
 */
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
 * usual integral promotion rules are not applied. If `T` is a fixed
 * point type, then `V` must be the representation type of `T`, and it
 * is required that `x` and `y` have the same scale.
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
    using Rep = typename T::rep;
    // Requirement, not checked, x and y have the same scale.
    static_assert(cuda::std::is_same_v<Rep, V>, "Must add rep type of fixed point to fixed point.");
    return T{numeric::scaled_integer<Rep>{add_sat(x.value(), y), x.scale()}};
  } else {
    static_assert(cuda::std::is_same_v<T, V>, "Cannot add mismatching types");

    if constexpr (cuda::std::is_floating_point_v<T>) {
      // Question: should adding a finite y to a finite x saturate at
      // numeric_limits::lowest()/max()?
      return x + y;
    } else if constexpr (cuda::std::is_signed_v<T>) {
      return x + y;
      // using U  = cuda::std::make_unsigned_t<T>;
      // U ux     = static_cast<U>(x);
      // U uy     = static_cast<U>(y);
      // U result = ux + uy;
      // ux       = (ux >> cuda::std::numeric_limits<T>::digits) +
      //      static_cast<U>(cuda::std::numeric_limits<T>::max());
      // // Note: this cast is implementation defined (until C++20) but all
      // // the platforms we care about do the twos-complement thing.
      // return static_cast<T>((ux ^ uy) | ~(uy ^ result)) >= 0 ? ux : result;
    } else if constexpr (cuda::std::is_unsigned_v<T>) {
      T result = x + y;
      // Only way we can overflow is in the positive direction
      // in which case result will be less than both of x and y.
      // To saturate, we bit-or with (T)-1 in this case
      return result | (-static_cast<T>(result < x));
    } else if constexpr (cudf::is_duration_t<T>()) {
      return T{add_sat(x.count(), y.count())};
    } else {
      static_assert(std::integral_constant<T, false>(),
                    "Saturating addition only for signed and unsigned integers, floats, "
                    "durations, fixed point, or timestamps.");
    }
  }
}

/**
 * @brief Functor to dispatch computation of clamped range-based
 * rolling window bounds.
 *
 * @tparam WindowType The type of window being computed
 * @tparam Direction The direction (preceding or following) of the
 * window being computed.
 * @tparam Order The sort order of the orderby column defining the
 * window.
 */
template <window_type WindowType, direction Direction, cudf::order Order>
struct range_window_clamper {
  /**
   * @brief Functor to compute distance from a given row to the edge
   * of the window.
   *
   * @tparam Grouping Object defining how the orderby column is
   * grouped.
   * @tparam OrderbyT Type of elements in the orderby columns.
   * @tparam DeltaT Type of the elements in the scalar delta (returned
   * by scalar.data()).
   * @param groups The grouping object.
   * @param row_delta Pointer to row delta on device.
   * @param begin Iterator to begin of orderby column on device.
   * @param end Iterator to end of orderby column on device.
   *
   * @note If the window is a bounded one, then the endpoint of the
   * window is always computed by ADDING the given @p row_delta to the
   * current row value (saturating at the data type bounds).
   */
  template <typename Grouping, typename OrderbyT, typename DeltaT>
  struct distance_kernel {
    Grouping groups;
    // Delta from current row that defines the interval endpoint.
    // The endpoint is always current_row_value + row_delta, saturated
    // at the datatype bounds.
    // Note that these are always value-wise, so if you have a
    // descending ordered column you often want row_delta to be
    // negative for the following window.
    // This pointer may be null for UNBOUNDED and CURRENT_ROW windows.
    DeltaT const* row_delta;
    column_device_view::const_iterator<OrderbyT> begin;
    column_device_view::const_iterator<OrderbyT> end;

    /**
     * @brief Compute the row defining the endpoint of the current
     *  window
     *
     * @param i The current row index.
     * @return Offset to the current row's window endpoint.
     */
    [[nodiscard]] __device__ size_type operator()(size_type i) const
    {
      using Comp             = op_t<WindowType, Order>;
      auto const label       = groups.group_label(i);
      auto const group_start = groups.group_start(label);
      auto const group_end   = groups.group_end(label);
      auto const null_count  = groups.null_count(label);
      auto const start       = groups.non_null_start(group_start, group_end, null_count);
      auto const end         = groups.non_null_end(group_start, group_end, null_count);
      if constexpr (Direction == direction::PRECEDING) {
        if constexpr (WindowType == window_type::UNBOUNDED) { return 0; }  // i - group_start + 1; }
        if (groups.is_null(i, null_count)) {
          return 0;  // i - groups.null_start(group_start, group_end, null_count) + 1;
        }
        if constexpr (WindowType == window_type::CURRENT_ROW) {
          return 0;  // 1 +
                     // thrust::distance(
                     //   thrust::lower_bound(thrust::seq, begin + start, begin + i, *(begin + i),
                     //   Comp{}), begin + i);
        } else if constexpr (WindowType != window_type::UNBOUNDED) {
          return begin + i -
                 thrust::lower_bound(thrust::seq,
                                     begin + start,
                                     begin + end,
                                     add_sat(*(begin + i), *row_delta),
                                     Comp{}) +
                 1;
        } else {
          CUDF_UNREACHABLE("Unexpected WindowType");
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
        } else {
          CUDF_UNREACHABLE("Unexpected WindowType");
        }
      }
    }
  };

  /**
   * @brief Compute the window bounds (possibly grouped) for an
   * orderby column.
   *
   * @tparam OrderbyT element type of the orderby column (dispatched
   * on)
   * @tparam ScalarT Concrete scalar type of the scalar row delta
   * @tparam
   */
  template <typename OrderbyT, typename ScalarT>
  [[nodiscard]] std::unique_ptr<column> window_bounds(
    column_view const& orderby,
    std::optional<std::pair<rmm::device_uvector<cudf::size_type> const&,
                            rmm::device_uvector<cudf::size_type> const&>> const& grouping,
    bool nulls_at_start,
    ScalarT const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    auto result = make_numeric_column(
      data_type(type_to_id<size_type>()), orderby.size(), mask_state::UNALLOCATED, stream, mr);
    auto d_orderby          = column_device_view::create(orderby, stream);
    auto d_begin            = d_orderby->begin<OrderbyT>();
    auto d_end              = d_orderby->end<OrderbyT>();
    auto const* d_row_delta = row_delta ? row_delta->data() : nullptr;
    using DeltaT = cuda::std::remove_cv_t<cuda::std::remove_pointer_t<decltype(d_row_delta)>>;
    auto copy_n  = [&](auto&& kernel) {
      thrust::copy_n(rmm::exec_policy_nosync(stream),
                     cudf::detail::make_counting_transform_iterator(0, kernel),
                     orderby.size(),
                     result->mutable_view().begin<size_type>());
    };
    if (!grouping.has_value()) {
      if (orderby.has_nulls()) {
        copy_n(distance_kernel<ungrouped_with_nulls, OrderbyT, DeltaT>{
          ungrouped_with_nulls{nulls_at_start, orderby.size(), orderby.null_count()},
          d_row_delta,
          d_begin,
          d_end});
      } else {
        copy_n(distance_kernel<ungrouped, OrderbyT, DeltaT>{
          ungrouped{orderby.size()}, d_row_delta, d_begin, d_end});
      }
    } else {
      auto [labels, offsets] = grouping.value();
      if (orderby.has_nulls()) {
        auto nulls_per_group =
          grouped_with_nulls::nulls_per_group(labels.size(), offsets.data(), *d_orderby, stream);
        copy_n(distance_kernel<grouped_with_nulls, OrderbyT, DeltaT>{grouped_with_nulls{
                                                                       nulls_at_start,
                                                                       labels.data(),
                                                                       offsets.data(),
                                                                       nulls_per_group.data(),
                                                                       *d_orderby,
                                                                     },
                                                                     d_row_delta,
                                                                     d_begin,
                                                                     d_end});
      } else {
        copy_n(distance_kernel<grouped, OrderbyT, DeltaT>{
          grouped{labels.data(), offsets.data()}, d_row_delta, d_begin, d_end});
      }
    }
    stream.synchronize();
    return result;
  }

  template <typename T>
  static constexpr bool is_supported()
  {
    return cudf::is_timestamp<T>() || cudf::is_numeric_not_bool<T>() || cudf::is_fixed_point<T>() ||
           cuda::std::is_same_v<T, cudf::string_view>;
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(cudf::is_timestamp<OrderbyT>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    std::optional<std::pair<rmm::device_uvector<cudf::size_type> const&,
                            rmm::device_uvector<cudf::size_type> const&>> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    using ScalarT = cudf::scalar_type_t<typename OrderbyT::duration>;
    CUDF_EXPECTS(!row_delta || cudf::is_duration(row_delta->type()),
                 "Row delta must be a duration type.",
                 cudf::data_type_error);
    CUDF_EXPECTS(!row_delta || row_delta->type().id() == type_to_id<typename OrderbyT::duration>(),
                 "Row delta must have same the resolution as orderby.",
                 cudf::data_type_error);
    return window_bounds<OrderbyT, ScalarT>(
      orderby, grouping, nulls_at_start, dynamic_cast<ScalarT const*>(row_delta), stream, mr);
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(cudf::is_fixed_point<OrderbyT>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    std::optional<std::pair<rmm::device_uvector<cudf::size_type> const&,
                            rmm::device_uvector<cudf::size_type> const&>> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    using ScalarT = cudf::scalar_type_t<OrderbyT>;
    CUDF_EXPECTS(!row_delta || cudf::have_same_types(orderby, *row_delta),
                 "Orderby column and row_delta must both be fixed point.",
                 cudf::data_type_error);
    CUDF_EXPECTS(!row_delta || row_delta->type().scale() == orderby.type().scale(),
                 "Orderby column and row_delta must have same fixed point scale.",
                 cudf::data_type_error);
    return window_bounds<OrderbyT, ScalarT>(
      orderby, grouping, nulls_at_start, dynamic_cast<ScalarT const*>(row_delta), stream, mr);
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(cudf::is_numeric_not_bool<OrderbyT>())>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    std::optional<std::pair<rmm::device_uvector<cudf::size_type> const&,
                            rmm::device_uvector<cudf::size_type> const&>> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    using ScalarT = cudf::scalar_type_t<OrderbyT>;
    CUDF_EXPECTS(!row_delta || cudf::have_same_types(orderby, *row_delta),
                 "Orderby column and row_delta must have the same type.",
                 cudf::data_type_error);
    return window_bounds<OrderbyT, ScalarT>(
      orderby, grouping, nulls_at_start, dynamic_cast<ScalarT const*>(row_delta), stream, mr);
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(cuda::std::is_same_v<OrderbyT, cudf::string_view>)>
  [[nodiscard]] std::unique_ptr<column> operator()(
    column_view const& orderby,
    std::optional<std::pair<rmm::device_uvector<cudf::size_type> const&,
                            rmm::device_uvector<cudf::size_type> const&>> const& grouping,
    bool nulls_at_start,
    scalar const* row_delta,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr) const
  {
    using ScalarT = cudf::scalar_type_t<OrderbyT>;
    if constexpr (WindowType == window_type::CURRENT_ROW || WindowType == window_type::UNBOUNDED) {
      return window_bounds<OrderbyT, ScalarT>(
        orderby, grouping, nulls_at_start, dynamic_cast<ScalarT const*>(row_delta), stream, mr);
    } else {
      CUDF_FAIL(
        "Range windows for strings only support UNBOUNDED and "
        "CURRENT_ROW windows.");
    }
  }

  template <typename OrderbyT, CUDF_ENABLE_IF(!is_supported<OrderbyT>())>
  std::unique_ptr<column> operator()(
    column_view const&,
    std::optional<std::pair<rmm::device_uvector<cudf::size_type> const&,
                            rmm::device_uvector<cudf::size_type> const&>> const&,
    bool,
    scalar const*,
    rmm::cuda_stream_view,
    rmm::device_async_resource_ref) const
  {
    CUDF_FAIL("Unsupported rolling window type.", cudf::data_type_error);
  }
};
}  // namespace detail::rolling
}  // namespace cudf
