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

#include "detail/range_utils.cuh"
#include "detail/rolling.hpp"
#include "detail/rolling_utils.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/rolling.hpp>
#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/binary_search.h>

#include <optional>

namespace cudf {
namespace detail {

/**
 * @brief Make a column representing the window offsets for a range-based window
 *
 * @tparam Direction Is this a preceding window or a following one.
 *
 * @param group_keys Table defining grouping of the windows. May be empty. If
 * non-empty, group keys must be sorted.
 * @param orderby Column use to define window ranges. If @p group_keys is empty,
 * must be sorted. If
 * @p group_keys is non-empty, must be sorted within each group. As well as
 * being sorted, must be sorted consistently with the @p order and @p null_order
 * parameters.
 * @param order The sort order of the @p orderby column.
 * @param null_order The sort order of nulls in the @p orderby column.
 * @param row_delta Pointer to scalar providing the delta for the window range.
 * May be null, but only if the @p window_type is @p CURRENT_ROW or @p
 * UNBOUNDED. Note that @p row_delta is always added to the current row value.
 * @param window_type The type of window we are computing bounds for.
 * @param stream CUDA stream used for device memory operations and kernel
 * launches.
 * @param mr Device memory resource used for allocations.
 */
template <rolling::direction Direction>
[[nodiscard]] std::unique_ptr<column> inline make_range_window_bounds(
  column_view const& orderby,
  std::optional<std::pair<rmm::device_uvector<cudf::size_type> const&,
                          rmm::device_uvector<cudf::size_type> const&>> const& grouping,
  order order,
  null_order null_order,
  scalar const* row_delta,
  rolling::window_type window,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(window == rolling::window_type::UNBOUNDED ||
                 window == rolling::window_type::CURRENT_ROW || row_delta != nullptr,
               "For bounded windows, row_delta must be non-null.");
  bool const nulls_at_start = (order == order::ASCENDING && null_order == null_order::BEFORE) ||
                              (order == order::DESCENDING && null_order == null_order::AFTER);

  if (window == rolling::window_type::UNBOUNDED && order == order::ASCENDING) {
    return type_dispatcher(
      orderby.type(),
      rolling::range_window_clamper<rolling::window_type::UNBOUNDED, Direction, order::ASCENDING>{},
      orderby,
      grouping,
      nulls_at_start,
      row_delta,
      stream,
      mr);
  } else if (window == rolling::window_type::UNBOUNDED && order == order::DESCENDING) {
    return type_dispatcher(orderby.type(),
                           rolling::range_window_clamper<rolling::window_type::UNBOUNDED,
                                                         Direction,
                                                         order::DESCENDING>{},
                           orderby,
                           grouping,
                           nulls_at_start,
                           row_delta,
                           stream,
                           mr);
  } else if (window == rolling::window_type::CURRENT_ROW && order == order::ASCENDING) {
    return type_dispatcher(orderby.type(),
                           rolling::range_window_clamper<rolling::window_type::CURRENT_ROW,
                                                         Direction,
                                                         order::ASCENDING>{},
                           orderby,
                           grouping,
                           nulls_at_start,
                           row_delta,
                           stream,
                           mr);
  } else if (window == rolling::window_type::CURRENT_ROW && order == order::DESCENDING) {
    return type_dispatcher(orderby.type(),
                           rolling::range_window_clamper<rolling::window_type::CURRENT_ROW,
                                                         Direction,
                                                         order::DESCENDING>{},
                           orderby,
                           grouping,
                           nulls_at_start,
                           row_delta,
                           stream,
                           mr);
  } else if (window == rolling::window_type::BOUNDED_OPEN && order == order::ASCENDING) {
    return type_dispatcher(orderby.type(),
                           rolling::range_window_clamper<rolling::window_type::BOUNDED_OPEN,
                                                         Direction,
                                                         order::ASCENDING>{},
                           orderby,
                           grouping,
                           nulls_at_start,
                           row_delta,
                           stream,
                           mr);
  } else if (window == rolling::window_type::BOUNDED_OPEN && order == order::DESCENDING) {
    return type_dispatcher(orderby.type(),
                           rolling::range_window_clamper<rolling::window_type::BOUNDED_OPEN,
                                                         Direction,
                                                         order::DESCENDING>{},
                           orderby,
                           grouping,
                           nulls_at_start,
                           row_delta,
                           stream,
                           mr);
  } else if (window == rolling::window_type::BOUNDED_CLOSED && order == order::ASCENDING) {
    return type_dispatcher(orderby.type(),
                           rolling::range_window_clamper<rolling::window_type::BOUNDED_CLOSED,
                                                         Direction,
                                                         order::ASCENDING>{},
                           orderby,
                           grouping,
                           nulls_at_start,
                           row_delta,
                           stream,
                           mr);
  } else if (window == rolling::window_type::BOUNDED_CLOSED && order == order::DESCENDING) {
    return type_dispatcher(orderby.type(),
                           rolling::range_window_clamper<rolling::window_type::BOUNDED_CLOSED,
                                                         Direction,
                                                         order::DESCENDING>{},
                           orderby,
                           grouping,
                           nulls_at_start,
                           row_delta,
                           stream,
                           mr);
  } else {
    CUDF_FAIL(
      "Unsupported window type and sorted order combination for range "
      "window bounds");
  }
}

}  // namespace detail
std::unique_ptr<column> grouped_range_rolling_window_v2(table_view const& group_keys,
                                                        column_view const& orderby,
                                                        column_view const& values,
                                                        cudf::order order,
                                                        cudf::null_order null_order,
                                                        range_window_bounds const& preceding_window,
                                                        range_window_bounds const& following_window,
                                                        size_type min_periods,
                                                        rolling_aggregation const& agg,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto make_preceding =
    [&](std::optional<std::pair<rmm::device_uvector<cudf::size_type> const&,
                                rmm::device_uvector<cudf::size_type> const&>> const& grouping) {
      return detail::make_range_window_bounds<detail::rolling::direction::PRECEDING>(
        orderby,
        grouping,
        order,
        null_order,
        &preceding_window.range_scalar(),
        detail::rolling::window_type::BOUNDED_CLOSED,
        stream,
        mr);
    };
  auto make_following =
    [&](std::optional<std::pair<rmm::device_uvector<cudf::size_type> const&,
                                rmm::device_uvector<cudf::size_type> const&>> const& grouping) {
      return detail::make_range_window_bounds<detail::rolling::direction::FOLLOWING>(
        orderby,
        grouping,
        order,
        null_order,
        &following_window.range_scalar(),
        detail::rolling::window_type::BOUNDED_CLOSED,
        stream,
        mr);
    };
  auto [preceding, following] = [&]() {
    if (group_keys.num_columns() > 0) {
      CUDF_EXPECTS(group_keys.num_rows() == orderby.size(),
                   "Grouping table and orderby column must have same number of rows.");
      using sort_helper = cudf::groupby::detail::sort::sort_groupby_helper;
      sort_helper helper{group_keys, null_policy::INCLUDE, sorted::YES, {}};
      auto const& labels  = helper.group_labels(stream);
      auto const& offsets = helper.group_offsets(stream);
      std::pair<rmm::device_uvector<cudf::size_type> const&,
                rmm::device_uvector<cudf::size_type> const&>
        pair = {labels, offsets};
      return std::pair{std::move(make_preceding(pair)), std::move(make_following(pair))};
    } else {
      return std::pair{std::move(make_preceding(std::nullopt)),
                       std::move(make_following(std::nullopt))};
    }
  }();
  return detail::rolling_window(values, preceding->view(), following->view(), 1, agg, stream, mr);
}
}  // namespace cudf
