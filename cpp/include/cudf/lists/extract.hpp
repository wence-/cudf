/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/lists/lists_column_view.hpp>

namespace cudf {
namespace lists {
/**
 * @addtogroup lists_extract
 * @{
 * @file
 */

/**
 * @brief Create a column where each row is the element at position `index` from the corresponding
 * sublist in the input `lists_column`.
 *
 * Output `column[i]` is set from element `lists_column[i][index]`.
 * If `index` is larger than the size of the sublist at `lists_column[i]`
 * then output `column[i] = null`.
 *
 * @code{.pseudo}
 * l = { {1, 2, 3}, {4}, {5, 6} }
 * r = extract_list_element(l, 1)
 * r is now {2, null, 6}
 * @endcode
 *
 * The `index` may also be negative in which case the row retrieved is offset
 * from the end of each sublist.
 *
 * @code{.pseudo}
 * l = { {"a"}, {"b", "c"}, {"d", "e", "f"} }
 * r = extract_list_element(l, -1)
 * r is now {"a", "c", "f"}
 * @endcode
 *
 * Any input where `lists_column[i] == null` will produce
 * output `column[i] = null`. Also, any element where
 * `lists_column[i][index] == null` will produce
 * output `column[i] = null`.
 *
 * @param lists_column Column to extract elements from.
 * @param index The row within each sublist to retrieve.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Column of extracted elements.
 */
std::unique_ptr<column> extract_list_element(
  lists_column_view const& lists_column,
  size_type index,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Create a column where each row is a single element from the corresponding sublist
 * in the input `lists_column`, selected using indices from the `indices` column.
 *
 * Output `column[i]` is set from element `lists_column[i][indices[i]]`.
 * If `indices[i]` is larger than the size of the sublist at `lists_column[i]`
 * then output `column[i] = null`.
 * Similarly, if `indices[i]` is `null`, then `column[i] = null`.
 *
 * @code{.pseudo}
 * l = { {1, 2, 3}, {4}, {5, 6} }
 * r = extract_list_element(l, {0, null, 2})
 * r is now {1, null, null}
 * @endcode
 *
 * `indices[i]` may also be negative, in which case the row retrieved is offset
 * from the end of each sublist.
 *
 * @code{.pseudo}
 * l = { {"a"}, {"b", "c"}, {"d", "e", "f"} }
 * r = extract_list_element(l, {-1, -2, -4})
 * r is now {"a", "b", null}
 * @endcode
 *
 * Any input where `lists_column[i] == null` produces output `column[i] = null`.
 * Any input where `lists_column[i][indices[i]] == null` produces output `column[i] = null`.
 *
 * @param lists_column Column to extract elements from.
 * @param indices The column whose rows indicate the element index to be retrieved from each list
 * row.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Column of extracted elements.
 * @throws cudf::logic_error If the sizes of `lists_column` and `indices` do not match.
 */
std::unique_ptr<column> extract_list_element(
  lists_column_view const& lists_column,
  column_view const& indices,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/** @} */  // end of group
}  // namespace lists
}  // namespace cudf
