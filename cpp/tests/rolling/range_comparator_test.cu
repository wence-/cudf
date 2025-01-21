/*
 * Copyright (c) 2023-2025, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <src/rolling/detail/range_utils.cuh>

#include <type_traits>

struct RangeComparatorTest : cudf::test::BaseFixture {};

template <typename T>
struct RangeComparatorTypedTest : RangeComparatorTest {};

using TestTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::FloatingPointTypes>;

TYPED_TEST_SUITE(RangeComparatorTypedTest, TestTypes);

TYPED_TEST(RangeComparatorTypedTest, TestLessComparator)
{
  auto const less     = cudf::detail::rolling::less<TypeParam>{};
  auto constexpr nine = TypeParam{9};
  auto constexpr ten  = TypeParam{10};

  EXPECT_TRUE(less(nine, ten));
  EXPECT_FALSE(less(ten, nine));
  EXPECT_FALSE(less(ten, ten));

  if constexpr (std::is_floating_point_v<TypeParam>) {
    auto constexpr NaN = std::numeric_limits<TypeParam>::quiet_NaN();
    auto constexpr Inf = std::numeric_limits<TypeParam>::infinity();
    // NaN.
    EXPECT_FALSE(less(NaN, ten));
    EXPECT_FALSE(less(NaN, NaN));
    EXPECT_FALSE(less(NaN, Inf));
    EXPECT_FALSE(less(NaN, -Inf));
    // Infinity.
    EXPECT_TRUE(less(Inf, NaN));
    EXPECT_FALSE(less(Inf, Inf));
    EXPECT_FALSE(less(Inf, ten));
    EXPECT_FALSE(less(Inf, -Inf));
    // -Infinity.
    EXPECT_TRUE(less(-Inf, NaN));
    EXPECT_TRUE(less(-Inf, Inf));
    EXPECT_TRUE(less(-Inf, ten));
    EXPECT_FALSE(less(-Inf, -Inf));
    // Finite.
    EXPECT_TRUE(less(ten, NaN));
    EXPECT_TRUE(less(ten, Inf));
    EXPECT_FALSE(less(ten, -Inf));
  }
}

TYPED_TEST(RangeComparatorTypedTest, TestGreaterComparator)
{
  auto const greater  = cudf::detail::rolling::greater<TypeParam>{};
  auto constexpr nine = TypeParam{9};
  auto constexpr ten  = TypeParam{10};

  EXPECT_FALSE(greater(nine, ten));
  EXPECT_TRUE(greater(ten, nine));
  EXPECT_FALSE(greater(ten, ten));

  if constexpr (std::is_floating_point_v<TypeParam>) {
    auto constexpr NaN = std::numeric_limits<TypeParam>::quiet_NaN();
    auto constexpr Inf = std::numeric_limits<TypeParam>::infinity();
    // NaN.
    EXPECT_TRUE(greater(NaN, ten));
    EXPECT_FALSE(greater(NaN, NaN));
    EXPECT_TRUE(greater(NaN, Inf));
    EXPECT_TRUE(greater(NaN, -Inf));
    // Infinity.
    EXPECT_FALSE(greater(Inf, NaN));
    EXPECT_FALSE(greater(Inf, Inf));
    EXPECT_TRUE(greater(Inf, ten));
    EXPECT_TRUE(greater(Inf, -Inf));
    // -Infinity.
    EXPECT_FALSE(greater(-Inf, NaN));
    EXPECT_FALSE(greater(-Inf, Inf));
    EXPECT_FALSE(greater(-Inf, ten));
    EXPECT_FALSE(greater(-Inf, -Inf));
    // Finite.
    EXPECT_FALSE(greater(ten, NaN));
    EXPECT_FALSE(greater(ten, Inf));
    EXPECT_TRUE(greater(ten, -Inf));
  }
}

TYPED_TEST(RangeComparatorTypedTest, TestAddSafe)
{
  using T = TypeParam;
  EXPECT_EQ(cudf::detail::rolling::add_sat(T{3}, T{4}), T{7});

  if constexpr (cuda::std::numeric_limits<T>::is_signed) {
    EXPECT_EQ(cudf::detail::rolling::add_sat(T{-3}, T{4}), T{1});
  }

  auto constexpr max = cuda::std::numeric_limits<T>::max();
  EXPECT_EQ(cudf::detail::rolling::add_sat(T{max - 5}, T{4}), max - 1);
  EXPECT_EQ(cudf::detail::rolling::add_sat(T{max - 4}, T{4}), max);
  EXPECT_EQ(cudf::detail::rolling::add_sat(T{max - 3}, T{4}), max);
  EXPECT_EQ(cudf::detail::rolling::add_sat(max, T{4}), max);
  if constexpr (std::is_signed_v<T>) {
    auto constexpr min = cuda::std::numeric_limits<T>::lowest();
    EXPECT_EQ(cudf::detail::rolling::add_sat(T{-10}, min), min);
  }

  if constexpr (std::is_floating_point_v<T>) {
    auto const NaN = std::numeric_limits<T>::quiet_NaN();
    auto const Inf = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(std::isnan(cudf::detail::rolling::add_sat(NaN, T{4})));
    EXPECT_EQ(cudf::detail::rolling::add_sat(Inf, T{4}), Inf);
  }
}

TYPED_TEST(RangeComparatorTypedTest, TestSubtractSafe)
{
  using T = TypeParam;
  EXPECT_EQ(cudf::detail::rolling::sub_sat(T{4}, T{3}), T{1});

  if constexpr (cuda::std::numeric_limits<T>::is_signed) {
    EXPECT_EQ(cudf::detail::rolling::sub_sat(T{3}, T{4}), T{-1});
  }

  auto constexpr min = cuda::std::numeric_limits<T>::lowest();
  auto constexpr max = cuda::std::numeric_limits<T>::max();
  EXPECT_EQ(cudf::detail::rolling::sub_sat(T{min + 5}, T{4}), min + 1);
  EXPECT_EQ(cudf::detail::rolling::sub_sat(T{min + 4}, T{4}), min);
  EXPECT_EQ(cudf::detail::rolling::sub_sat(T{min + 3}, T{4}), min);
  EXPECT_EQ(cudf::detail::rolling::sub_sat(min, T{4}), min);
  EXPECT_EQ(cudf::detail::rolling::sub_sat(min, max), min);

  if constexpr (std::is_signed_v<T>) {
    EXPECT_EQ(cudf::detail::rolling::sub_sat(T{max - 1}, min), max);
  }
  if constexpr (std::is_floating_point_v<T>) {
    auto const NaN = std::numeric_limits<T>::quiet_NaN();
    auto const Inf = std::numeric_limits<T>::infinity();
    EXPECT_TRUE(std::isnan(cudf::detail::rolling::sub_sat(NaN, T{4})));
    EXPECT_EQ(cudf::detail::rolling::sub_sat(-Inf, T{4}), -Inf);
  }
}
