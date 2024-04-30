# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from polars.testing.asserts import assert_frame_equal


def assert_gpu_result_equal(
    lazydf,
    *,
    check_row_order: bool = True,
    check_column_order: bool = True,
    check_dtype: bool = True,
    check_exact: bool = True,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    categorical_as_str: bool = False,
):
    """
    Assert that collection of a lazyframe on GPU produces correct results.

    Parameters
    ----------
    lazydf
        frame to collect.
    check_row_order
        Expect rows to be in same order
    check_column_order
        Expect columns to be in same order
    check_dtype
        Expect dtypes to match
    check_exact
        Require exact equality for floats, if `False` compare using
        rtol and atol.
    rtol
        Relative tolerance for float comparisons
    atol
        Absolute tolerance for float comparisons
    categorical_as_str
        Decat categoricals to strings before comparing

    Raises
    ------
    AssertionError
        If the GPU and CPU collection do not match.
    NotImplementedError
        If GPU collection failed in some way.
    """
    expect = lazydf.collect(use_gpu=False)
    got = lazydf.collect(use_gpu=True, cpu_fallback=False)
    assert_frame_equal(
        expect,
        got,
        check_row_order=check_row_order,
        check_column_order=check_column_order,
        check_dtype=check_dtype,
        check_exact=check_exact,
        rtol=rtol,
        atol=atol,
        categorical_as_str=categorical_as_str,
    )
