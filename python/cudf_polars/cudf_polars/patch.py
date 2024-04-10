# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools

import polars as pl
from polars.lazyframe.frame import InProcessQuery, wrap_df

from cudf_polars.dataframe import DataFrame
from cudf_polars.plan import execute_plan

_WAS_PATCHED = False


def patch_collect(use_gpu_default=False, cpu_fallback_default=True):
    """Monkey patch LazyFrame.collect to enable GPU execution."""
    global _WAS_PATCHED
    if _WAS_PATCHED:
        return
    _WAS_PATCHED = True
    import rmm

    rmm.reinitialize(pool_allocator=True)

    # TODO: docstring is not correct
    @functools.wraps(pl.LazyFrame.collect)
    def collect_with_gpu(
        self,
        *,
        use_gpu: bool = use_gpu_default,
        return_on_gpu: bool = False,
        cpu_fallback: bool = cpu_fallback_default,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        slice_pushdown: bool = True,
        comm_subplan_elim: bool = True,
        comm_subexpr_elim: bool = True,
        no_optimization: bool = False,
        streaming: bool = False,
        background: bool = False,
        _eager: bool = False,
    ):
        if streaming or _eager or background:
            use_gpu = False
        if no_optimization or _eager:
            predicate_pushdown = False
            projection_pushdown = False
            slice_pushdown = False
            comm_subplan_elim = False
            comm_subexpr_elim = False

        if streaming:
            comm_subplan_elim = False
        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            comm_subplan_elim,
            comm_subexpr_elim,
            streaming,
            _eager,
        )
        if use_gpu:
            # This can't fail
            visitor = ldf.visit()
            try:
                gdf = execute_plan(visitor, profile=False)
                if return_on_gpu:
                    return gdf
                else:
                    result = pl.from_arrow(gdf.to_arrow())
                    for name, val in gdf.sorted().items():
                        if val != DataFrame.IsSorted.NOT:
                            result = result.set_sorted(
                                name,
                                descending=val
                                is DataFrame.IsSorted.DESCENDING,
                            )
                    return result
            except Exception:
                # TODO: Convert todo!()s to something that raises an
                # exception we can catch
                if cpu_fallback:
                    # CPU fallback
                    print("Didn't execute successfully on GPU")
                else:
                    raise
        if background:
            return InProcessQuery(ldf.collect_concurrently())

        return wrap_df(ldf.collect())

    # TODO: docstring is not correct
    @functools.wraps(pl.LazyFrame.profile)
    def profile_with_gpu(
        self,
        *,
        use_gpu: bool = use_gpu_default,
        return_on_gpu: bool = False,
        cpu_fallback: bool = cpu_fallback_default,
        type_coercion: bool = True,
        predicate_pushdown: bool = True,
        projection_pushdown: bool = True,
        simplify_expression: bool = True,
        no_optimization: bool = False,
        slice_pushdown: bool = True,
        comm_subplan_elim: bool = True,
        comm_subexpr_elim: bool = True,
        show_plot: bool = False,
        truncate_nodes: int = 0,
        figsize: tuple[int, int] = (18, 8),
        streaming: bool = False,
    ):
        if streaming:
            use_gpu = False
        if no_optimization:
            predicate_pushdown = False
            projection_pushdown = False
            comm_subplan_elim = False
            comm_subexpr_elim = False

        ldf = self._ldf.optimization_toggle(
            type_coercion,
            predicate_pushdown,
            projection_pushdown,
            simplify_expression,
            slice_pushdown,
            comm_subplan_elim,
            comm_subexpr_elim,
            streaming,
            _eager=False,
        )
        if use_gpu:
            visitor = ldf.visit()
            try:
                gdf, profile = execute_plan(visitor, profile=True)
                if return_on_gpu:
                    return gdf, profile.as_dataframe()
                else:
                    result = pl.from_arrow(gdf.to_arrow())
                    for name, val in gdf.sorted().items():
                        if val != DataFrame.IsSorted.NOT:
                            result = result.set_sorted(
                                name,
                                descending=val
                                is DataFrame.IsSorted.DESCENDING,
                            )
                    return result, profile.as_dataframe()
            except Exception:
                if cpu_fallback:
                    # CPU fallback
                    print("Didn't execute successfully on GPU")
                else:
                    raise
        return tuple(map(wrap_df, ldf.profile()))

    # This is our one (hopefully) API hook for monkeypatching so that
    # we can seamlessly run polars plans on GPU
    pl.LazyFrame.collect = collect_with_gpu
    pl.LazyFrame.profile = profile_with_gpu
