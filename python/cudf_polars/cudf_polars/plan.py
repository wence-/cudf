# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import itertools
import operator
import time
from functools import partial, reduce, singledispatch
from typing import TYPE_CHECKING, NamedTuple

import cudf
import cudf._lib.pylibcudf as plc
import numpy as np
import nvtx
import polars as pl
import pyarrow as pa
from cudf.core.column import as_column, column_empty
from polars.polars import nodes

from cudf_polars.dataframe import DataFrame
from cudf_polars.expressions import (
    ExprVisitor,
    _post_aggregate,
    _rolling,
    collect_aggs,
)
from cudf_polars.utils import placeholder_column, sort_order

if TYPE_CHECKING:
    from cudf_polars.typing import Plan, Visitor


def _dataframe_from_cudf(df):
    return DataFrame(
        {name: col.to_pylibcudf(mode="read") for name, col in df._data.items()}
    )


class ExecutionProfiler:
    """Object for recording execution timeline."""

    class _context:
        def __init__(self, prof, name):
            self.prof = prof
            self.name = name

        def __enter__(self):
            self.node_begin = time.perf_counter_ns()

        def __exit__(self, *args):
            self.prof.stack.append(
                (self.name, self.node_begin, time.perf_counter_ns())
            )

    def __init__(self):
        self.start = time.perf_counter_ns()
        self.stack = []

    def record(self, name: str):
        """
        Return a context manager for recording time for a block of code.

        Parameters
        ----------
        name
            Name of the block for presentation purposes

        Returns
        -------
        Context manager for timing
        """
        return ExecutionProfiler._context(self, name)

    def as_dataframe(self):
        """Return the profiling information as a dataframe."""
        names, starts, ends = zip(*sorted(self.stack, key=lambda x: x[1]))
        starts = (np.asarray(starts, dtype="uint64") - self.start) // 1000
        ends = (np.asarray(ends, dtype="uint64") - self.start) // 1000
        return pl.DataFrame({"node": names, "start": starts, "end": ends})


class NoopProfiler:
    """No-op profiling object to mimic ExecutionProfiler."""

    def __init__(self):
        self.ctx = contextlib.nullcontext()

    def record(self, name: str):
        """
        Return a null context manager.

        Parameters
        ----------
        name
            Ignored

        Returns
        -------
        Null context manager.
        """
        return self.ctx


class PlanVisitor(NamedTuple):
    """Object holding rust visitor and utility methods."""

    visitor: Visitor
    cache: dict[int, DataFrame]
    profiler: ExecutionProfiler | NoopProfiler
    expr_visitor: ExprVisitor

    def __call__(self, n: int | None = None) -> DataFrame:
        """
        Evaluate a plan node to produce a dataframe.

        Parameters
        ----------
        n
            Node to evaluate (optional), if not provided uses the internal
            visitor's "current" node.

        Returns
        -------
        New dataframe giving the evaluation of the plan.
        """
        if n is None:
            node = self.visitor.view_current_node()
        else:
            node = self.visitor.view_node(n)
        return _execute_plan(node, self)

    def record(self, name: str):
        """
        Return a context manager for profiling a block of code.

        Parameters
        ----------
        name
            name of the block

        Returns
        -------
        Context manager recording execution time of a block.
        """
        return self.profiler.record(name)


@nvtx.annotate("Execute Polars plan", domain="cudf_polars")
def execute_plan(
    visitor: Visitor, *, profile: bool = False
) -> DataFrame | tuple[DataFrame, ExecutionProfiler]:
    """
    Execute a polars logical plan using cudf.

    Parameters
    ----------
    visitor
        Rust visitor of a logical plan
    profile
        Produce profiling data for plan node execution time?

    Returns
    -------
    DataFrame representing the execution of the plan
    """
    profiler: ExecutionProfiler | NoopProfiler
    if profile:
        profiler = ExecutionProfiler()
        return PlanVisitor(visitor, {}, profiler, ExprVisitor(visitor))(
            n=None
        ), profiler
    else:
        profiler = NoopProfiler()
        return PlanVisitor(visitor, {}, profiler, ExprVisitor(visitor))(n=None)


@singledispatch
def _execute_plan(plan: Plan, visitor: PlanVisitor) -> DataFrame:
    """
    Run a polars logical plan.

    Parameters
    ----------
    plan
        Plan to run
    visitor
        Wrapper of rust visitor

    Returns
    -------
    An evaluation of the plan as a DataFrame
    """
    raise AssertionError(f"Unhandled plan type {type(plan)}")


@_execute_plan.register
def _python_scan(plan: nodes.PythonScan, visitor: PlanVisitor):
    with visitor.record("PythonScan"):
        (
            scan_fn,
            schema,
            output_schema,
            with_columns,
            is_pyarrow,
            predicate,
            nrows,
        ) = plan.options
        predicate = plan.predicate
        if is_pyarrow:
            raise NotImplementedError("Don't know what to do here")
        context = scan_fn(with_columns, predicate, nrows)
        if predicate is not None:
            mask = visitor.expr_visitor(predicate.node, context)
            return context.filter(mask)
        else:
            return context


@_execute_plan.register
def _scan(plan: nodes.Scan, visitor: PlanVisitor):
    scan_type = plan.scan_type
    with visitor.record(f"{scan_type}"):
        paths = plan.paths
        options = plan.file_options
        n_rows = options.n_rows
        with_columns = options.with_columns
        row_index = options.row_index
        schema = plan.output_schema
        # TODO: Send all the options through to the libcudf readers where appropriate
        if n_rows is not None:
            # TODO: read_csv supports n_rows, but if we have more than one
            # file to read how should we apply it?
            raise NotImplementedError("Row limit in scan is not supported")
        if scan_type == "csv":
            # Note: We could use pylibcudf for the concatenation
            # already, but we don't have I/O there yet so we might as
            # well hold off until we can do the entire op with
            # pylibcudf.
            df = _dataframe_from_cudf(
                cudf.concat(
                    [cudf.read_csv(p, usecols=with_columns) for p in paths]
                )
            )
        elif scan_type == "parquet":
            df = _dataframe_from_cudf(
                cudf.read_parquet(paths, columns=with_columns)
            )
        else:
            raise NotImplementedError(f"Unhandled {scan_type=}")
        if row_index is not None:
            (name, offset) = row_index
            # TODO: Handle properly
            assert schema[name] == pl.UInt32
            index = as_column(
                range(offset, offset + df.num_rows()), dtype=np.uint32
            )
            df = DataFrame({name: index} | df)
        if plan.predicate is None:
            return df
        else:
            # TODO: cudf's read_parquet only handles DNF expressions of single
            # column predicates. polars allows for an arbitrary expression
            # that evaluates to a boolean.
            mask = visitor.expr_visitor(plan.predicate.node, df)
            return df.filter(mask)


@_execute_plan.register
def _cache(plan: nodes.Cache, visitor: PlanVisitor):
    key = plan.id_
    cache = visitor.cache
    try:
        return cache[key]
    except KeyError:
        return cache.setdefault(key, visitor(plan.input))


@_execute_plan.register
def _dataframescan(plan: nodes.DataFrameScan, visitor: PlanVisitor):
    with visitor.record("DataFrameScan"):
        pdf = pl.DataFrame._from_pydf(plan.df)
        # Run column projection as zero-copy on the polars dataframe
        if plan.projection is not None:
            pdf = (
                pdf.lazy()
                .select(plan.projection)
                .collect(use_gpu=False, _eager=True)
            )

        arrow_table = pdf.to_arrow()
        # replace any `large_string` in the schema with `string`:
        # cudf doesn't support `large_string` yet
        schema = arrow_table.schema
        for i, field in enumerate(schema):
            if field.type == pa.large_string():
                schema = schema.set(i, pa.field(field.name, pa.string()))
        arrow_table = arrow_table.cast(schema)

        plc_table = plc.interop.from_arrow(arrow_table)
        context = DataFrame(
            dict(
                zip(arrow_table.column_names, plc_table.columns(), strict=True)
            )
        )
        if plan.selection is not None:
            # Filters
            mask = visitor.expr_visitor(plan.selection.node, context)
            return context.filter(mask)
        else:
            return context


@_execute_plan.register
def _select(plan: nodes.Select, visitor: PlanVisitor):
    context = visitor(plan.input)
    with visitor.record("select"):
        # TODO: loses sortedness properties
        for cse in plan.cse_expr:
            context |= {
                cse.output_name: visitor.expr_visitor(cse.node, context)
            }
        return DataFrame(
            {
                e.output_name: visitor.expr_visitor(e.node, context)
                for e in plan.expr
            }
        )


@_execute_plan.register
def _groupby(plan: nodes.GroupBy, visitor: PlanVisitor):
    name = "group_by" if plan.options.rolling is None else "rolling"
    # Input frame to groupby
    context = visitor(plan.input)
    agg_names = [e.output_name for e in plan.aggs]
    agg_nodes = [e.node for e in plan.aggs]
    with visitor.record(name):
        # This should be list of mappings of names to columns. This
        # will happily produce grouping keys that are expressions from
        # the input.
        keys = DataFrame(
            {
                k.output_name: visitor.expr_visitor(k.node, context)
                for k in plan.keys
            }
        )
        # TODO: handle dropna options
        # TODO: One is allowed, in polars to aggregate arbitrary
        # expressions, and even nest aggregation requests. This could be
        # supported in this setup with a multi-pass implementation,
        # assuming that all of the shapes line up. But for now,
        # collect_aggs bails out if it observes nested aggregations.
        input_columns, requests, aggs_to_replace = collect_aggs(
            agg_nodes, context, visitor.expr_visitor
        )

        if plan.options.rolling is None:  # regular group-by aggregation
            if plan.maintain_order:
                raise NotImplementedError("Maintaining order in group_by")
            grouper = plc.groupby.GroupBy(keys.to_pylibcudf())
            agg_requests = [
                plc.groupby.GroupByRequest(
                    column
                    if column is not None
                    else placeholder_column(keys.num_rows()),
                    reqs,
                )
                for column, reqs in zip(input_columns, requests, strict=True)
            ]
            # TODO: check that all aggs were performed
            group_keys, raw_tables = grouper.aggregate(agg_requests)
            group_keys = group_keys.columns()
            raw_columns = [table.columns() for table in raw_tables]
        else:  # rolling group-by aggregation
            # TODO: I think grouped rolling aggregations need the input
            # dataframe to be sorted by the group keys.
            # TODO: count
            if any(c is None for c in input_columns):
                raise NotImplementedError("count aggregation in rolling")
            options = plan.options.rolling
            index_column = context[options.index_column]
            roll = partial(
                _rolling,
                index_column=index_column,
                period=options.period,
                keys=None if len(keys) == 0 else keys.columns(),
            )
            raw_columns = [
                roll(input_column=col, aggs=request)
                for request, col in zip(requests, input_columns, strict=True)
            ]
            keys = DataFrame(keys | {options.index_column: index_column})
            group_keys = keys.columns()

        agg_columns = _post_aggregate(
            raw_columns, agg_nodes, aggs_to_replace, visitor.expr_visitor
        )
        zlice = plan.options.slice
        result = DataFrame(
            zip(
                itertools.chain(keys.names(), agg_names),
                itertools.chain(group_keys, agg_columns),
                strict=True,
            )
        )
        if zlice is not None:
            return result.slice(*zlice)
        else:
            return result


@_execute_plan.register
def _join(plan: nodes.Join, visitor: PlanVisitor):
    left = visitor(plan.input_left)
    right = visitor(plan.input_right)
    with visitor.record("join"):
        left_on = plc.Table(
            [visitor.expr_visitor(e.node, left) for e in plan.left_on]
        )
        right_on = plc.Table(
            [visitor.expr_visitor(e.node, right) for e in plan.right_on]
        )
        how, join_nulls, zlice, suffix = plan.options
        null_equality = (
            plc.types.NullEquality.EQUAL
            if join_nulls
            else plc.types.NullEquality.UNEQUAL
        )
        suffix = "_right" if suffix is None else suffix
        if how == "cross":
            raise NotImplementedError("cross join not implemented")
        coalesce_key_columns = True
        if how == "outer":
            coalesce_key_columns = False
            raise NotImplementedError("Non-coalescing outer join")
        elif how == "outer_coalesce":
            how = "outer"
        joiner, left_policy, right_policy = {
            "inner": (
                plc.join.inner_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
            ),
            "left": (
                plc.join.left_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
            ),
            "outer": (
                plc.join.full_join,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
            ),
            "leftsemi": (
                plc.join.left_semi_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                None,
            ),
            "leftanti": (
                plc.join.left_anti_join,
                plc.copying.OutOfBoundsPolicy.DONT_CHECK,
                None,
            ),
        }[how]
        if right_policy is None:
            lg = joiner(left_on, right_on, null_equality)
            result = DataFrame.from_pylibcudf(
                left.names(),
                plc.copying.gather(left.to_pylibcudf(), lg, left_policy),
            )
        else:
            lg, rg = joiner(left_on, right_on, null_equality)
            left = DataFrame.from_pylibcudf(
                left.names(),
                plc.copying.gather(left.to_pylibcudf(), lg, left_policy),
            )
            right_key_names = {e.output_name for e in plan.right_on}
            right_names = [
                name if name not in left else f"{name}{suffix}"
                for name in right.names()
                if name not in right_key_names
            ]
            right = DataFrame.from_pylibcudf(
                right_names,
                plc.copying.gather(
                    right.discard(right_key_names).to_pylibcudf(),
                    rg,
                    right_policy,
                ),
            )
            if how == "outer" and coalesce_key_columns:
                for name, replacement in zip(
                    (e.output_name for e in plan.left_on),
                    plc.copying.gather(right_on, rg, right_policy).columns(),
                    strict=True,
                ):
                    left[name] = plc.replace.replace_nulls(
                        left[name], replacement
                    )
            result = DataFrame(left | right)
        if zlice is not None:
            return result.slice(*zlice)
        else:
            return result


@_execute_plan.register
def _hstack(plan: nodes.HStack, visitor: PlanVisitor):
    result = visitor(plan.input)
    with visitor.record("hstack"):
        columns = {
            e.output_name: visitor.expr_visitor(e.node, result)
            for e in plan.exprs
        }
        # TODO: loses sortedness property
        return DataFrame(result | columns)


@_execute_plan.register
def _distinct(plan: nodes.Distinct, visitor: PlanVisitor):
    result = visitor(plan.input)
    with visitor.record("distinct"):
        (keep, subset, maintain_order, zlice) = plan.options
        keep = {
            "first": plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
            "last": plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
            "none": plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
            "any": plc.stream_compaction.DuplicateKeepOption.KEEP_ANY,
        }[keep]
        if subset is not None:
            subset = set(subset)
            indices = [i for i, k in enumerate(result.keys()) if k in subset]
        else:
            subset = set(result.keys())
            indices = list(range(len(result)))
        sortedness = result.sorted()
        keys_sorted = all(
            sortedness[c] != DataFrame.IsSorted.NOT for c in subset
        )
        if keys_sorted:
            table = plc.stream_compaction.unique(
                result.to_pylibcudf(),
                indices,
                keep,
                plc.types.NullEquality.EQUAL,
            )
        else:
            compact = (
                plc.stream_compaction.stable_distinct
                if maintain_order
                else plc.stream_compaction.distinct
            )
            table = compact(
                result.to_pylibcudf(),
                indices,
                keep,
                plc.types.NullEquality.EQUAL,
                plc.types.NanEquality.ALL_EQUAL,
            )
        result = DataFrame.from_pylibcudf(result.names(), table)
        if keys_sorted or maintain_order:
            result = result.set_sorted(sortedness)
        if zlice is not None:
            return result.slice(*zlice)
        else:
            return result


@_execute_plan.register
def _sort(plan: nodes.Sort, visitor: PlanVisitor):
    result = visitor(plan.input)
    with visitor.record("sort"):
        input_col_ids = set(map(id, result.values()))
        sort_keys = [
            visitor.expr_visitor(e.node, result) for e in plan.by_column
        ]
        (stable, nulls_last, descending, zlice) = plan.args
        descending, column_order, null_precedence = sort_order(
            descending, nulls_last=nulls_last, num_keys=len(sort_keys)
        )
        do_sort = (
            plc.sorting.stable_sort_by_key
            if stable
            else plc.sorting.sort_by_key
        )
        result = DataFrame.from_pylibcudf(
            result.names(),
            do_sort(
                result.to_pylibcudf(),
                plc.Table(sort_keys),
                column_order,
                null_precedence,
            ),
        )
        sortedness = {
            name: (
                DataFrame.IsSorted.DESCENDING
                if d
                else DataFrame.IsSorted.ASCENDING
            )
            for d, name, col in zip(
                descending,
                (e.output_name for e in plan.by_column),
                sort_keys,
                strict=True,
            )
            if id(col) in input_col_ids
        }
        result = result.set_sorted(sortedness)
        if zlice is not None:
            return result.slice(*zlice)
        else:
            return result


@_execute_plan.register
def _slice(plan: nodes.Slice, visitor: PlanVisitor):
    result = visitor(plan.input)
    with visitor.record("slice"):
        return result.slice(plan.offset, plan.len)


@_execute_plan.register
def _filter(plan: nodes.Filter, visitor: PlanVisitor):
    result = visitor(plan.input)
    with visitor.record("filter"):
        mask = visitor.expr_visitor(plan.predicate.node, result)
        return result.filter(mask)


@_execute_plan.register
def _simple_projection(plan: nodes.SimpleProjection, visitor: PlanVisitor):
    result = visitor(plan.input)
    schema = plan.columns
    with visitor.record("simple_projection"):
        return DataFrame({name: result[name] for name in schema})


@_execute_plan.register
def _map_function(plan: nodes.MapFunction, visitor: PlanVisitor):
    typ, *args = plan.function
    profiler = visitor.record(f"function-{typ}")
    if typ == "unnest":
        (to_unnest,) = args
        raise NotImplementedError("unnest")
    elif typ == "drop_nulls":
        context = visitor(plan.input)
        with profiler:
            (subset,) = args
            subset = set(subset)
            column_names = context.names()
            indices = [
                i for i, name in enumerate(column_names) if name in subset
            ]
            return DataFrame.from_pylibcudf(
                column_names,
                plc.stream_compaction.drop_nulls(
                    context.to_pylibcudf(), indices, len(indices)
                ),
            )
    elif typ == "rechunk":
        # No-op in a non-chunked setting
        return visitor(plan.input)
    elif typ == "merge_sorted":
        pieces = plan.input
        # merge_sorted operates on Union inputs
        # but carries a horrible implementation detail that Union(a,
        # b) produces a vstacked dataframe with a in a single chunk
        # and then b. So that the merge_sorted implementation can
        # extract the pieces by extracting the chunks to reproduce a
        # and b.
        # We don't have that luxury so we assume we have a union, and
        # evaluate the pieces.
        assert isinstance(pieces, nodes.Union)
        first, *rest = (visitor(piece) for piece in pieces.inputs)
        with profiler:
            (key_column,) = args
            column_names = first.names()
            # TODO: do we need to check this?
            if not all(column_names == r.names() for r in rest):
                raise ValueError(
                    "DataFrame names must all align in merge_sorted"
                )
            indices = [
                i for i, name in enumerate(column_names) if name == key_column
            ]
            # TODO: polars merge_sorted doesn't allow specification of
            # more than one key column or null location nulls always sort
            # first, merging is always with ascending data
            num_keys = len(indices)

            return DataFrame.from_pylibcudf(
                column_names,
                plc.merge.merge_sorted(
                    [df.to_pylibcudf() for df in [first, *rest]],
                    indices,
                    [plc.types.Order.ASCENDING] * num_keys,
                    [plc.types.NullOrder.BEFORE] * num_keys,
                ),
            )
    elif typ == "rename":
        context = visitor(plan.input)
        with profiler:
            old_names, new_names, _ = args
            return context.rename(dict(zip(old_names, new_names, strict=True)))
    elif typ == "explode":
        context = visitor(plan.input)
        with profiler:
            column_names, schema = args
            if len(column_names) > 1:
                # TODO: straightforward, but need to error check
                # polars requires that all to-explode columns have the
                # same sub-shapes
                raise NotImplementedError("Explode with more than one column")
            (column_name,) = column_names
            idx = context.names().index(column_name)
            return DataFrame.from_pylibcudf(
                context.names(),
                plc.lists.explode_outer(context.to_pylibcudf(), idx),
            )
    elif typ == "melt":
        raise NotImplementedError("TODO: melt")
    elif typ == "row_index":
        raise NotImplementedError("TODO: row_index")
    else:
        raise ValueError(f"Unexpected map function type: {typ}")


@_execute_plan.register
def _union(plan: nodes.Union, visitor: PlanVisitor):
    input_tables = [visitor(p) for p in plan.inputs]
    with visitor.record("union"):
        # ordered set
        all_names = list(
            itertools.chain.from_iterable(t.names() for t in input_tables)
        )
        schema = reduce(operator.or_, (t.schema() for t in input_tables))
        tables = [
            plc.Table(
                [
                    (
                        table.get(k, None)
                        or column_empty(
                            table.num_rows(), dtype=schema[k], masked=True
                        )
                    )
                    for k in all_names
                ]
            )
            for table in input_tables
        ]
        zlice = plan.options
        result = DataFrame.from_pylibcudf(
            all_names, plc.concatenate.concatenate(tables)
        )
        if zlice is not None:
            return result.slice(*zlice)
        else:
            return result


@_execute_plan.register
def _hconcat(plan: nodes.HConcat, visitor: PlanVisitor):
    return DataFrame(
        reduce(
            operator.or_,
            (visitor(p) for p in plan.inputs),
            {},
        )
    )


@_execute_plan.register
def _extcontext(plan: nodes.ExtContext, visitor: PlanVisitor):
    result = visitor(plan.input)
    # TODO: This is not right, e.g. if there is a projection that
    # selects some subset of the columns. But it seems it is not
    # pushed inside the ExtContext node, so we need some other way of
    # handling that.
    return DataFrame(
        reduce(
            operator.or_,
            (visitor(p) for p in plan.contexts),
            result,
        )
    )


@_execute_plan.register
def _sink(plan: nodes.Sink, visitor: PlanVisitor):
    raise NotImplementedError("We can never see this node")