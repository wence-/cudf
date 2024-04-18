# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import operator
from collections import defaultdict
from functools import reduce, singledispatch
from typing import TYPE_CHECKING, NamedTuple

import cudf
import cudf._lib as libcudf
import cudf._lib.pylibcudf as plc
import numpy as np
import pyarrow as pa
from cudf._lib.types import (
    dtype_from_pylibcudf_column,
    dtype_to_pylibcudf_type,
)
from cudf.core.column import as_column
from cudf.utils import cudautils
from polars import polars as plrs
from polars.polars import expr_nodes

from cudf_polars.dataframe import DataFrame
from cudf_polars.utils import (
    placeholder_column,
    sort_order,
    to_cudf_dtype,
    to_pylibcudf_dtype,
)

if TYPE_CHECKING:
    from typing_extensions import Self

    from cudf_polars.typing import ColumnType, Expr, Visitor


class ExprVisitor(NamedTuple):
    """Object holding rust visitor and utility methods."""

    visitor: Visitor

    def node(self, n: int) -> Expr:
        """
        Translate node to python object.

        Parameters
        ----------
        n
            Node to replace.

        Returns
        -------
        Python representation of the node.
        """
        return self.visitor.view_expression(n)

    def with_replacements(self, mapping: list[tuple[int, Expr]]) -> Self:
        """
        Return a new visitor with nodes replaced by new ones.

        Parameters
        ----------
        mapping
            List of pairs mapping node ids to their replacement expression.

        Returns
        -------
        New node visitor with replaced expressions.
        """
        return type(self)(self.visitor.replace_expressions(mapping))


@singledispatch
def evaluate_expr(
    expr: Expr, context: DataFrame, visitor: ExprVisitor
) -> DataFrame:
    """
    Evaluate a polars expression given a DataFrame for context.

    Parameters
    ----------
    expr
        Expression to evaluate
    context
        Context for evaluation
    visitor
        Visitor callback object

    Returns
    -------
    DataFrame the results of evaluating the expression
    """
    raise AssertionError(f"Unhandled expression type {type(expr)}")


@evaluate_expr.register(list)
@evaluate_expr.register(tuple)
def _expr_list(expr: list | tuple, context: DataFrame, visitor: ExprVisitor):
    results = [evaluate_expr(visitor.node(e), context, visitor) for e in expr]
    n_columns = sum(map(len, (r.keys() for r in results)))
    result = DataFrame(reduce(operator.or_, results, {}))
    if len(result.keys()) != n_columns:
        raise ValueError("Evaluation of list of expressions lost columns")
    return result


@evaluate_expr.register
def _expr_function(
    expr: expr_nodes.Function, context: DataFrame, visitor: ExprVisitor
):
    fname, *fargs = expr.function_data
    data = evaluate_expr(expr.input, context, visitor)
    if fname == "argwhere":
        ((name, mask),) = data.items()
        indices = plc.stream_compaction.apply_boolean_mask(
            plc.Table(
                [
                    as_column(
                        range(mask.size()), dtype=libcudf.types.size_type_dtype
                    ).to_pylibcudf(mode="read")
                ]
            ),
            mask,
        )
        return DataFrame.from_pylibcudf([name], indices)
    elif fname == "setsorted":
        (name,) = data.keys()
        (flag,) = fargs
        return data.set_sorted(
            {name: getattr(DataFrame.IsSorted, flag.upper())}
        )
    elif fname == "is_not_null":
        ((name, column),) = data.items()
        return DataFrame.from_pylibcudf(
            data.names(),
            plc.Table([plc.unary.is_valid(column)]),
        )
    else:
        raise NotImplementedError(f"Function expression {fname=}")


@evaluate_expr.register
def _expr_window(
    expr: expr_nodes.Window, context: DataFrame, visitor: ExprVisitor
):
    if isinstance(expr.options, expr_nodes.PyWindowMapping):
        raise NotImplementedError(".over() not supported")
    (col,), (requests,), aggs_to_replace, (name,) = collect_aggs(
        [expr.function], context, visitor
    )
    if isinstance(expr.options, expr_nodes.PyRollingGroupOptions):
        # Rolling window plan node saves the partition_by column, but
        # only for the purposes of stashing it for the optimiser. We
        # should ignore it here.
        key_columns = None
    else:
        key_columns = evaluate_expr(expr.partition_by, context, visitor)
    index_column = context[expr.options.index_column]
    out_cols = _rolling(
        index_column, col, expr.options.period, requests, key_columns
    )
    return _post_aggregate(
        [out_cols], [expr.function], aggs_to_replace, visitor
    )


@evaluate_expr.register
def _alias(expr: expr_nodes.Alias, context: DataFrame, visitor: ExprVisitor):
    result = evaluate_expr(visitor.node(expr.expr), context, visitor)
    (old_name,) = result.names()
    return result.rename({old_name: expr.name})


@evaluate_expr.register
def _literal(
    expr: expr_nodes.Literal, context: DataFrame, visitor: ExprVisitor
):
    # TODO: This is bad because it's lying about the dataframe property
    dtype = to_cudf_dtype(expr.dtype)
    value = expr.value
    return DataFrame(
        {"literal": cudf.Scalar(value, dtype)}  # type: ignore
    )  # type: ignore


@evaluate_expr.register
def _sort(expr: expr_nodes.Sort, context: DataFrame, visitor: ExprVisitor):
    to_sort = evaluate_expr(visitor.node(expr.expr), context, visitor)
    (name,) = to_sort.names()
    (stable, nulls_last, descending) = expr.options
    descending, column_order, null_precedence = sort_order(
        [descending], nulls_last=nulls_last, num_keys=1
    )
    do_sort = plc.sorting.stable_sort if stable else plc.sorting.sort
    result = do_sort(to_sort.to_pylibcudf(), column_order, null_precedence)
    flag = (
        DataFrame.IsSorted.DESCENDING
        if descending
        else DataFrame.IsSorted.ASCENDING
    )
    return DataFrame.from_pylibcudf(to_sort.names(), result).set_sorted(
        {name: flag}
    )


@evaluate_expr.register
def _sort_by(
    expr: expr_nodes.SortBy, context: DataFrame, visitor: ExprVisitor
):
    to_sort = evaluate_expr(visitor.node(expr.expr), context, visitor)
    descending = expr.descending
    sort_keys = evaluate_expr(expr.by, context, visitor)
    # TODO: no stable to sort_by in polars
    descending, column_order, null_precedence = sort_order(
        descending, nulls_last=True, num_keys=len(sort_keys)
    )
    result = plc.sorting.sort_by_key(
        to_sort.to_pylibcudf(),
        sort_keys.to_pylibcudf(),
        column_order,
        null_precedence,
    )
    return DataFrame.from_pylibcudf(to_sort.keys(), result)


@evaluate_expr.register
def _gather(expr: expr_nodes.Gather, context: DataFrame, visitor: ExprVisitor):
    # TODO: (maybe) the libcudf call is better as a table gather,
    # rather than a sequence of column gathers.
    # However, polars delivers a list of single-column gather
    # expressions, so we would need to traverse lists to potentially
    # identify gathers that could be agglomerated.
    # Since libcudf just turns a table gather into a sequence of
    # column gathers it may be useful instead just to rely on
    # stream-ordered launching to overlap things.
    if expr.scalar:
        raise NotImplementedError("scalar gather")
    result = evaluate_expr(visitor.node(expr.expr), context, visitor)
    (indices,) = evaluate_expr(
        visitor.node(expr.idx), context, visitor
    ).columns()
    # TODO: check out of bounds
    return result.gather(
        indices, bounds_policy=plc.copying.OutOfBoundsPolicy.DONT_CHECK
    )


@evaluate_expr.register
def _filter(expr: expr_nodes.Filter, context: DataFrame, visitor: ExprVisitor):
    result = evaluate_expr(visitor.node(expr.input), context, visitor)
    (mask,) = evaluate_expr(visitor.node(expr.by), context, visitor).columns()
    return result.filter(mask)


# TODO: in unoptimized plans sometimes the cast doesn't appear?
# Do we need to handle it in schemas?
@evaluate_expr.register
def _cast(expr: expr_nodes.Cast, context: DataFrame, visitor: ExprVisitor):
    context = evaluate_expr(visitor.node(expr.expr), context, visitor)
    dtype = to_pylibcudf_dtype(expr.dtype)
    return DataFrame(
        (
            name,
            plc.unary.cast(column, dtype),
        )
        for name, column in context.items()
    )


@evaluate_expr.register
def _column(expr: expr_nodes.Column, context: DataFrame, visitor: ExprVisitor):
    return DataFrame({expr.name: context[expr.name]})


@evaluate_expr.register
def _agg(expr: expr_nodes.Agg, context: DataFrame, visitor: ExprVisitor):
    name = expr.name
    result = evaluate_expr(visitor.node(expr.arguments), context, visitor)
    ((colname, column),) = result.items()
    # TODO: handle options
    options = expr.options

    if name in {"min", "max"}:
        # options is bool, propagate_nans
        if not options:
            column = plc.stream_compaction.drop_nulls(
                plc.Table([column]), [0], column.size()
            ).columns()[0]
        res = plc.reduce.reduce(
            column, getattr(plc.aggregation, name)(), column.type()
        )
        return DataFrame({colname: plc.Column.from_scalar(res, 1)})
    elif name in {"median", "mean", "sum"}:
        # polars always ignores nulls
        column = plc.stream_compaction.drop_nulls(
            plc.Table([column]), [0], column.size()
        ).columns()[0]
        res = plc.reduce.reduce(
            column, getattr(plc.aggregation, name)(), column.type()
        )
        return DataFrame({colname: plc.Column.from_scalar(res, 1)})
    elif name == "nunique":
        return DataFrame(
            {
                colname: plc.Column.from_scalar(
                    # TODO: explicit datatype
                    plc.interop.from_arrow(
                        pa.scalar(
                            plc.stream_compaction.distinct_count(
                                column,
                                plc.types.NullPolicy.EXCLUDE,
                                plc.types.NanPolicy.NAN_IS_VALID,
                            )
                        )
                    ),
                    1,
                )
            }
        )
    elif name == "first":
        return result.slice(0, 1)
    elif name == "last":
        return result.slice(-1, 1)
    elif name == "count":
        include_null = options
        return DataFrame(
            {
                colname: plc.Column.from_scalar(
                    plc.interop.from_arrow(
                        pa.scalar(
                            column.size()
                            - (0 if include_null else column.null_count())
                        )
                    ),
                    1,
                )
            }
        )
    elif name in {"std", "var"}:
        ddof = options
        # TODO: nan handling is wrong (?) in cudf?
        return DataFrame(
            {
                colname: plc.Column.from_scalar(
                    plc.reduce.reduce(
                        column,
                        getattr(plc.aggregation, name)(ddof=ddof),
                        column.type(),
                    ),
                    1,
                )
            }
        )
    else:
        raise NotImplementedError(f"Haven't implemented aggregation {name=}")


BINOP_MAPPING = {
    # (binop, result_dtype) # => None means same as input
    expr_nodes.PyOperator.Eq: (plc.binaryop.BinaryOperator.EQUAL, np.bool_),
    expr_nodes.PyOperator.EqValidity: (
        plc.binaryop.BinaryOperator.NULL_EQUALS,
        np.bool_,
    ),
    expr_nodes.PyOperator.NotEq: (
        plc.binaryop.BinaryOperator.NOT_EQUAL,
        np.bool_,
    ),
    # expr_nodes.PyOperator.NotEqValidity: (plc.binaryop.BinaryOperator., None),
    expr_nodes.PyOperator.Lt: (plc.binaryop.BinaryOperator.LESS, np.bool_),
    expr_nodes.PyOperator.LtEq: (
        plc.binaryop.BinaryOperator.LESS_EQUAL,
        np.bool_,
    ),
    expr_nodes.PyOperator.Gt: (plc.binaryop.BinaryOperator.GREATER, np.bool_),
    expr_nodes.PyOperator.GtEq: (
        plc.binaryop.BinaryOperator.GREATER_EQUAL,
        np.bool_,
    ),
    expr_nodes.PyOperator.Plus: (plc.binaryop.BinaryOperator.ADD, None),
    expr_nodes.PyOperator.Minus: (plc.binaryop.BinaryOperator.SUB, None),
    expr_nodes.PyOperator.Multiply: (plc.binaryop.BinaryOperator.MUL, None),
    expr_nodes.PyOperator.Divide: (plc.binaryop.BinaryOperator.DIV, None),
    expr_nodes.PyOperator.TrueDivide: (
        plc.binaryop.BinaryOperator.TRUE_DIV,
        None,
    ),
    expr_nodes.PyOperator.FloorDivide: (
        plc.binaryop.BinaryOperator.FLOOR_DIV,
        None,
    ),
    expr_nodes.PyOperator.Modulus: (plc.binaryop.BinaryOperator.PYMOD, None),
    expr_nodes.PyOperator.And: (plc.binaryop.BinaryOperator.BITWISE_AND, None),
    expr_nodes.PyOperator.Or: (plc.binaryop.BinaryOperator.BITWISE_OR, None),
    expr_nodes.PyOperator.Xor: (plc.binaryop.BinaryOperator.BITWISE_XOR, None),
    expr_nodes.PyOperator.LogicalAnd: (
        plc.binaryop.BinaryOperator.LOGICAL_AND,
        None,
    ),
    expr_nodes.PyOperator.LogicalOr: (
        plc.binaryop.BinaryOperator.LOGICAL_OR,
        None,
    ),
}


def _as_plc(val):
    return (
        val.device_value.c_value
        if isinstance(val, cudf.Scalar)
        else val.c_value
        if isinstance(val, libcudf.scalar.DeviceScalar)
        else val
    )


@evaluate_expr.register
def _binop(
    expr: expr_nodes.BinaryExpr, context: DataFrame, visitor: ExprVisitor
):
    left = evaluate_expr(visitor.node(expr.left), context, visitor)
    op = expr.op
    right = evaluate_expr(visitor.node(expr.right), context, visitor)
    (lop,) = left.columns()
    (rop,) = right.columns()
    # TODO: Fix dtype logic below to not be mixing pylibcudf and cudf types.
    # Probably easiest after we've fully switched over scalars too.
    try:
        op, dtype = BINOP_MAPPING[op]
        left_dtype = (
            dtype_from_pylibcudf_column(lop)
            if isinstance(lop, plc.Column)
            else lop.dtype
        )
        right_dtype = (
            dtype_from_pylibcudf_column(rop)
            if isinstance(rop, plc.Column)
            else rop.dtype
        )
        if op == plc.binaryop.BinaryOperator.TRUE_DIV:
            if left_dtype.kind == right_dtype.kind == "i":
                dtype = "float64"
        dtype = dtype_to_pylibcudf_type(dtype or left_dtype)
    except KeyError as err:
        raise NotImplementedError(f"Unhandled binop {op=}") from err
    return DataFrame.from_pylibcudf(
        left.names(),
        plc.Table(
            [
                plc.binaryop.binary_operation(
                    _as_plc(lop), _as_plc(rop), op, dtype
                )
            ]
        ),
    )


# Aggregations, need to be shared between plan and expression
# evaluation, but circular dep, so we put them here.
# TODO: document approach here properly
def agg_depth(agg, visitor: ExprVisitor) -> int:
    """
    Determine the depth of aggregations in an expression.

    Parameters
    ----------
    agg
        Expression containing aggregations
    visitor
        Callback visitor

    Returns
    -------
    Depth in the expression tree that an aggregation request was observed.

    Raises
    ------
    NotImplementedError
        If an aggregation request is nested inside another aggregation
        request, or an unhandled expression is seen.
    """
    agg = visitor.node(agg)
    if isinstance(agg, expr_nodes.Column):
        return 0
    elif isinstance(agg, expr_nodes.Alias):
        return agg_depth(agg.expr, visitor)
    elif isinstance(agg, expr_nodes.BinaryExpr):
        ldepth = agg_depth(agg.left, visitor)
        rdepth = agg_depth(agg.right, visitor)
        maxdepth = max(ldepth, rdepth)
        assert ldepth == rdepth
        return maxdepth
    elif isinstance(agg, expr_nodes.Len):
        return 1
    elif isinstance(agg, expr_nodes.Agg):
        # TODO: currently only singleton arguments (that's all that
        # the Agg object has right now)
        depth = agg_depth(agg.arguments, visitor)
        if depth >= 1:
            raise NotImplementedError("Nesting aggregations not supported")
        return depth + 1
    else:
        raise NotImplementedError(f"Unhandled agg {agg=}")


# TODO: would love to find a way to avoid the multiple traversal
# right now, we must run agg_depth first.
def collect_agg(
    node: int, context: DataFrame, depth: int, visitor: ExprVisitor
) -> tuple[
    list[ColumnType | None], list[tuple[plc.aggregation.Aggregation, int]], str
]:
    """
    Collect the aggregation requirements of a single aggregation request.

    Returns
    -------
    tuple of list of columns, list of (libcudf-agg-name, agg-expression) pairs,
    name of the final output

    Notes
    -----
    The aggregation expression is returned because we do a post-pass
    on the list of aggregations to evaluate with the new
    aggregation-enabled dataframe context.
    """
    agg = visitor.node(node)
    if isinstance(agg, expr_nodes.Column):
        return (
            [context[agg.name]],
            [(plc.aggregation.collect_list(), node)],
            agg.name,
        )
    elif isinstance(agg, expr_nodes.Alias):
        col, req, _ = collect_agg(agg.expr, context, depth, visitor)
        return col, req, agg.name
    elif isinstance(agg, expr_nodes.Len):
        return (
            [placeholder_column(context.num_rows())],
            [
                (
                    plc.aggregation.count(
                        null_handling=plc.types.NullPolicy.INCLUDE
                    ),
                    node,
                )
            ],
            "len",
        )
    elif isinstance(agg, expr_nodes.Agg):
        if depth > 0:
            raise NotImplementedError("Nested aggregations not yet supported")
        request = agg.name
        column, _, name = collect_agg(
            agg.arguments, context, depth + 1, visitor
        )
        if request == "agg_groups":
            # TODO: libcudf supports a ROW_NUMBER aggregation but it
            # is not exposed in python and is not available for
            # groupby aggregations.
            column = [
                as_column(
                    range(context.num_rows()),
                    dtype=libcudf.types.size_type_dtype,
                ).to_pylibcudf(mode="read")
            ]
            request = plc.aggregation.collect_list()
        elif request == "implode":
            raise NotImplementedError("implode in groupby not implemented")
        elif request == "count":
            request = plc.aggregation.count(
                null_handling=plc.types.NullPolicy.INCLUDE
                if agg.options
                else plc.types.NullPolicy.EXCLUDE
            )
        else:
            # TODO: ensure all options are handled correctly
            request = getattr(plc.aggregation, request)()
        return column, [(request, node)], name
    elif isinstance(agg, expr_nodes.BinaryExpr):
        # TODO: no nested agg(binop(agg)) right now
        if depth == 0:
            # Not inside an aggregation yet
            lcol, lreq, lname = collect_agg(agg.left, context, depth, visitor)
            rcol, rreq, _ = collect_agg(agg.right, context, depth, visitor)
            # Name of binop result comes from name of left child
            return [*lcol, *rcol], [*lreq, *rreq], lname
        else:
            # TODO: Inside an aggregation, this needs to disallow (for now)
            # seeing an aggregation request.
            ((name, column),) = evaluate_expr(agg, context, visitor).items()
            return [column], [(plc.aggregation.collect_list(), node)], name
    else:
        raise NotImplementedError


def collect_aggs(
    agg_exprs: list[int], context: DataFrame, visitor: ExprVisitor
) -> tuple[
    list[ColumnType | None],
    list[list[plc.aggregation.Aggregation]],
    list[list[list[int]]],
    list[str],
]:
    """
    Collect all the unique aggregation requests.

    Parameters
    ----------
    agg_exprs
        list of aggregation requests (expression node indices)
    context
        DataFrame that maps columns to data
    visitor
        Callback visitor

    Returns
    -------
    list of columns to aggregate, list of list of aggregation
    requests per column, list of mappings from aggregation requests
    to list of aggregation expressions that match that request/column
    pair, list of result column names.
    """
    groups: dict[
        int, tuple[ColumnType | None, list[str], dict[str, Expr]]
    ] = {}
    names: list[str] = []
    # TODO: ugly
    for columns, requests, name in (
        collect_agg(agg, context, 0, visitor) for agg in agg_exprs
    ):
        # Gather aggregation requests by the column they are operating
        # on. We use the id of the column object as the key.
        for column, (request, agg) in zip(columns, requests):
            colid = id(column)
            if colid in groups:
                _, column_requests, to_replace = groups[colid]
            else:
                _, column_requests, to_replace = groups.setdefault(
                    colid, (column, [], defaultdict(list))
                )
            # We're only going to ask libcudf for unique aggregation requests
            if request not in column_requests:
                column_requests.append(request)
            # But we need to record all the aggregation expressions
            to_replace[request].append(agg)
        names.append(name)
    raw_columns, raw_requests, aggs_to_replace = list(
        map(list, zip(*groups.values()))
    )
    return (
        raw_columns,
        raw_requests,
        [list(a.values()) for a in aggs_to_replace],
        names,
    )


def _post_aggregate(
    raw_columns: list[list[ColumnType]],
    aggs: list[int],
    aggs_to_replace: list[list[list[int]]],
    visitor: ExprVisitor,
) -> DataFrame:
    # rewrite the agg expression tree to replace agg requests with
    # the performed leaf aggregations and evaluate the resulting
    # expression to handle any expression-based stuff.
    # We can only handle pointwise expressions in this manner, but
    # collect_aggs checks for that.
    context: dict[str, ColumnType] = {}
    mapping: list[tuple[int, Expr]] = []
    for cols, agg_exprs in zip(raw_columns, aggs_to_replace, strict=True):
        for col, agg_expr in zip(cols, agg_exprs, strict=True):
            # Here's the placeholder that will replace the agg request
            name = f"tempcol{len(mapping)}"
            newcol = plrs.col(name)
            context[name] = col
            for agg in agg_expr:
                mapping.append((agg, newcol))
    return evaluate_expr(
        aggs, DataFrame(context), visitor.with_replacements(mapping)
    )


def _rolling(
    index_column: ColumnType,
    input_column: ColumnType,
    period: tuple,
    aggs: list[str],
    keys: DataFrame | None = None,
) -> list[ColumnType]:
    # first, compute the windows:
    months, weeks, days, nanoseconds, parsed_int = period
    if months:
        raise NotImplementedError("Months in window not supported")
    if parsed_int:
        raise NotImplementedError("Int as window not supported")
    offset = (
        np.timedelta64(weeks, "W")
        + np.timedelta64(days, "D")
        + np.timedelta64(nanoseconds, "ns")
    )
    input_column = input_column
    if keys is not None:
        # grouped rolling window
        grouper = plc.groupby.GroupBy(keys.to_pylibcudf())
        group_starts, _, _ = grouper.get_groups()
        group_starts = np.asarray(group_starts)
        group_starts = group_starts[:-1].repeat(np.diff(group_starts))
        pre_column_window = as_column(
            cudautils.grouped_window_sizes_from_offset(
                libcudf.column.Column.from_pylibcudf(
                    index_column
                ).data_array_view(mode="write"),
                as_column(group_starts),
                offset,
            )
        )
        fwd_column_window = as_column(
            0, length=pre_column_window.size, dtype=pre_column_window.dtype
        ).to_pylibcudf(mode="read")
        pre_column_window = pre_column_window.to_pylibcudf(mode="read")
    else:
        # regular rolling window
        pre_column_window = as_column(
            cudautils.window_sizes_from_offset(
                libcudf.column.Column.from_pylibcudf(
                    index_column
                ).data_array_view(mode="write"),
                offset,
            )
        )
        fwd_column_window = as_column(
            0, length=pre_column_window.size, dtype=pre_column_window.dtype
        ).to_pylibcudf(mode="read")
        pre_column_window = pre_column_window.to_pylibcudf(mode="read")

    # perform windowed aggregation for each column:
    return [
        plc.rolling.rolling_window(
            input_column,
            pre_column_window,
            fwd_column_window,
            0,
            agg,
        )
        for agg in aggs
    ]
