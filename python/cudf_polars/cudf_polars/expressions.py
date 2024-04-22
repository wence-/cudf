# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from functools import singledispatch
from typing import TYPE_CHECKING

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


class ExprVisitor:
    """Object holding rust visitor and utility methods."""

    __slots__ = ("visitor", "in_groupby")
    visitor: Visitor
    in_groupby: bool

    def __init__(self, visitor: Visitor):
        self.visitor = visitor
        self.in_groupby = False

    def __call__(self, node: int, context: DataFrame) -> ColumnType:
        """
        Return the evaluation of an expression node in a context.

        Parameters
        ----------
        node
            The node to evaluate
        context
            The dataframe providing context

        Returns
        -------
        New column as the evaluation of the expression.
        """
        return evaluate_expr(self.visitor.view_expression(node), context, self)

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
) -> ColumnType:
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
    Column representing result of evaluating the expression
    """
    raise AssertionError(f"Unhandled expression type {type(expr)}")


@evaluate_expr.register
def _expr_function(
    expr: expr_nodes.Function, context: DataFrame, visitor: ExprVisitor
):
    fname, *fargs = expr.function_data
    arguments = [visitor(e, context) for e in expr.input]
    if fname == "argwhere":
        (mask,) = arguments
        (indices,) = plc.stream_compaction.apply_boolean_mask(
            plc.Table(
                [
                    as_column(
                        range(mask.size()), dtype=libcudf.types.size_type_dtype
                    ).to_pylibcudf(mode="read")
                ]
            ),
            mask,
        ).columns()
        return indices
    elif fname == "setsorted":
        # TODO: tracking sortedness
        (column,) = arguments
        return column
        # (name,) = data.keys()
        # (flag,) = fargs
        # return data.set_sorted(
        #     {name: getattr(DataFrame.IsSorted, flag.upper())}
        # )
    elif fname == "is_not_null":
        (column,) = arguments
        return plc.unary.is_valid(column)
    else:
        raise NotImplementedError(f"Function expression {fname=}")


@evaluate_expr.register
def _expr_window(
    expr: expr_nodes.Window, context: DataFrame, visitor: ExprVisitor
):
    if isinstance(expr.options, expr_nodes.PyWindowMapping):
        raise NotImplementedError(".over() not supported")
    (col,), (requests,), aggs_to_replace = collect_aggs(
        [expr.function], context, visitor
    )
    if isinstance(expr.options, expr_nodes.PyRollingGroupOptions):
        # Rolling window plan node saves the partition_by column, but
        # only for the purposes of stashing it for the optimiser. We
        # should ignore it here.
        key_columns = None
    else:
        key_columns = [visitor(e, context) for e in expr.partition_by]
    index_column = context[expr.options.index_column]
    out_cols = _rolling(
        index_column, col, expr.options.period, requests, key_columns
    )
    (result,) = _post_aggregate(
        [out_cols], [expr.function], aggs_to_replace, visitor
    )
    return result


@evaluate_expr.register
def _alias(expr: expr_nodes.Alias, context: DataFrame, visitor: ExprVisitor):
    # TODO: optimizer should strip these from plan nodes
    return visitor(expr.expr, context)


@evaluate_expr.register
def _literal(
    expr: expr_nodes.Literal, context: DataFrame, visitor: ExprVisitor
):
    # TODO: This is bad because it's lying about the Column property
    dtype = to_cudf_dtype(expr.dtype)
    value = expr.value
    return cudf.Scalar(value, dtype)  # type: ignore


@evaluate_expr.register
def _sort(expr: expr_nodes.Sort, context: DataFrame, visitor: ExprVisitor):
    if visitor.in_groupby:
        raise NotImplementedError("sort inside groupby")
    to_sort = visitor(expr.expr, context)
    (stable, nulls_last, descending) = expr.options
    descending, column_order, null_precedence = sort_order(
        [descending], nulls_last=nulls_last, num_keys=1
    )
    do_sort = plc.sorting.stable_sort if stable else plc.sorting.sort
    result = do_sort(to_sort.to_pylibcudf(), column_order, null_precedence)
    return result
    # TODO: track sortedness
    # flag = (
    #     DataFrame.IsSorted.DESCENDING
    #     if descending
    #     else DataFrame.IsSorted.ASCENDING
    # )
    # return DataFrame.from_pylibcudf(to_sort.names(), result).set_sorted(
    #     {name: flag}
    # )


@evaluate_expr.register
def _sort_by(
    expr: expr_nodes.SortBy, context: DataFrame, visitor: ExprVisitor
):
    if visitor.in_groupby:
        raise NotImplementedError("sort_by inside groupby")
    to_sort = visitor(expr.expr, context)
    descending = expr.descending
    sort_keys = [visitor(e, context) for e in expr.by]
    # TODO: no stable to sort_by in polars
    descending, column_order, null_precedence = sort_order(
        descending, nulls_last=True, num_keys=len(sort_keys)
    )
    return plc.sorting.sort_by_key(
        plc.Table([to_sort]),
        plc.Table(sort_keys),
        column_order,
        null_precedence,
    )


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
    result = visitor(expr.expr, context)
    indices = visitor(expr.idx, context)
    # TODO: check out of bounds
    (column,) = plc.copying.gather(
        plc.Table([result]),
        indices,
        bounds_policy=plc.copying.OutOfBoundsPolicy.DONT_CHECK,
    ).columns()
    return column


@evaluate_expr.register
def _filter(expr: expr_nodes.Filter, context: DataFrame, visitor: ExprVisitor):
    if visitor.in_groupby:
        raise NotImplementedError("filter inside groupby")
    result = visitor(expr.input, context)
    mask = visitor(expr.by, context)
    (column,) = plc.stream_compaction.apply_boolean_mask(
        plc.Table([result]), mask
    ).columns()
    return column


# TODO: in unoptimized plans sometimes the cast doesn't appear?
# Do we need to handle it in schemas?
@evaluate_expr.register
def _cast(expr: expr_nodes.Cast, context: DataFrame, visitor: ExprVisitor):
    column = visitor(expr.expr, context)
    dtype = to_pylibcudf_dtype(expr.dtype)
    return plc.unary.cast(column, dtype)


@evaluate_expr.register
def _column(expr: expr_nodes.Column, context: DataFrame, visitor: ExprVisitor):
    return context[expr.name]


@evaluate_expr.register
def _agg(expr: expr_nodes.Agg, context: DataFrame, visitor: ExprVisitor):
    if visitor.in_groupby:
        raise NotImplementedError("nested agg in group_by")
    name = expr.name
    column = visitor(expr.arguments, context)
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
        return plc.Column.from_scalar(res, 1)
    elif name in {"median", "mean", "sum"}:
        # polars always ignores nulls
        column = plc.stream_compaction.drop_nulls(
            plc.Table([column]), [0], column.size()
        ).columns()[0]
        res = plc.reduce.reduce(
            column, getattr(plc.aggregation, name)(), column.type()
        )
        return plc.Column.from_scalar(res, 1)
    elif name == "nunique":
        return plc.Column.from_scalar(
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
    elif name == "first":
        (column,) = plc.copying.slice(
            plc.Table([column]), [0, min(1, column.size())]
        ).columns()
        return column
    elif name == "last":
        (column,) = plc.copying.slice(
            plc.Table([column]), [max(column.size() - 1, 0), column.size()]
        ).columns()
        return column
    elif name == "count":
        include_null = options
        return plc.Column.from_scalar(
            plc.interop.from_arrow(
                pa.scalar(
                    column.size()
                    - (0 if include_null else column.null_count())
                )
            ),
            1,
        )
    elif name in {"std", "var"}:
        ddof = options
        # TODO: nan handling is wrong (?) in cudf?
        return plc.Column.from_scalar(
            plc.reduce.reduce(
                column,
                getattr(plc.aggregation, name)(ddof=ddof),
                column.type(),
            ),
            1,
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
    lop = visitor(expr.left, context)
    op = expr.op
    rop = visitor(expr.right, context)
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
    return plc.binaryop.binary_operation(_as_plc(lop), _as_plc(rop), op, dtype)


def collect_agg(
    node: int, context: DataFrame, depth: int, visitor: ExprVisitor
) -> tuple[
    list[ColumnType | None], list[tuple[plc.aggregation.Aggregation, int]]
]:
    """
    Collect the aggregation requirements of a single aggregation request.

    Parameters
    ----------
    node
        Node representing aggregation to collect
    context
        DataFrame context
    depth
        Depth of the aggregation in tree
    visitor
        Visitor for translating nodes

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
    agg = visitor.visitor.view_expression(node)
    if isinstance(agg, expr_nodes.Column):
        return (
            [context[agg.name]],
            [(plc.aggregation.collect_list(), node)],
        )
    elif isinstance(agg, expr_nodes.Alias):
        # TODO: should we see this?
        return collect_agg(agg.expr, context, depth, visitor)
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
        )
    elif isinstance(agg, expr_nodes.Agg):
        if depth > 0:
            raise NotImplementedError("Nested aggregations not yet supported")
        request = agg.name
        column, _ = collect_agg(agg.arguments, context, depth + 1, visitor)
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
        return column, [(request, node)]
    elif isinstance(agg, expr_nodes.BinaryExpr):
        # TODO: no nested agg(binop(agg)) right now
        if depth == 0:
            # Not inside an aggregation yet
            lcol, lreq = collect_agg(agg.left, context, depth, visitor)
            rcol, rreq = collect_agg(agg.right, context, depth, visitor)
            return [*lcol, *rcol], [*lreq, *rreq]
        else:
            # TODO: Ugly non-local method of saying "we're in a groupby, disallow"
            visitor.in_groupby = True
            column = evaluate_expr(agg, context, visitor)
            visitor.in_groupby = False
            return [column], [(plc.aggregation.collect_list(), node)]
    elif isinstance(agg, expr_nodes.Literal):
        # Scalar value, constant across the groups
        return [], []
    else:
        raise NotImplementedError


def collect_aggs(
    agg_exprs: list[int], context: DataFrame, visitor: ExprVisitor
) -> tuple[
    list[ColumnType | None],
    list[list[plc.aggregation.Aggregation]],
    list[list[list[int]]],
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
    # TODO: ugly
    for columns, requests in (
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
            if request is None:
                # Literals, which don't produce requests since they must be
                # uniform across the group
                continue
            # We're only going to ask libcudf for unique aggregation requests
            if request not in column_requests:
                column_requests.append(request)
            # But we need to record all the aggregation expressions
            to_replace[request].append(agg)
    raw_columns, raw_requests, aggs_to_replace = list(
        map(list, zip(*groups.values()))
    )
    return (
        raw_columns,
        raw_requests,
        [list(a.values()) for a in aggs_to_replace],
    )


def _post_aggregate(
    raw_columns: list[list[ColumnType]],
    aggs: list[int],
    aggs_to_replace: list[list[list[int]]],
    visitor: ExprVisitor,
) -> list[ColumnType]:
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
    context = DataFrame(context)
    v = visitor.with_replacements(mapping)
    return [v(agg, context) for agg in aggs]


def _rolling(
    index_column: ColumnType,
    input_column: ColumnType,
    period: tuple,
    aggs: list[str],
    keys: list[ColumnType] | None = None,
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
        grouper = plc.groupby.GroupBy(plc.Table(keys))
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
