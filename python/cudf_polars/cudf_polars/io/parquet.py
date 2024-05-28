# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Read parquet utilities."""

from __future__ import annotations

import datetime
from collections.abc import Hashable
from functools import singledispatch
from typing import TYPE_CHECKING, Callable, Generic, TypeVar, cast

import numpy as np

from polars.polars import _expr_nodes as pl_expr

import cudf
import cudf._lib as libcudf
import cudf._lib.pylibcudf as plc

from cudf_polars.dsl import expr

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping

    import polars.type_aliases as pl_types

__all__ = ["read_parquet", "as_libcudf_expr"]

U = TypeVar("U", bound=Hashable)
V = TypeVar("V")


class Memoizer(Generic[U, V]):
    def __init__(self, fn: Callable[[U, Callable[[U], V]], V]):
        self.cache: MutableMapping[Hashable, V] = {}
        self.fn = fn

    def __call__(self, node: U) -> V:
        try:
            return self.cache[node]
        except KeyError:
            return self.cache.setdefault(node, self.fn(node, self))


@singledispatch
def _as_libcudf_expr(
    node: expr.Expr, self: Memoizer[expr.Expr, libcudf.expressions.Expression]
) -> libcudf.expressions.Expression:
    raise NotImplementedError(f"{type(node)}")


@_as_libcudf_expr.register
def _(
    node: expr.Col, self: Memoizer[expr.Expr, libcudf.expressions.Expression]
) -> libcudf.expressions.Expression:
    return libcudf.expressions.ColumnNameReference(node.name.encode())


@_as_libcudf_expr.register
def _(
    node: expr.Literal, self: Memoizer[expr.Expr, libcudf.expressions.Expression]
) -> libcudf.expressions.Expression:
    val = node.value.as_py()
    if isinstance(val, datetime.datetime):
        val = np.datetime64(val, node.value.type.unit)
    elif isinstance(val, datetime.date):
        val = np.datetime64(val, "D")
    # TODO: pylibcudf wrappers that just accept a pylibcudf scalar.
    return libcudf.expressions.Literal(val)


_BINOP_MAPPING: Mapping[
    plc.binaryop.BinaryOperator, libcudf.expressions.ASTOperator
] = {
    plc.binaryop.BinaryOperator.EQUAL: libcudf.expressions.ASTOperator.EQUAL,
    plc.binaryop.BinaryOperator.NULL_EQUALS: libcudf.expressions.ASTOperator.NULL_EQUAL,
    plc.binaryop.BinaryOperator.NOT_EQUAL: libcudf.expressions.ASTOperator.NOT_EQUAL,
    plc.binaryop.BinaryOperator.LESS: libcudf.expressions.ASTOperator.LESS,
    plc.binaryop.BinaryOperator.LESS_EQUAL: libcudf.expressions.ASTOperator.LESS_EQUAL,
    plc.binaryop.BinaryOperator.GREATER: libcudf.expressions.ASTOperator.GREATER,
    plc.binaryop.BinaryOperator.GREATER_EQUAL: libcudf.expressions.ASTOperator.GREATER_EQUAL,
    plc.binaryop.BinaryOperator.ADD: libcudf.expressions.ASTOperator.ADD,
    plc.binaryop.BinaryOperator.SUB: libcudf.expressions.ASTOperator.SUB,
    plc.binaryop.BinaryOperator.MUL: libcudf.expressions.ASTOperator.MUL,
    plc.binaryop.BinaryOperator.DIV: libcudf.expressions.ASTOperator.DIV,
    plc.binaryop.BinaryOperator.TRUE_DIV: libcudf.expressions.ASTOperator.TRUE_DIV,
    plc.binaryop.BinaryOperator.FLOOR_DIV: libcudf.expressions.ASTOperator.FLOOR_DIV,
    plc.binaryop.BinaryOperator.PYMOD: libcudf.expressions.ASTOperator.PYMOD,
    plc.binaryop.BinaryOperator.BITWISE_AND: libcudf.expressions.ASTOperator.BITWISE_AND,
    plc.binaryop.BinaryOperator.BITWISE_OR: libcudf.expressions.ASTOperator.BITWISE_OR,
    plc.binaryop.BinaryOperator.BITWISE_XOR: libcudf.expressions.ASTOperator.BITWISE_XOR,
    plc.binaryop.BinaryOperator.LOGICAL_AND: libcudf.expressions.ASTOperator.LOGICAL_AND,
    plc.binaryop.BinaryOperator.LOGICAL_OR: libcudf.expressions.ASTOperator.LOGICAL_OR,
}


@_as_libcudf_expr.register
def _(
    node: expr.BinOp, self: Memoizer[expr.Expr, libcudf.expressions.Expression]
) -> libcudf.expressions.Expression:
    children = [self(c) for c in node.children]
    if node.op == plc.binaryop.BinaryOperator.NULL_NOT_EQUALS:
        # AST doesn't have this binop
        op = libcudf.expressions.ASTOperator.NULL_EQUAL
        value = libcudf.expressions.Operation(op, *children)
        self.cache[(node, node.op)] = value
        return libcudf.expressions.Operation(libcudf.expressions.ASTOperator.NOT, value)
    op = _BINOP_MAPPING[node.op]
    if all(c.dtype.id() == plc.TypeId.BOOL8 for c in node.children):
        if node.op == plc.binaryop.BinaryOperator.BITWISE_AND:
            op = libcudf.expressions.ASTOperator.LOGICAL_AND
        elif node.op == plc.binaryop.BinaryOperator.BITWISE_OR:
            op = libcudf.expressions.ASTOperator.LOGICAL_OR
    return libcudf.expressions.Operation(op, *children)


_BETWEEN_OPS: Mapping[
    pl_types.ClosedInterval,
    tuple[plc.binaryop.BinaryOperator, plc.binaryop.BinaryOperator],
] = {
    "none": (
        libcudf.expressions.ASTOperator.GREATER,
        libcudf.expressions.ASTOperator.LESS,
    ),
    "left": (
        libcudf.expressions.ASTOperator.GREATER_EQUAL,
        libcudf.expressions.ASTOperator.LESS,
    ),
    "right": (
        libcudf.expressions.ASTOperator.GREATER,
        libcudf.expressions.ASTOperator.LESS_EQUAL,
    ),
    "both": (
        libcudf.expressions.ASTOperator.GREATER_EQUAL,
        libcudf.expressions.ASTOperator.LESS_EQUAL,
    ),
}


@_as_libcudf_expr.register
def _(
    node: expr.BooleanFunction,
    self: Memoizer[expr.Expr, libcudf.expressions.Expression],
) -> libcudf.expressions.Expression:
    if node.name == pl_expr.BooleanFunction.IsBetween:
        column, lo, hi = map(self, node.children)
        (closed,) = node.options
        lop, rop = _BETWEEN_OPS[closed]
        left = libcudf.expressions.Operation(lop, column, lo)
        right = libcudf.expressions.Operation(rop, column, hi)
        self.cache[(node, lop)] = left
        self.cache[(node, rop)] = right
        return libcudf.expressions.Operation(
            libcudf.expressions.ASTOperator.LOGICAL_AND, left, right
        )
    raise NotImplementedError


def as_libcudf_expr(
    node: expr.Expr,
) -> tuple[libcudf.expression.Expression, list[libcudf.expression.Expression]]:
    """Convert an expression to libcudf AST expressions."""
    mapper = Memoizer(_as_libcudf_expr)
    result = mapper(node)
    return result, list(mapper.cache.values())


def read_parquet(
    paths: list[str],
    columns: list[str] | None,
    mask: libcudf.expression.Expression | None,
) -> cudf.DataFrame:
    """
    Read a parquet file and return a cudf DataFrame.

    Parameters
    ----------
    paths
        List of paths to parquet files
    columns
        Optional list of columns to select in the returned dataframe
    mask
        Optional row mask to apply during read.

    Return
    ------
    New cudf DataFrame

    Raises
    ------
    NotImplementedError if the optional mask cannot be translated to a
    libcudf AST expression.
    """
    return cast(
        cudf.DataFrame,
        libcudf.parquet.read_parquet(
            paths, columns=columns, use_pandas_metadata=False, filters=mask
        ),
    )
