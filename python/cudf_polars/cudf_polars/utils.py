# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import types

import cudf
import cudf._lib.pylibcudf as plc
import polars as pl


def placeholder_column(n: int):
    """
    Produce a placeholder pylibcudf column with NO BACKING DATA.

    Parameters
    ----------
    n
        Number of rows the column will advertise

    Returns
    -------
    pylibcudf Column that is almost unusable. DO NOT ACCESS THE DATA BUFFER.

    Notes
    -----
    This is used to avoid allocating data for count aggregations.
    """
    return plc.Column(
        plc.DataType(plc.TypeId.INT8),
        n,
        plc.gpumemoryview(
            types.SimpleNamespace(__cuda_array_interface__={"data": (1, True)})
        ),
        None,
        0,
        0,
        [],
    )


def sort_order(
    descending: list[bool], *, nulls_last: bool, num_keys: int
) -> tuple[list[bool], list[plc.types.Order], list[plc.types.NullOrder]]:
    """
    Produce sort order arguments.

    Parameters
    ----------
    descending
        List indicating order for each column
    nulls_last
        Should nulls sort last or first?
    num_keys
        Number of sort keys

    Returns
    -------
    tuple of broadcast descending, column_order and null_precendence
    suitable for passing to sort routines
    """
    # Mimicking polars broadcast handling of descending
    if num_keys > (n := len(descending)) and n == 1:
        descending = [descending[0]] * num_keys
    column_order = [
        plc.types.Order.DESCENDING if d else plc.types.Order.ASCENDING
        for d in descending
    ]
    null_precedence = []
    for asc in column_order:
        if (asc == plc.types.Order.ASCENDING) ^ (not nulls_last):
            null_precedence.append(plc.types.NullOrder.AFTER)
        elif (asc == plc.types.Order.ASCENDING) ^ nulls_last:
            null_precedence.append(plc.types.NullOrder.BEFORE)
    return descending, column_order, null_precedence


PLC_TYPE_MAP = {
    pl.Int8: plc.types.DataType(plc.types.TypeId.INT8),
    pl.Int16: plc.types.DataType(plc.types.TypeId.INT16),
    pl.Int32: plc.types.DataType(plc.types.TypeId.INT32),
    pl.Int64: plc.types.DataType(plc.types.TypeId.INT64),
    pl.UInt8: plc.types.DataType(plc.types.TypeId.UINT8),
    pl.UInt16: plc.types.DataType(plc.types.TypeId.UINT16),
    pl.UInt32: plc.types.DataType(plc.types.TypeId.UINT32),
    pl.UInt64: plc.types.DataType(plc.types.TypeId.UINT64),
    pl.Float32: plc.types.DataType(plc.types.TypeId.FLOAT32),
    pl.Float64: plc.types.DataType(plc.types.TypeId.FLOAT64),
    pl.Boolean: plc.types.DataType(plc.types.TypeId.BOOL8),
    pl.String: plc.types.DataType(plc.types.TypeId.STRING),
}


def to_pylibcudf_dtype(dtype):
    """
    Convert a polars dtype to a pylibcudf dtype.

    Parameters
    ----------
    dtype
         Polars dtype to convert

    Returns
    -------
    Matching pylibcudf data type.
    """
    if isinstance(dtype, pl.Datetime):
        unit = dtype.time_unit
        if dtype.time_zone is not None:
            raise NotImplementedError("time zone")
        if unit == "ms":
            return plc.types.DataType(plc.types.TypeId.TIMESTAMP_MILLISECONDS)
        elif unit == "us":
            return plc.types.DataType(plc.types.TypeId.TIMESTAMP_MICROSECONDS)
        elif unit == "ns":
            return plc.types.DataType(plc.types.TypeId.TIMESTAMP_NANOSECONDS)
        else:
            raise NotImplementedError(unit)
    else:
        try:
            return PLC_TYPE_MAP[type(dtype)]
        except KeyError as e:
            raise NotImplementedError(type(dtype)) from e


CUDF_TYPE_MAP = {
    pl.Int8: cudf.dtype("int8"),
    pl.Int16: cudf.dtype("int16"),
    pl.Int32: cudf.dtype("int32"),
    pl.Int64: cudf.dtype("int64"),
    pl.UInt8: cudf.dtype("uint8"),
    pl.UInt16: cudf.dtype("uint16"),
    pl.UInt32: cudf.dtype("uint32"),
    pl.UInt64: cudf.dtype("uint64"),
    pl.Float32: cudf.dtype("float32"),
    pl.Float64: cudf.dtype("float64"),
    pl.Boolean: cudf.dtype("bool"),
    pl.String: cudf.dtype("O"),
}


def to_cudf_dtype(dtype):
    """
    Convert a polars dtype to a cudf dtype.

    Parameters
    ----------
    dtype
         Polars dtype to convert

    Returns
    -------
    Matching cudf data type.
    """
    if isinstance(dtype, pl.Datetime):
        if dtype.time_zone is not None:
            raise NotImplementedError("time zone")
        return cudf.dtype(f"datetime64[{dtype.time_unit}]")
    else:
        try:
            return CUDF_TYPE_MAP[type(dtype)]
        except KeyError as e:
            raise NotImplementedError(type(dtype)) from e
