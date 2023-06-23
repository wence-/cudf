# Copyright (c) 2023, NVIDIA CORPORATION.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
from typing_extensions import TypeAlias

import cudf
from cudf.api.types import (
    _is_non_decimal_numeric_dtype,
    _is_scalar_or_zero_d_array,
    is_bool_dtype,
    is_integer,
    is_integer_dtype,
)
from cudf.core import copy_types as ct
from cudf.utils.dtypes import to_cudf_compatible_scalar


# Poor man's algebraic data types
class EmptyIndexer:
    """An indexer that will produce an empty result"""

    pass


@dataclass
class MapIndexer:
    """An indexer for a gather map"""

    gather_map: ct.GatherMap


@dataclass
class MaskIndexer:
    """An indexer for a boolean mask"""

    mask: ct.BooleanMask


@dataclass
class SliceIndexer:
    """An indexer for a slice"""

    slice: slice


@dataclass
class ScalarIndexer:
    """An indexer for a scalar value"""

    gather_map: ct.GatherMap


IndexingSpec: TypeAlias = (
    EmptyIndexer | MapIndexer | MaskIndexer | ScalarIndexer | SliceIndexer
)

ColumnLabels: TypeAlias = List[str]


def destructure_iloc_key(
    key: Any, frame: cudf.Series | cudf.DataFrame
) -> tuple[Any, ...]:
    """
    Destructure a potentially tuple-typed key into row and column indexers

    Tuple arguments to iloc indexing are treated specially. They are
    picked apart into indexers for the row and column. If the number
    of entries is less than the number of modes of the frame, missing
    entries are slice-expanded.

    If the user-provided key is not a tuple, it is treated as if it
    were a singleton tuple, and then slice-expanded.

    Once this destructuring has occurred, any entries that are
    callables are then called with the indexed frame. This should
    return a valid indexing object for the rows (respectively
    columns), namely one of:

    - A boolean mask of the same length as the frame in the given
      dimension
    - A scalar integer that indexes the frame
    - An array-like of integers that index the frame
    - A slice that indexes the frame

    Integer and slice-based indexing follows usual Python conventions.

    Parameters
    ----------
    key
        The key to destructure
    frame
        DataFrame or Series to provide context

    Returns
    -------
    tuple of indexers with length equal to the dimension of the frame

    Raises
    ------
    IndexError
        If there are too many indexers, or any individual indexer is a tuple.
    """
    n = len(frame.shape)
    if isinstance(key, tuple):
        # Key potentially indexes rows and columns, slice-expand to
        # shape of frame
        indexers = key + (slice(None),) * (n - len(key))
        if (ni := len(indexers)) > n:
            raise IndexError(f"Too many indexers: got {ni} expected {n}")
    else:
        # Key indexes rows, slice-expand to shape of frame
        indexers = (key, *(slice(None),) * (n - 1))
    indexers = tuple(k(frame) if callable(k) else k for k in indexers)
    if any(isinstance(k, tuple) for k in indexers):
        raise IndexError(
            "Too many indexers: can't have nested tuples in iloc indexing"
        )
    return indexers


def destructure_dataframe_iloc_indexer(
    key: Any, frame: cudf.DataFrame
) -> Tuple[Any, Tuple[bool, ColumnLabels]]:
    """Destructure an index key for DataFrame iloc getitem.

    Parameters
    ----------
    key
        Key to destructure
    frame
        DataFrame to provide context context

    Returns
    -------
    tuple
        2-tuple of a key for the rows and tuple of
        (column_index_is_scalar, column_names) for the columns

    Raises
    ------
    TypeError
        If the column indexer is invalid
    IndexError
        If the provided key does not destructure correctly
    NotImplementedError
        If the requested column indexer repeats columns
    """
    rows, cols = destructure_iloc_key(key, frame)
    if cols is Ellipsis:
        cols = slice(None)
    scalar = is_integer(cols)
    try:
        column_names: ColumnLabels = list(
            frame._data.get_labels_by_index(cols)
        )
        if len(set(column_names)) != len(column_names):
            raise NotImplementedError(
                "cudf DataFrames do not support repeated column names"
            )
    except TypeError:
        raise TypeError(
            "Column indices must be integers, slices, "
            "or list-like of integers"
        )
    if scalar:
        assert (
            len(column_names) == 1
        ), "Scalar column indexer should not produce more than one column"

    return (rows, (scalar, column_names))


def destructure_series_iloc_indexer(key: Any, frame: cudf.Series) -> Any:
    """Destructure an index key for Series iloc getitem.

    Parameters
    ----------
    key
        Key to destructure
    frame
        Series for unpacking context

    Returns
    -------
    Single key that will index the rows
    """
    (rows,) = destructure_iloc_key(key, frame)
    return rows


def parse_row_iloc_indexer(key: Any, n: int, *, check_bounds) -> IndexingSpec:
    """
    Normalize and produce structured information about a row indexer

    Given a row indexer that has already been destructured by
    :func:`destructure_iloc_key`, inspect further and produce structured
    information for indexing operations to act upon.

    Parameters
    ----------
    key
        Suitably destructured key for row indexing
    n
        Length of frame to index
    check_bounds
        If True, perform bounds checking of the key if it is a gather
        map.

    Returns
    -------
    IndexingSpec
        Structured data for indexing. A tag + parsed data.

    Raises
    ------
    IndexError
        If a valid type of indexer is provided, but it is out of
        bounds
    TypeError
        If the indexing key is otherwise invalid.
    """
    if key is Ellipsis:
        return SliceIndexer(slice(None))
    elif isinstance(key, slice):
        return SliceIndexer(key)
    elif _is_scalar_or_zero_d_array(key):
        return ScalarIndexer(
            ct.as_gather_map(key, n, nullify=False, check_bounds=check_bounds)
        )
    else:
        key = cudf.core.column.as_column(key)
        if isinstance(key, cudf.core.column.CategoricalColumn):
            key = key.as_numerical_column(key.codes.dtype)
        if is_bool_dtype(key.dtype):
            return MaskIndexer(ct.as_boolean_mask(key, n))
        elif len(key) == 0:
            return EmptyIndexer()
        elif is_integer_dtype(key.dtype):
            return MapIndexer(
                ct.as_gather_map(
                    key, n, nullify=False, check_bounds=check_bounds
                )
            )
        else:
            raise TypeError(
                "Cannot index by location "
                f"with non-integer key of type {type(key)}"
            )


@dataclass
class ScalarValue:
    value: cudf.Scalar


@dataclass
class ColumnValue:
    value: cudf.core.column.ColumnBase


ValueSpec: TypeAlias = ScalarValue | ColumnValue


def parse_series_iloc_value(
    key: IndexingSpec, value: Any, n: int, series_type: Any
) -> ValueSpec:
    if isinstance(value, list) and isinstance(series_type, cudf.ListDtype):
        # No type-casting for lists
        try:
            return ScalarValue(cudf.Scalar(value, dtype=series_type))
        except TypeError:
            column = cudf.core.column.as_column(value)
            if column.dtype != series_type:
                raise ValueError()
            return ColumnValue(column)
    elif isinstance(value, dict) and isinstance(series_type, cudf.StructDtype):
        # Or for floats
        scalar = cudf.Scalar(value)
        if scalar.dtype != series_type:
            raise ValueError()
        return ScalarValue(scalar)
    if _is_scalar_or_zero_d_array(value):
        value = to_cudf_compatible_scalar(value)
        constructor = ScalarValue
    else:
        value = cudf.core.column.as_column(value)
        constructor = ColumnValue
    value_type = value.dtype
    if _is_non_decimal_numeric_dtype(
        value_type
    ) and _is_non_decimal_numeric_dtype(series_type):
        result_type = np.result_type(value_type, series_type)
        value = value.astype(result_type)
    return constructor(value)
