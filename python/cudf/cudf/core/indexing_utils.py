# Copyright (c) 2023, NVIDIA CORPORATION.

from __future__ import annotations

import enum
import itertools
from typing import Any, Tuple, TypeAlias

import numpy as np

import cudf
import cudf._lib as libcudf
from cudf.api.types import (
    _is_scalar_or_zero_d_array,
    is_bool_dtype,
    is_integer_dtype,
)


class Indexer(enum.IntEnum):
    SLICE = enum.auto()
    MASK = enum.auto()
    INDICES = enum.auto()
    SCALAR = enum.auto()


# Oh for coproduct types.
IndexSpec: TypeAlias = Tuple[Indexer, Any]
ColumnLabels: TypeAlias = Tuple[str, ...]


def unpack_iloc_key(
    key: Any, frame: cudf.DataFrame | cudf.Series
) -> Tuple[Any, ...]:
    """Unpack a user-level key to iloc.__getitem__

    Parameters
    ----------
    key
        Key to unpack
    frame
        DataFrame or Series to provide context

    Returns
    -------
    tuple
        Tuple of row and (for dataframes) column keys

    Raises
    ------
    IndexError
        If provided a structurally invalid key
    """
    # This is more consistent than pandas, using a fixed point
    # iteration to remove all callables.
    # See https://github.com/pandas-dev/pandas/issues/53533
    if callable(key):
        return unpack_iloc_key(key(frame), frame)
    n = len(frame.shape)
    if isinstance(key, tuple):
        indexers = tuple(
            itertools.chain(key, itertools.repeat(slice(None), n - len(key)))
        )
        if (ni := len(indexers)) > n:
            raise IndexError(f"Too many indexers: got {ni} expected {n}")
        if any(isinstance(k, tuple) for k in indexers):
            # Only one level of tuple-nesting allowed
            raise IndexError(
                "Too many indexers: can't have nested tuples for iloc"
            )
        return tuple(unpack_iloc_key(k, frame) for k in key)
    # No special-casing, key gets rows, and if a dataframe second part
    # gets all columns
    return (key, slice(None))[:n]


def unpack_loc_key(key, frame: cudf.DataFrame | cudf.Series):
    """Unpack a user-level key to loc.__getitem__

    Parameters
    ----------
    key
        Key to unpack
    frame
        DataFrame or Series to provide context

    Returns
    -------
    tuple
        Tuple of row and (for dataframes) column keys

    Raises
    ------
    IndexError
        If a structurally invalid key is provided
    NotImplementedError
        If either the row or column indices of the frame are multiindices
    """
    if isinstance(frame.index, cudf.MultiIndex) or frame._data.multiindex:
        raise NotImplementedError("Not supported yet")
    # For non-multiindex, we can unpack as for iloc, except that pandas
    # has inconsistent semantics, see
    # https://github.com/pandas-dev/pandas/issues/53535
    # So this is a slight difference, though arguably we are fine
    # since we can't have tuple labels anyway.
    return unpack_iloc_key(key, frame)


def unpack_dataframe_iloc_indexer(
    key: Any, frame: cudf.DataFrame
) -> Tuple[Any, ColumnLabels]:
    """Unpack and index key for DataFrame iloc getitem.

    Parameters
    ----------
    key
        Key to unpack
    frame
        DataFrame for unpacking context

    Returns
    -------
    tuple
        2-tuple of a key for the rows and a sequence of column names

    Raises
    ------
    TypeError
        If the column indexer is invalid
    """
    rows, cols = unpack_iloc_key(key, frame)
    try:
        column_names: ColumnLabels = frame._data.get_labels_by_index(cols)
    except TypeError:
        raise TypeError(
            "Column indices must be integers, slices, "
            "or list-like of integers"
        )
    return (rows, column_names)


def unpack_series_iloc_indexer(key: Any, frame: cudf.Series) -> Any:
    """Unpack an index key for Series iloc getitem.

    Parameters
    ----------
    key
        Key to unpack
    frame
        Series for unpacking context

    Returns
    -------
    Single key that will index the rows
    """
    (rows,) = unpack_iloc_key(key, frame)
    return rows


def normalize_row_iloc_indexer(
    key: Any, n: int, check_bounds=False
) -> IndexSpec:
    """
    Normalize and produce structured information about a row indexer

    Given a row indexer that has already been normalized by
    :func:`unpack_iloc_key`, inspect further and produce structured
    information for indexing operations to act upon.

    Parameters
    ----------
    key
        Suitably normalized key for row indexing
    n
        Length of frame to index
    check_bounds
        If True, perform bounds checking of the key if it is a gather
        map.

    Returns
    -------
    IndexSpec
        Structured data for indexing. The first entry is a
        :class:`Indexer` tag, the second entry is normalized
        arguments to the tag-specific indexing routine.

    Raises
    ------
    IndexError
        If a valid type of indexer is provided, but it is out of
        bounds
    TypeError
        If the indexing key is otherwise invalid.
    """
    if key is Ellipsis:
        key = slice(None)
    if isinstance(key, slice):
        return (Indexer.SLICE, key.indices(n))
    else:
        if _is_scalar_or_zero_d_array(key):
            key = np.asarray(key)
            if not is_integer_dtype(key.dtype):
                raise TypeError(
                    "Cannot index by location with non-integer key"
                )
            if key < 0:
                key += n
            if not 0 <= key < n:
                raise IndexError("Positional indexer is out-of-bounds")
            return (Indexer.SCALAR, key.astype(np.int32))
        key = cudf.core.column.as_column(key)
        if isinstance(key, cudf.core.column.CategoricalColumn):
            key = key.as_categorical_column(key.categories.dtype)
        if is_bool_dtype(key.dtype):
            if (kn := len(key)) != n:
                raise IndexError(
                    f"Invalid length for boolean mask (got {kn}, need {n})"
                )
            return (Indexer.MASK, key)
        elif is_integer_dtype(key.dtype) or len(key) == 0:
            if check_bounds and not libcudf.copying._gather_map_is_valid(
                key, n, True, False
            ):
                raise IndexError("Gather map index is out of bounds.")
            return (Indexer.INDICES, key)
        else:
            raise TypeError(
                "Cannot index by location "
                f"with non-integer key of type {type(key)}"
            )
