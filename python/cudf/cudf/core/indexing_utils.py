# Copyright (c) 2023, NVIDIA CORPORATION.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, List, Tuple, Union

from typing_extensions import TypeAlias

import cudf
import cudf._lib as libcudf
from cudf._lib.types import size_type_dtype
from cudf.api.types import (
    _is_scalar_or_zero_d_array,
    is_bool_dtype,
    is_integer,
    is_integer_dtype,
    is_scalar,
)
from cudf.core import copy_types as ct


# Poor man's algebraic data types
class EmptyIndexer:
    """An indexer that will produce an empty result"""

    pass


@dataclass
class MapIndexer:
    """An indexer for a gather map"""

    key: ct.GatherMap


@dataclass
class MaskIndexer:
    """An indexer for a boolean mask"""

    key: ct.BooleanMask


@dataclass
class SliceIndexer:
    """An indexer for a slice"""

    key: slice


@dataclass
class ScalarIndexer:
    """An indexer for a scalar value"""

    key: ct.GatherMap


IndexingSpec: TypeAlias = Union[
    EmptyIndexer, MapIndexer, MaskIndexer, ScalarIndexer, SliceIndexer
]

ColumnLabels: TypeAlias = List[str]


def destructure_iloc_key(
    key: Any, frame: Union[cudf.Series, cudf.DataFrame]
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


def destructure_loc_key(
    key: Any, frame: cudf.Series | cudf.DataFrame
) -> tuple[Any, ...]:
    """
    Destructure a potentially tuple-typed key into row and column indexers

    Tuple arguments to loc indexing are treated specially. They are
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
    - A scalar label looked up in the index
    - A scalar integer that indexes the frame
    - An array-like of labels looked up in the idnex
    - A slice of the index

    Slice-based indexing is on the closed interval [start, end], rather
    than the semi-open interval [start, end)

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
    if (
        isinstance(frame.index, cudf.MultiIndex)
        and n == 2
        and isinstance(key, tuple)
        and all(map(is_scalar, key))
    ):
        # This is "best-effort"
        warnings.warn(
            "Guessing what scalar tuple key means. This is ambiguous "
            "and will be removed in a future version.",
            FutureWarning,
        )
        if len(key) == 2:
            if key[1] in frame.index._columns[1]:
                # key just indexes the rows
                key = (key,)
            elif key[1] in frame._data:
                # key indexes rows and columns
                key = key
            else:
                # key indexes rows and we will raise a keyerror
                key = (key,)
        else:
            # key just indexes rows
            key = (key,)
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
    return indexers


def destructure_dataframe_loc_indexer(
    key: Any, frame: cudf.DataFrame
) -> Tuple[Any, Tuple[bool, ColumnLabels]]:
    """Destructure an index key for DataFrame loc getitem.

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
    rows, cols = destructure_loc_key(key, frame)
    if cols is Ellipsis:
        cols = slice(None)
    scalar = is_integer(cols)
    try:
        # TODO here
        column_names: ColumnLabels = list(
            frame._data.select_by_label(cols).names
        )
        if len(set(column_names)) != len(column_names):
            raise NotImplementedError(
                "cudf DataFrames do not support repeated column names"
            )
    except TypeError:
        raise TypeError(
            "Column indices must be names, slices, "
            "list-like of names, or boolean mask"
        )
    if scalar:
        assert (
            len(column_names) == 1
        ), "Scalar column indexer should not produce more than one column"

    return (rows, (scalar, column_names))


def destructure_series_loc_indexer(key: Any, frame: cudf.Series) -> Any:
    """Destructure an index key for Series loc getitem.

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
    (rows,) = destructure_loc_key(key, frame)
    return rows


def parse_row_loc_indexer(key: Any, index: cudf.BaseIndex) -> IndexingSpec:
    """
    Normalize and produce structured information about a row indexer

    Given a row indexer that has already been destructured by
    :func:`destructure_loc_key`, inspect further and produce structured
    information for indexing operations to act upon.

    Parameters
    ----------
    key
        Suitably destructured key for row indexing
    n
        Length of frame to index

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
    n = len(index)
    # TODO: multiindices need to be treated separately
    if key is Ellipsis:
        return SliceIndexer(slice(None))
    elif isinstance(key, slice):
        # Convert label slice to index slice
        parsed_key = index.find_label_range(key)
        if len(range(n)[parsed_key]) == 0:
            return EmptyIndexer()
        else:
            return SliceIndexer(parsed_key)
    else:
        is_scalar = _is_scalar_or_zero_d_array(key)
        key = cudf.core.column.as_column(key)
        if isinstance(key, cudf.core.column.CategoricalColumn):
            key
            key = key.as_numerical_column(key.codes.dtype)
        if is_bool_dtype(key.dtype):
            # The only easy one.
            return MaskIndexer(ct.as_boolean_mask(key, n))
        elif len(key) == 0:
            return EmptyIndexer()
        else:
            # TODO promote to Index objects, so this can handle
            # categoricals correctly
            needle = key
            haystack = index._values
            dtype_kinds = {needle.dtype.kind, haystack.dtype.kind}
            try:
                needle = needle.astype(haystack.dtype)
            except ValueError:
                raise KeyError("Dtype mismatch in label lookup")
            # if dtype_kinds.issubset({"i", "u"}) or len(dtype_kinds) == 1:
            #     needle = needle.astype(haystack.dtype)
            # else:
            #     raise KeyError("Dtype mismatch in label lookup")
            left, right = libcudf.join.join([needle], [haystack], how="inner")
            if len(left) != len(needle):
                raise KeyError("Not all keys in index")
            (ordering,) = libcudf.copying.gather(
                [cudf.core.column.arange(len(needle), dtype=size_type_dtype)],
                left,
                nullify=False,
            )
            # stable sort only required for pandas-compat, arguably.
            (right,) = libcudf.sort.sort_by_key(
                [right], [ordering], [True], ["last"], stable=True
            )
            gather_map = ct.as_gather_map(
                right, n, nullify=False, check_bounds=False
            )
            if is_scalar:
                return ScalarIndexer(gather_map)
            else:
                return MapIndexer(gather_map)
