# Copyright (c) 2023, NVIDIA CORPORATION.

from __future__ import annotations

import operator
import warnings
from dataclasses import dataclass
from functools import partial, reduce
from typing import Any, List, Tuple, Union

import numpy as np
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
from cudf.core.column_accessor import ColumnAccessor


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
) -> Tuple[Any, Tuple[bool, ColumnAccessor]]:
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
        ca = frame._data.select_by_index(cols)
    except TypeError:
        raise TypeError(
            "Column indices must be integers, slices, "
            "or list-like of integers"
        )
    if scalar:
        assert (
            len(ca) == 1
        ), "Scalar column indexer should not produce more than one column"

    return (rows, (scalar, ca))


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
) -> Tuple[Any, Tuple[bool, ColumnAccessor]]:
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
    try:
        scalar = cols in frame._data
    except TypeError:
        scalar = False
    try:
        ca = frame._data.select_by_label(cols)
    except TypeError:
        raise TypeError(
            "Column indices must be names, slices, "
            "list-like of names, or boolean mask"
        )
    if scalar:
        assert (
            len(ca) == 1
        ), "Scalar column indexer should not produce more than one column"

    return (rows, (scalar, ca))


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
        # TODO: datetime index must be handled specially (unless we go for
        # pandas 2 compatibility)
        parsed_key = index.find_label_range(key)
        if len(range(n)[parsed_key]) == 0:
            return EmptyIndexer()
        else:
            return SliceIndexer(parsed_key)
    else:
        is_scalar = _is_scalar_or_zero_d_array(key)
        if is_scalar and isinstance(key, np.ndarray):
            key = cudf.core.column.as_column(key.item(), dtype=key.dtype)
        else:
            key = cudf.core.column.as_column(key)
        if (
            isinstance(key, cudf.core.column.CategoricalColumn)
            and index.dtype != key.dtype
        ):
            # TODO: is this right?
            key = key._get_decategorized_column()
        if is_bool_dtype(key.dtype):
            # The only easy one.
            return MaskIndexer(ct.as_boolean_mask(key, n))
        elif len(key) == 0:
            return EmptyIndexer()
        else:
            # TODO promote to Index objects, so this can handle
            # categoricals correctly
            if isinstance(index, cudf.DatetimeIndex):
                key = cudf.core.column.as_column(key, dtype=index.dtype)
            needle = key
            haystack = index._values
            needle_kind = needle.dtype.kind
            haystack_kind = haystack.dtype.kind
            if haystack_kind == "O":
                try:
                    needle = needle.astype(haystack.dtype)
                except ValueError:
                    raise KeyError("Dtype mismatch in label lookup")
            elif needle_kind == haystack_kind or {
                haystack_kind,
                needle_kind,
            }.issubset({"i", "u", "f"}):
                needle = needle.astype(haystack.dtype)
            elif needle.dtype != haystack.dtype:
                raise KeyError("Dtype mismatch in label lookup")
            # if dtype_kinds.issubset({"i", "u"}) or len(dtype_kinds) == 1:
            #     needle = needle.astype(haystack.dtype)
            # else:
            #     raise KeyError("Dtype mismatch in label lookup")
            left, right = libcudf.join.join([needle], [haystack], how="left")
            (left_order,) = libcudf.copying.gather(
                [cudf.core.column.arange(len(needle), dtype=size_type_dtype)],
                left,
                nullify=False,
            )
            (right_order,) = libcudf.copying.gather(
                [
                    cudf.core.column.arange(
                        len(haystack), dtype=size_type_dtype
                    )
                ],
                right,
                nullify=True,
            )

            if right_order.null_count > 0:
                raise KeyError("Not all keys in index")
            (right,) = libcudf.sort.sort_by_key(
                [right],
                [left_order, right_order],
                [True, True],
                ["last", "last"],
                stable=True,
            )
            gather_map = ct.as_gather_map(
                right, n, nullify=False, check_bounds=False
            )
            if is_scalar and len(right) == 1:
                return ScalarIndexer(gather_map)
            else:
                return MapIndexer(gather_map)


def parse_row_loc_indexer_multiindex(key: Any, index: cudf.MultiIndex):
    # Idea, we bitmask & the sub-indexers together
    # To do this, start with a bool mask of all True
    # Apply from each indexer from right to left
    # If we end up with
    # What is the best way to do this?
    # Slices can be intersected, bitmasks can be anded
    # maps make things difficult.
    # What produces a keyerror:
    # - bitmasks that are not all False anded together to produce a False result
    # - empty slices do not
    # - maps do (if the keys to their left produce a subset that removes the labels)
    # translate everything to indices
    # then work left to right
    # For the gathers, scatter True into a mask of False to produce a bitmask
    # Need to check first for out of boundsness
    n = len(index)
    nlevel = index.nlevels
    assert isinstance(key, tuple) and len(key) == nlevel
    if any(
        isinstance(subkey, slice)
        and (isinstance(subkey.start, tuple) or isinstance(subkey.stop, tuple))
        for subkey in key
    ):
        raise NotImplementedError(
            "CuDF does not support multiindex slicing with tuple ranges"
        )

    new_keys = []
    for subkey, subcolumn in zip(key, index._columns):
        if isinstance(subkey, slice):
            # Convert label slice to index slice
            # TODO: datetime index must be handled specially (unless we go for
            # pandas 2 compatibility)
            parsed_key = index.find_label_range(key)
            if len(range(n)[parsed_key]) == 0:
                new_keys.append(EmptyIndexer())
            else:
                new_keys.append(SliceIndexer(parsed_key))
        else:
            is_scalar = _is_scalar_or_zero_d_array(key)
            if is_scalar and isinstance(key, np.ndarray):
                key = cudf.core.column.as_column(key.item(), dtype=key.dtype)
            else:
                key = cudf.core.column.as_column(key)
            if (
                isinstance(key, cudf.core.column.CategoricalColumn)
                and index.dtype != key.dtype
            ):
                # TODO: is this right?
                key = key._get_decategorized_column()
            if is_bool_dtype(key.dtype):
                # The only easy one.
                new_keys.append(MaskIndexer(ct.as_boolean_mask(key, n)))
            elif len(key) == 0:
                new_keys.append(EmptyIndexer())
            else:
                # TODO promote to Index objects, so this can handle
                # categoricals correctly
                if isinstance(index, cudf.DatetimeIndex):
                    key = cudf.core.column.as_column(key, dtype=index.dtype)
                needle = key
                haystack = index._values
                needle_kind = needle.dtype.kind
                haystack_kind = haystack.dtype.kind
                if haystack_kind == "O":
                    try:
                        needle = needle.astype(haystack.dtype)
                    except ValueError:
                        raise KeyError("Dtype mismatch in label lookup")
                elif needle_kind == haystack_kind or {
                    haystack_kind,
                    needle_kind,
                }.issubset({"i", "u", "f"}):
                    needle = needle.astype(haystack.dtype)
                elif needle.dtype != haystack.dtype:
                    raise KeyError("Dtype mismatch in label lookup")
                # if dtype_kinds.issubset({"i", "u"}) or len(dtype_kinds) == 1:
                #     needle = needle.astype(haystack.dtype)
                # else:
                #     raise KeyError("Dtype mismatch in label lookup")
                left, right = libcudf.join.join(
                    [needle], [haystack], how="left"
                )
                # TODO: This reordering isn't required if the way we
                # handle gather is to construct a mask and scatter
                # into it.
                (left_order,) = libcudf.copying.gather(
                    [
                        cudf.core.column.arange(
                            len(needle), dtype=size_type_dtype
                        )
                    ],
                    left,
                    nullify=False,
                )
                (right_order,) = libcudf.copying.gather(
                    [
                        cudf.core.column.arange(
                            len(haystack), dtype=size_type_dtype
                        )
                    ],
                    right,
                    nullify=True,
                )

                if right_order.null_count > 0:
                    raise KeyError("Not all keys in index")
                (right,) = libcudf.sort.sort_by_key(
                    [right],
                    [left_order, right_order],
                    [True, True],
                    ["last", "last"],
                    stable=True,
                )
                gather_map = ct.as_gather_map(
                    right, n, nullify=False, check_bounds=False
                )
                if is_scalar and len(right) == 1:
                    new_keys.append(ScalarIndexer(gather_map))
                else:
                    new_keys.append(MapIndexer(gather_map))
    return combine_keys(new_keys, index)


def combine_keys(keys: list[IndexingSpec], index):
    # Now we have all the keys as iloc pieces
    # intersect slices
    # construct bitmasks for gathermaps
    # and bitmasks together
    # slice away bitmasks using complement of intersected slices
    # now we have a single boolean mask
    # Pandas does "approximately" the above. This doesn't preserve the
    # reordering induced by list-like gather maps, nor does it handle
    # slice reordering (with negative steps).
    # Possible plan: do the right and proper thing when there is an
    # unambiguous result (basically boolean masks or slices with step
    # 1 strides), or "single-level" indexing.
    # Raise AmbiguousIndexError otherwise

    boolean_masks = list(k for k in keys if isinstance(k, MaskIndexer))
    gather_maps = list(
        k for k in keys if isinstance(k, (MapIndexer, ScalarIndexer))
    )
    slices = list(k for k in keys if isinstance(k, SliceIndexer))
    empty = list(k for k in keys if isinstance(k, EmptyIndexer))
    if empty:
        return EmptyIndexer()
    slice_intersection = intersect_slices(len(index), *(s.key for s in slices))
    if len(slice_intersection) == 0:
        return EmptyIndexer()
    if slice_intersection == range(len(index)):
        slicer = None
    else:
        slicer = slice(
            slice_intersection.start,
            slice_intersection.stop,
            slice_intersection.step,
        )
    if boolean_masks:
        mask = reduce(
            partial(libcudf.binaryop.binaryop, op="__l_and__", dtype=bool),
            (k.key.column for k in boolean_masks),
        )
        all_false = not mask.sum()
        if all_false and any(k.key.column.sum() for k in boolean_masks):
            # Masks and together to give no results, but individual masks wanted something
            raise KeyError
        if slicer:
            mask = mask.slice(slicer.start, slicer.stop, slicer.step)
            indices = cudf.core.column.arange(
                slicer.start, slicer.stop, slicer.step, dtype=size_type_dtype
            )
            (indices,) = libcudf.stream_compaction.apply_boolean_mask(
                [indices], mask
            )
            if not all_false and len(indices) == 0:
                # Sliced part of masked indices is all false but we have a not all false mask
                raise KeyError
            gather_maps.append(
                MapIndexer(
                    ct.as_gather_map(
                        indices, len(index), nullify=False, check_bounds=False
                    )
                )
            )
            boolean_masks = []
        else:
            boolean_masks = [MaskIndexer(ct.as_boolean_mask(mask, len(index)))]
    if gather_maps:
        # If we want pandas-like behaviour we can dedup and just use
        # the gather maps to populate bitmasks which we can and
        # together.
        # If we want "consistent with non-multiindex" maps then we
        # must compute the ordered set intersection of all the gather
        # maps
        # If there is a slice this adds another constraint to the intersection problem
        to_intersect = list(k.key.column for k in gather_maps)
        if slicer:
            to_intersect.append(
                cudf.core.column.arange(
                    slicer.start,
                    slicer.stop,
                    slicer.step,
                    dtype=size_type_dtype,
                )
            )


def extended_euclid(a: int, b: int) -> Tuple[int, int, int]:
    """
    Compute an integer pair (x, y) such that

    a x + b y = gcd(x, y)

    Per Bézout's theorem, there are two such "small" pairs, and this
    algorithm is guaranteed to find one of them.

    Returns
    -------
    tuple
        x, y, gcd
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    return (old_s, old_t, old_r)


def intersect_slices(n: int, *slices: slice) -> range:
    if not slices:
        return range(n)
    # Normalise to positive step ranges
    slices = tuple(s if s.step else slice(s.start, s.stop, 1) for s in slices)
    step_sign = 1 - 2 * (reduce(operator.mul, (s.step for s in slices)) < 0)
    ranges = list(
        range(n)[s] if s.step > 0 else range(n)[s][::-1] for s in slices
    )
    if all(r.step == 1 for r in ranges):
        # Ranges are dense, so we can just compute overlaps
        lo = max(r.start for r in ranges)
        hi = min(r.stop for r in ranges)
        return range(lo, hi, 1)[::step_sign]
    result = ranges.pop()
    while ranges:
        current = ranges.pop()
        s, _, gcd = extended_euclid(result.step, current.step)
        if (result.start - current.start) % gcd:
            # Points will never intersect
            return range(0)[::step_sign]
        start = (
            result.start
            + (current.start - result.start) * result.step // gcd * s
        )
        step = result.step * current.step // gcd
        nstep = -((start - max(result.start, current.start)) // step)
        start = start + step * nstep
        result = range(start, min(result.stop, current.stop), step)
    return result[::step_sign]
