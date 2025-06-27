# Copyright (c) 2023-2025, NVIDIA CORPORATION.

from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import reduce
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np

import pylibcudf as plc

from cudf.api.types import _is_scalar_or_zero_d_array, is_integer
from cudf.core.column.column import as_column
from cudf.core.copy_types import BooleanMask, GatherMap
from cudf.core.dtypes import CategoricalDtype
from cudf.core.index import Index
from cudf.core.multiindex import MultiIndex

if TYPE_CHECKING:
    from collections.abc import Callable

    from cudf.core.column.column import ColumnBase
    from cudf.core.column_accessor import ColumnAccessor
    from cudf.core.dataframe import DataFrame
    from cudf.core.series import Series


class EmptyIndexer:
    """An indexer that will produce an empty result."""

    pass


@dataclass
class MapIndexer:
    """An indexer for a gather map."""

    key: GatherMap


@dataclass
class MaskIndexer:
    """An indexer for a boolean mask."""

    key: BooleanMask


@dataclass
class SliceIndexer:
    """An indexer for a slice."""

    key: slice


@dataclass
class ScalarIndexer:
    """An indexer for a scalar value."""

    key: GatherMap


IndexingSpec: TypeAlias = (
    EmptyIndexer | MapIndexer | MaskIndexer | ScalarIndexer | SliceIndexer
)


# Helpers for code-sharing between loc and iloc paths
def expand_key(key: Any, frame: DataFrame | Series) -> tuple[Any, ...]:
    """Slice-expand key to match dimension of the frame being indexed.

    Parameters
    ----------
    key
        Key to expand
    frame
        DataFrame or Series to expand to the dimension of.

    Returns
    -------
    tuple
        New key of length equal to the dimension of the frame.

    Raises
    ------
    IndexError
        If the provided key is a tuple and has more entries than the frame dimension.

    Notes
    -----
    If any individual entry in the key is a callable, it is called
    with the provided frame as argument and is required to be converted
    into a supported indexing type.
    """
    dim = len(frame.shape)
    if isinstance(key, tuple):
        # Key potentially indexes rows and columns, slice-expand to
        # shape of frame
        indexers = key + (slice(None),) * (dim - len(key))
        if len(indexers) > dim:
            raise IndexError(
                f"Too many indexers: got {len(indexers)} expected {dim}"
            )
    else:
        # Key indexes rows, slice-expand to shape of frame
        indexers = (key, *(slice(None),) * (dim - 1))
    return tuple(k(frame) if callable(k) else k for k in indexers)


def destructure_dataframe_indexer(
    key: Any,
    frame: DataFrame,
    destructure: Callable[[Any, DataFrame], tuple[Any, Any]],
    is_scalar: Callable[[Any, ColumnAccessor], bool],
    get_ca: str,
):
    """
    Pick apart an indexing key for a DataFrame into constituent pieces.

    Parameters
    ----------
    key
        The key to unpick.
    frame
        The DataFrame being indexed.
    destructure
        Callable to split the key into a two-tuple of row keys and
        column keys.
    is_scalar
        Callable to report if the column indexer produces a single
        column.
    get_ca
        Method name to obtain the column accessor from the frame.

    Returns
    -------
    rows
        Indexing expression for the rows
    tuple
        Two-tuple indicating if the column indexer produces a scalar and
        a subsetted ColumnAccessor.

    Raises
    ------
    TypeError
        If the column indexer is invalid.
    """
    rows, cols = destructure(key, frame)

    from cudf.core.series import Series

    if cols is Ellipsis:
        cols = slice(None)
    elif isinstance(cols, (Index, Series)):
        cols = cols.to_pandas()
    try:
        ca = getattr(frame._data, get_ca)(cols)
    except TypeError as e:
        raise TypeError(
            "Column indices must be names, slices, "
            "list-like of names, or boolean mask"
        ) from e
    scalar = is_scalar(cols, ca)
    if scalar:
        assert len(ca) == 1, (
            "Scalar column indexer should not produce more than one column"
        )
    return rows, (scalar, ca)


def destructure_iloc_key(
    key: Any, frame: Series | DataFrame
) -> tuple[Any, ...]:
    """
    Destructure a potentially tuple-typed key into row and column indexers.

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
    tuple
        Indexers with length equal to the dimension of the frame

    Raises
    ------
    IndexError
        If there are too many indexers, or any individual indexer is a tuple.
    """
    indexers = expand_key(key, frame)
    if any(isinstance(k, tuple) for k in indexers):
        raise IndexError(
            "Too many indexers: can't have nested tuples in iloc indexing"
        )
    return indexers


def destructure_dataframe_iloc_indexer(
    key: Any, frame: DataFrame
) -> tuple[Any, tuple[bool, ColumnAccessor]]:
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
        (column_index_is_scalar, ColumnAccessor) for the columns

    Raises
    ------
    TypeError
        If the column indexer is invalid
    IndexError
        If the provided key does not destructure correctly
    NotImplementedError
        If the requested column indexer repeats columns
    """
    return destructure_dataframe_indexer(
        key,
        frame,
        destructure_iloc_key,
        lambda col, _ca: is_integer(col),
        "select_by_index",
    )


def destructure_series_iloc_indexer(key: Any, frame: Series) -> Any:
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


def parse_row_iloc_indexer(key: Any, n: int) -> IndexingSpec:
    """
    Normalize and produce structured information about a row indexer.

    Given a row indexer that has already been destructured by
    :func:`destructure_iloc_key`, inspect further and produce structured
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
    if key is Ellipsis:
        return SliceIndexer(slice(None))
    elif isinstance(key, slice):
        return SliceIndexer(key)
    elif _is_scalar_or_zero_d_array(key):
        return ScalarIndexer(GatherMap(key, n, nullify=False))
    else:
        key = as_column(key)
        if isinstance(key.dtype, CategoricalDtype):
            key = key.astype(key.codes.dtype)
        if key.dtype.kind == "b":
            return MaskIndexer(BooleanMask(key, n))
        elif len(key) == 0:
            return EmptyIndexer()
        elif key.dtype.kind in "iu":
            return MapIndexer(GatherMap(key, n, nullify=False))
        else:
            raise TypeError(
                "Cannot index by location "
                f"with non-integer key of type {type(key)}"
            )


def destructure_loc_key(
    key: Any, frame: Series | DataFrame
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
    - An array-like of labels looked up in the index
    - A slice of the index
    - For multiindices, a tuple of per level indexers

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
        If there are too many indexers.
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
    return expand_key(key, frame)


def destructure_dataframe_loc_indexer(
    key: Any, frame: DataFrame
) -> tuple[Any, tuple[bool, ColumnAccessor]]:
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
        (column_index_is_scalar, ColumnAccessor) for the columns

    Raises
    ------
    TypeError
        If the column indexer is invalid
    IndexError
        If the provided key does not destructure correctly
    NotImplementedError
        If the requested column indexer repeats columns
    """

    def is_scalar(name: Any, ca: ColumnAccessor) -> bool:
        try:
            return name in ca
        except TypeError:
            return False

    return destructure_dataframe_indexer(
        key, frame, destructure_loc_key, is_scalar, "select_by_label"
    )


def destructure_series_loc_indexer(key: Any, frame: Series) -> Any:
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


def ordered_find(
    needles: ColumnBase,
    haystack: ColumnBase,
    *,
    how: Literal["left", "inner"] = "left",
) -> GatherMap:
    """Find locations of needles in a haystack preserving order

    Parameters
    ----------
    needles
        Labels to look for
    haystack
        Haystack to search in
    how
        Type of join to perform when matching needles and haystack.
        Use inner if all needles are guaranteed to be in the haystack.

    Returns
    -------
    NumericalColumn
        Integer gather map of locations needles were found in haystack

    Raises
    ------
    KeyError
        If not all needles were found in the haystack.
        If needles cannot be converted to the dtype of haystack.

    Notes
    -----
    This sorts the gather map so that the result comes back in the
    order the needles were specified (and are found in the haystack).
    """
    # Pre-process to match dtypes
    needle_kind = needles.dtype.kind
    haystack_kind = haystack.dtype.kind
    if haystack_kind == "O":
        try:
            needles = needles.astype(haystack.dtype)
        except ValueError:
            # Pandas raise KeyError here
            raise KeyError("Dtype mismatch in label lookup")
    elif needle_kind == haystack_kind or {
        haystack_kind,
        needle_kind,
    }.issubset({"i", "u", "f"}):
        needles = needles.astype(haystack.dtype)
    elif needles.dtype != haystack.dtype:
        # Pandas raise KeyError here
        raise KeyError("Dtype mismatch in label lookup")
    # Can't always do an inner join because then we can't check if we
    # had missing keys (can't check the length because the entries in
    # the needle might appear multiple times in the haystack).

    joiner = plc.join.inner_join if how == "inner" else plc.join.left_join
    right_policy = (
        plc.copying.OutOfBoundsPolicy.DONT_CHECK
        if how == "inner"
        else plc.copying.OutOfBoundsPolicy.NULLIFY
    )
    left_rows, right_rows = joiner(
        plc.Table([needles.to_pylibcudf(mode="read")]),
        plc.Table([haystack.to_pylibcudf(mode="read")]),
        plc.types.NullEquality.EQUAL,
    )
    right_order = plc.copying.gather(
        plc.Table(
            [
                plc.filling.sequence(
                    len(haystack), plc.Scalar.from_py(0), plc.Scalar.from_py(1)
                )
            ]
        ),
        right_rows,
        right_policy,
    ).columns()[0]
    if right_order.null_count() > 0:
        raise KeyError("Not all keys in index")
    left_order = plc.copying.gather(
        plc.Table(
            [
                plc.filling.sequence(
                    len(needles), plc.Scalar.from_py(0), plc.Scalar.from_py(1)
                )
            ]
        ),
        left_rows,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
    ).columns()[0]

    right_rows = plc.sorting.stable_sort_by_key(
        plc.Table([right_rows]),
        plc.Table([left_order, right_order]),
        [plc.types.Order.ASCENDING] * 2,
        [plc.types.NullOrder.AFTER] * 2,
    ).columns()[0]
    return GatherMap.from_column_unchecked(
        type(haystack).from_pylibcudf(right_rows),  # type: ignore[arg-type]
        len(haystack),
        nullify=False,
    )


def find_label_range_or_mask(
    key: slice, index: Index
) -> EmptyIndexer | SliceIndexer:
    """
    Convert a slice of labels into a slice of positions

    Parameters
    ----------
    key
        Slice to convert
    index
        Index to look up in

    Returns
    -------
    IndexingSpec
        Structured data for indexing (but never a :class:`ScalarIndexer`)

    Raises
    ------
    KeyError
        If the index is unsorted and not a DatetimeIndex
    """
    parsed_key = index.find_label_range(key)
    if len(range(len(index))[parsed_key]) == 0:
        return EmptyIndexer()
    else:
        return SliceIndexer(parsed_key)


def parse_single_row_loc_key(
    key: Any,
    index: Index,
) -> IndexingSpec:
    """
    Turn a single label-based row indexer into structured information.

    This converts label-based lookups into structured positional
    lookups.

    Valid values for the key are
    - a slice (endpoints are looked up)
    - a scalar label
    - a boolean mask of the same length as the index
    - a column of labels to look up (may be empty)

    Parameters
    ----------
    key
        Key for label-based row indexing
    index
        Index to act as haystack for labels

    Returns
    -------
    IndexingSpec
        Structured information for indexing

    Raises
    ------
    KeyError
        If any label is not found
    ValueError
        If labels cannot be coerced to index dtype
    """
    n = len(index)
    if isinstance(key, slice):
        return find_label_range_or_mask(key, index)
    else:
        is_scalar = _is_scalar_or_zero_d_array(key)
        if is_scalar and isinstance(key, np.ndarray):
            key = as_column(key.item())
        else:
            key = as_column(key)
        if (
            isinstance(key.dtype, CategoricalDtype)
            and index.dtype != key.dtype
        ):
            # TODO: is this right?
            key = key._get_decategorized_column()
        if len(key) == 0:
            return EmptyIndexer()
        else:
            # TODO: promote to Index objects, so this can handle
            # categoricals correctly?
            if key.dtype.kind == "b":
                if is_scalar and index.dtype.kind != "b":
                    raise KeyError(
                        "boolean label cannot be used without a boolean index"
                    )
                else:
                    return MaskIndexer(BooleanMask(key, n))
            elif index.dtype.kind == "M":
                # Try to turn strings into datetimes
                key = as_column(key, dtype=index.dtype)
            haystack = index._column
            gather_map = ordered_find(key, haystack)
            if is_scalar and len(gather_map.column) == 1:
                return ScalarIndexer(gather_map)
            else:
                return MapIndexer(gather_map)


def parse_row_loc_indexer(key: Any, index: Index) -> IndexingSpec:
    """
    Normalize to return structured information for a label-based row indexer.

    Given a label-based row indexer that has already been destructured by
    :func:`destructure_loc_key`, inspect further and produce structured
    information for indexing operations to act upon.

    Parameters
    ----------
    key
        Suitably destructured key for row indexing
    index
        Index to provide context

    Returns
    -------
    IndexingSpec
        Structured data for indexing. A tag + parsed data.

    Raises
    ------
    KeyError
        If a valid type of indexer is provided, but not all keys are
        found
    TypeError
        If the indexing key is otherwise invalid.
    """
    if isinstance(index, MultiIndex):
        raise NotImplementedError(
            "This code path is not designed for MultiIndex"
        )
    # TODO: multiindices need to be treated separately
    if key is Ellipsis:
        # Ellipsis is handled here because multiindex level-based
        # indices don't handle ellipsis in pandas.
        return SliceIndexer(slice(None))
    else:
        return parse_single_row_loc_key(key, index)


def parse_row_loc_indexer_multiindex(key: Any, index: cudf.MultiIndex):
    # Idea, we bitmask & the sub-indexers together
    # To do this, start with a bool mask of all True
    # Apply from each indexer from right to left
    # If we end up with
    # What is the best way to do this?
    # Slices can be intersected, bitmasks can be anded
    # maps make things difficult.
    # What produces a keyerror:
    # - bitmasks that are not all False anded together to produce a False
    #   result
    # - empty slices do not
    # - maps do (if the keys to their left produce a subset that removes
    #   the labels)
    # translate everything to indices
    # then work left to right
    # For the gathers, scatter True into a mask of False to produce
    # a bitmask
    # Need to check first for out of boundsness
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

    new_keys = list(
        parse_single_loc_key(
            subkey, cudf.core.index._index_from_data({None: subcolumn})
        )
        for subkey, subcolumn in zip(key, index._columns, strict=True)
    )
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
            # Masks and together to give no results, but individual
            # masks wanted something
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
                # Sliced part of masked indices is all false but we
                # have a not all false mask
                raise KeyError
            gather_maps.append(
                MapIndexer(
                    GatherMap.from_column_unchecked(
                        indices, len(index), nullify=False
                    )
                )
            )
            slicer = None
            boolean_masks = []
        else:
            boolean_masks = [
                MaskIndexer(BooleanMask.from_column_unchecked(mask))
            ]
    if gather_maps:
        # If we want pandas-like behaviour we can dedup and just use
        # the gather maps to populate bitmasks which we can and
        # together. (sometimes)
        # If we want "consistent with non-multiindex" maps then we
        # must compute the ordered set intersection of all the gather
        # maps
        # If this intersection (either way) is empty then we must
        # raise KeyError
        # If there is a slice this adds another constraint to the
        # intersection problem
        to_intersect = list(k.key.column for k in gather_maps)[::-1]
        if boolean_masks:
            (bmask,) = boolean_masks
            mask = bmask.key.column
            (indices,) = libcudf.stream_compaction.apply_boolean_mask(
                [cudf.core.column.arange(len(mask), dtype=size_type_dtype)],
                mask,
            )
            to_intersect.append(indices)
            boolean_masks = []
        to_intersect = to_intersect[::-1]
        intersection = to_intersect.pop()
        while to_intersect:
            right = to_intersect.pop()
            rgather = ordered_find(intersection, right, how="inner")
            if len(rgather.column) == 0:
                raise KeyError
            (intersection,) = libcudf.copying.gather(
                [right], rgather, nullify=False
            )
        # Whether or not slices removing the dataframe such that the
        # result is empty produce a keyerror depends on the order.
        # If the slice is last, no KeyError, if it is before, then
        # KeyError
        # This is not consistent with the idea that the multiindex
        # produces a pseudo-n-D representation with each level indexer
        # projecting to the sub-piece on an axis.
        # I suppose this is because if the data representation are
        # sparse then the intuition from slicing a dense nD tensor
        # falls by the way side a bit.
        if slicer:
            (intersection,) = ordered_find(
                intersection,
                cudf.core.column.arange(
                    slicer.start,
                    slicer.stop,
                    slicer.step,
                    dtype=size_type_dtype,
                ),
                how="inner",
            )
        slicer = None
        gather_map = GatherMap.from_column_unchecked(
            intersection, len(index), nullify=False
        )
        if (
            any(isinstance(k, ScalarIndexer) for k in gather_maps)
            and len(intersection) == 1
        ):
            gather_maps = [ScalarIndexer(gather_map)]
        else:
            gather_maps = [MapIndexer(gather_map)]
    if slicer:
        assert not boolean_masks and not gather_maps
        return SliceIndexer(slicer)
    if boolean_masks:
        assert not gather_maps
        (bmask,) = boolean_masks
        return bmask
    if gather_maps:
        assert not boolean_masks
        (map_,) = gather_maps
        return map_
    raise RuntimeError("Unpossible!")


def extended_euclid(a: int, b: int) -> tuple[int, int, int]:
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


def intersect_slices(
    n: int, *to_intersect: slice[int, int, int | None]
) -> range:
    """
    Produce a range representing the intersection of slices.

    Parameters
    ----------
    n
        Length of the range being sliced.
    to_intersect
        Slices to intersect over the range.

    Returns
    -------
    range
        A subset of the range [0, n) containing indices which are
        selected by all slices.

    Notes
    -----
    This is not equivalent to successively applying the slices from
    left to right to the dense range (unless all the steps are
    coprime). Consider intersecting [::2] and [::4], the intersection
    is [::4], but the repeated application is [::8].
    """
    if not to_intersect:
        return range(n)
    # Normalise to positive step ranges
    slices: tuple[slice[int, int, int]] = tuple(  # type: ignore[assign]
        s if s.step is not None else slice(s.start, s.stop, 1)
        for s in to_intersect
    )
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
