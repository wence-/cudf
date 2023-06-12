# Copyright (c) 2023, NVIDIA CORPORATION.
from typing import TYPE_CHECKING, NamedTuple

import cudf._lib as libcudf
from cudf._lib.types import size_type_dtype

if TYPE_CHECKING:
    from cudf.core.column import ColumnBase


class GatherMap(NamedTuple):
    column: "ColumnBase"
    nrows: int
    nullify: bool


def as_gather_map(
    column: "ColumnBase",
    nrows: int,
    *,
    nullify: bool,
    check_bounds: bool,
) -> GatherMap:
    if len(column) == 0:
        # Any empty column is valid as a gather map
        # This is necessary because as_column([]) defaults to float64
        # TODO: we should fix this further up.
        # Alternately we can have an Optional[Column] and handle None
        # specially in _gather.
        return GatherMap(column.astype(size_type_dtype), nrows, nullify)
    if column.dtype.kind != "i":
        raise IndexError("Gather map must have integer dtype")
    if not nullify and check_bounds:
        lo, hi = libcudf.reduce.minmax(column)
        if lo.value < -nrows or hi.value >= nrows:
            raise IndexError(f"Gather map is out of bounds for [0, {nrows})")
    return GatherMap(column, nrows, nullify)
