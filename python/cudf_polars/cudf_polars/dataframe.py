# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import enum
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Literal

import cudf._lib.pylibcudf as plc
from cudf._lib.types import dtype_from_pylibcudf_column

from cudf_polars.typing import ColumnType

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pyarrow as pa
    from typing_extensions import Self


class DataFrame(dict[str, ColumnType]):
    """
    A basic representation of a dataframe with effectively no methods.

    This is a trivial wrapper around a dictionary.

    Turn it into a fully-featured dataframe by doing cudf.DataFrame(obj)
    """

    class IsSorted(IntEnum):
        """Flag tracked sortedness of a column."""

        ASCENDING = enum.auto()
        DESCENDING = enum.auto()
        NOT = enum.auto()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sortedness = {k: self.IsSorted.NOT for k in self.keys()}
        # Correct by construction
        if len(self) != 0:
            try:
                (_,) = set(map(len, self.values()))
            except ValueError as err:
                raise ValueError("All columns must have same length") from err
            except TypeError:
                # HACK: scalars
                pass

    def names(self) -> list[str]:
        """The column names."""
        return list(self.keys())

    def columns(self) -> list[ColumnType]:
        """The columns."""
        return list(self.values())

    def select(self, names: list[str]) -> Self:
        """
        Select a subset of columns.

        Parameters
        ----------
        names
            List of column names to select

        Returns
        -------
        New subsetted dataframe, sharing column data
        """
        return type(self)(
            zip(names, (self[name] for name in names), strict=True)
        ).set_sorted(self.sorted(names))

    def discard(self, names: set[str]) -> Self:
        """
        Discard a set of columns.

        Parameters
        ----------
        names
            Names to discard

        Returns
        -------
        New subsetted dataframe, sharing column data
        """
        if names - self.keys():
            raise ValueError("Trying to discard some names that don't exist")
        return self.select([n for n in self.keys() if n not in names])

    def sorted(
        self, names: list[str] | None = None
    ) -> dict[str, DataFrame.IsSorted]:
        """
        Return sortedness of given columns.

        Parameters
        ----------
        names
            Columns to get sortedness for (or None for all columns)

        Returns
        -------
        dict mapping column names to sortedness status
        """
        if names is None:
            return self._sortedness
        else:
            return {name: self._sortedness[name] for name in names}

    def set_sorted(self, sortedness: dict[str, DataFrame.IsSorted]) -> Self:
        """
        Set sortedness of specified columns.

        Parameters
        ----------
        sortedness
            mapping from column names to their sortedness flag. Any
            columns that are not specified here will keep their
            sortedness from the input dataframe.

        Returns
        -------
        DataFrame
            New dataframe with updated sortedness flags.
        """
        result = type(self)(self)
        result._sortedness = self._sortedness | sortedness
        return result

    def rename(self, name_map: dict[str, str]) -> Self:
        """
        Rename columns.

        Parameters
        ----------
        name_map
            Mapping of old name to new name, any missing names are
            taken from the existing names.

        Returns
        -------
        Renamed dataframe
        """
        result = type(self)((name_map.get(k, k), v) for k, v in self.items())
        result._sortedness = {
            name_map.get(k, k): v for k, v in self._sortedness.items()
        }
        return result

    def schema(self) -> dict[str, Any]:
        """A mapping of column names to dtypes."""
        return {k: dtype_from_pylibcudf_column(v) for k, v in self.items()}

    def num_rows(self) -> int:
        """Return the number of rows in the dataframe."""
        if len(self) == 0:
            return 0
        c = next(iter(self.values()))
        return c.size()

    def num_columns(self) -> int:
        """Return the number of columns in the dataframe."""
        return len(self)

    def gather(
        self, rows: ColumnType, *, bounds_policy: plc.copying.OutOfBoundsPolicy
    ) -> DataFrame:
        """
        Gather rows of a dataframe.

        Parameters
        ----------
        rows
            Rows to gather
        bounds_policy
            What to do with out of bounds indices

        Returns
        -------
        New dataframe

        Notes
        -----
        No bounds-checking is performed on the entries of rows.
        """
        return self.from_pylibcudf(
            self.names(),
            plc.copying.gather(self.to_pylibcudf(), rows, bounds_policy),
        )

    def slice(self, start: int, length: int) -> Self:
        """
        Slice a dataframe.

        Parameters
        ----------
        start
            Start of slice (negative value treated as for python indexing)
        length
            Length of slice

        Returns
        -------
        Sliced dataframe
        """
        if start < 0:
            start += self.num_rows()
        # Polars slice can take an arbitrary positive integer and slices "to the end"
        (table,) = plc.copying.slice(
            self.to_pylibcudf(), [start, min(start + length, self.num_rows())]
        )
        return self.from_pylibcudf(self.names(), table).set_sorted(
            self.sorted(self.names())
        )

    def filter(self, mask: ColumnType) -> Self:
        """
        Filter a dataframe.

        Parameters
        ----------
        mask
            Boolean mask

        Returns
        -------
        New dataframe with rows corresponding to true values in the
        mask column
        """
        return self.from_pylibcudf(
            self.names(),
            plc.stream_compaction.apply_boolean_mask(
                self.to_pylibcudf(), mask
            ),
        ).set_sorted(self.sorted(self.names()))

    def to_arrow(self) -> pa.Table:
        """Convert to an arrow table."""
        return plc.interop.to_arrow(
            self.to_pylibcudf(),
            self.names(),
        )

    def to_pylibcudf(
        self, *, mode: Literal["read", "write"] = "read"
    ) -> plc.Table:
        """
        Convert to a pylibcudf table.

        Parameters
        ----------
        mode
            Access mode, generally "read"

        Returns
        -------
        pylibcudf Table of the columns
        """
        # Note: The resulting table will have references to the same
        # underlying columns
        return plc.Table(self.columns())

    @classmethod
    def from_pylibcudf(
        cls, names: Sequence[str] | Iterable[str], table: plc.Table
    ) -> Self:
        """
        Convert from a pylibcudf table.

        Parameters
        ----------
        names
            Column names
        table
            pylibcudf table

        Returns
        -------
        New dataframe
        """
        return cls(
            zip(
                names,
                table.columns(),
                strict=True,
            )
        )
