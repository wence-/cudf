from typing import List, Optional, Tuple

from cudf.core.column.column import ColumnBase

def join(
    lhs: List, rhs: List, how: Optional[str] = None
) -> Tuple[ColumnBase, ColumnBase]: ...
def semi_join(
    lhs: List, rhs: List, how: Optional[str] = None
) -> Tuple[ColumnBase, None]: ...
