# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, TypeAlias

from cudf._lib.pylibcudf.column import Column

ColumnType: TypeAlias = Column
Expr: TypeAlias = Any
Plan: TypeAlias = Any
Visitor: TypeAlias = Any
