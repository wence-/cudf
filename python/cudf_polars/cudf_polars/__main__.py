# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from cudf_polars.patch import patch_collect

if __name__ == "__main__":
    patch_collect()
