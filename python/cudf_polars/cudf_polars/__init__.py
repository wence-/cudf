# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import os

from cudf_polars.patch import patch_collect


def pytest_load_initial_conftests(early_config, parser, args):
    """Enable use of this module as a pytest plugin to enable GPU collection."""
    cpu_fallback = "FORBID_CPU_FALLBACK" not in os.environ
    patch_collect(use_gpu_default=True, cpu_fallback_default=cpu_fallback)
