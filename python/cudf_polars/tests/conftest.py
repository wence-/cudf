# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0


def pytest_sessionstart(session):
    from cudf_polars.patch import _WAS_PATCHED

    if not _WAS_PATCHED:
        # We could also just patch in the test, but this approach
        # provides a canary for failures with patching that we might
        # observe in trying this with other tests.
        raise RuntimeError("Patch was not applied")
