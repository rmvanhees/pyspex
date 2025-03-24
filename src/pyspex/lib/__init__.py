# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2025 SRON
#   All Rights Reserved
#
# License:  BSD-3-Clause
"""Necessary but empty file."""

from __future__ import annotations

__all__ = ["pyspex_version"]

import contextlib
from importlib.metadata import PackageNotFoundError, version


def pyspex_version(full: bool = False, githash: bool = False) -> str:
    """Return the software version as obtained from git."""
    with contextlib.suppress(PackageNotFoundError):
        __version__ = version("pyspex")

    if full:
        return __version__

    if githash:
        res = __version__.split("+g")
        if len(res) > 1:
            return res[1].split(".")[0]

        return "v" + "".join([f"{int(x):02d}" for x in res[0].split(".")])

    return __version__.split("+")[0]
