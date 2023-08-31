#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause

"""Provide access to the software version as obtained from git."""

from __future__ import annotations

__all__ = ['pyspex_version']

from pyspex import __version__


def pyspex_version(full: bool = False, githash: bool = False) -> str:
    """Return the software version as obtained from git."""
    if full:
        return __version__

    if githash:
        res = __version__.split('+g')
        if len(res) > 1:
            return res[1].split('.')[0]

        return 'v' + ''.join([f'{int(x):02d}' for x in res[0].split('.')])

    return __version__.split('+')[0]
