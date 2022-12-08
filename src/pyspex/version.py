#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Provide access to the software version as obtained from git.
"""

from pyspex import __version__


def get(full=False, githash=False):
    """Returns software version as obtained from git.
    """
    if full:
        return __version__

    if githash:
        return __version__.split('+g')[1].split('.')[0]

    return __version__.split('+')[0]
