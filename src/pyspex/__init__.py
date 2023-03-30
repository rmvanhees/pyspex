# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause

"""Python package pyspex contains software to access
   and create SPEXone L1A and L1B products.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    print(__name__)
    print(__name__, version(__name__))
    __version__ = version(__name__)
except PackageNotFoundError:
    pass
