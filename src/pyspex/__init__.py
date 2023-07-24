# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause

"""Python package pyspex contains software to read PACE HKT products,
read SPEXone Level-0 products and read/write SPEXone Level-1A products.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass
