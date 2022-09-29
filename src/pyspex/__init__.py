# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#   All Rights Reserved
#
# License:  BSD-3-Clause

"""Python package pyspex contains software to access
   and create SPEXone L1A and L1B products.
"""

from pkg_resources import get_distribution, DistributionNotFound

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass
