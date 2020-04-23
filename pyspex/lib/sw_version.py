"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Copyright (c) 2019 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from ..version import version as __version__

def get(full=False):
    """
    Returns software version as obtained from git
    """
    if full:
        return __version__

    return __version__.split('+')[0]
