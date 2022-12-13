#!/usr/bin/env python3
#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Python script to store SPEXone Level-0 data in a Level-1A product."""

import sys

from pyspex.gen_l1a.cli import main

"""
main function

Returns
-------
err_code : int
Non-zero value indicates error code, or zero on success.

Notes
-----
Currently, the following return values are implemented:

* 2 if an error occurred durng parsing of the command-line arguments.

* 100 if one of the input files is unreadable.

* 101 if file not recognized as SPEXone level-0 data.

* 110 if we have issues in reading science data.

* 111 if no detector measurements are found in the level-0 data.

* 130 if the writing of the Level-1A failes due to permission denied.

* 131 if the writing of the Level-1A failed due netCDF/HDF5 errors.

* 132 if fail to write (ASCII) data dump.

* 139 if the writing of the Level-1A failed with a 'Segmentation fault',
caused by a 'Disk full error.

The following return values will be implemented:

* 120 if we have problems with ancillary info, e.g. start/stop time of
data.
"""

# --------------------------------------------------
if __name__ == '__main__':
    sys.exit(main())
