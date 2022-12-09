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
"""
Python script to store SPEXone Level-0 data in a Level-1A product.
"""
import sys

from pyspex.lv0_io import dump_lv0_data, read_lv0_data
from pyspex.lv1_io import write_l1a

from pyspex.lv1_args import get_l1a_settings

# - global parameters ------------------------------

# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """
    main function

    Returns
    -------
    err_code : int
       Non-zero value indicates error code, or zero on success.

    Notes
    -----
    Currently, the following return values are implemented:

      * 100 if one of the input files is unreadable.

      * 110 if we have issues in reading science data, e.g. no science
        packages, no image headers, corrupted science data.

      * 130 if the writing of the Level-1A failed due to permissio denied.

      * 131 if the writing of the Level-1A failed due netCDF/HDF5 errors.

      * 139 if the writing of the Level-1A failed with a 'Segmentation fault',
        caused by a 'Disk full error.

    The following return values will be implemented:

      * 120 if we have problems with ancillary info, e.g. start/stop time of
        data.
    """
    # parse command-line parameters and YAML file for settings
    try:
        config = get_l1a_settings()
    except FileNotFoundError as exc:
        print(f'[FATAL]: FileNotFoundError exception raised with "{exc}".')
        sys.exit(100)
    except TypeError as exc:
        print(f'[FATAL]: TypeError exception raised with "{exc}".')
        sys.exit(1)

    # show the user command-line steeings after calling `check_input_files`
    if config.verbose:
        print(config)

    # read level 0 data as Science and TmTC packages
    try:
        res = read_lv0_data(config.l0_list, config.l0_format,
                            config.debug, config.verbose)
    except ValueError as exc:
        print(f'[FATAL]: ValueError exception raised with "{exc}".')
        sys.exit(100)
    if config.debug:
        return

    # perform an ASCII dump of level 0 headers parameters
    if config.dump:
        dump_lv0_data(config.l0_list, config.outdir, *res)
        if config.verbose:
            print(f'Wrote ASCII dump in directory: {config.outdir}')
        return

    # we will not create a Level-1A product without Science data.
    if not res[0]:
        # inform the caller with a warning message and exit status
        print('[WARNING]: no science data found in L0 data, exit')
        sys.exit(110)

    # Write Level-1A product.
    # ToDo add try/except
    write_l1a(config, res[0], res[1])
    sys.exit(0)


# --------------------------------------------------
if __name__ == '__main__':
    main()
