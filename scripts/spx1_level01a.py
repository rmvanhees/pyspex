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
    # parse command-line parameters and YAML file for settings
    try:
        config = get_l1a_settings()
    except FileNotFoundError as exc:
        print(f'[FATAL]: FileNotFoundError exception raised with "{exc}".')
        sys.exit(100)
    except TypeError as exc:
        print(f'[FATAL]: TypeError exception raised with "{exc}".')
        sys.exit(101)

    # show the user command-line steeings after calling `check_input_files`
    if config.verbose:
        print(config)

    # read level 0 data as Science and TMTC packages
    try:
        res = read_lv0_data(config.l0_list, config.l0_format,
                            config.debug, config.verbose)
    except ValueError as exc:
        print(f'[FATAL]: ValueError exception raised with "{exc}".')
        sys.exit(110)
    if config.debug:
        return

    # perform an ASCII dump of level 0 headers parameters
    if config.dump:
        try:
            dump_lv0_data(config.l0_list, config.outdir, *res)
        except FileNotFoundError as exc:
            print(f'[FATAL]: FileNotFoundError exception raised with "{exc}".')
            sys.exit(132)

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
    try:
        write_l1a(config, res[0], res[1])
    except PermissionError as exc:
        print(f'[FATAL]: "{exc}"')
        sys.exit(130)
    except (KeyError, RuntimeError) as exc:
        print(f'[FATAL]: "{exc}"')
        sys.exit(131)
    sys.exit(0)


# --------------------------------------------------
if __name__ == '__main__':
    main()
