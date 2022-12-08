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
import argparse
from datetime import datetime
from pathlib import Path
import sys
import yaml

import xarray as xr

from pyspex.hkt_io import HKTio
from pyspex.lv0_io import (coverage_time,
                           dump_lv0_data,
                           read_lv0_data,
                           select_lv0_data,
                           write_lv0_data)

from pyspex.level_1a.options import get_settings

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
        config = get_settings()
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

    # select Science and NomHK packages from level 0 data
    if config.eclipse is None:
        # this are "OCAL data" try to write all data to one L1A product.
        pass
    elif not config.eclipse:
        # this are "Science data": binned data in "Science mode".
        pass
    else:
        # this can be "Dark data": binned data using "Science mode" MPSes
        # and/or "Calibration data”: full frame data in "Diagonstic mode".
    science, nomhk = select_lv0_data(args.select, res[0], res[1],
                                     config.verbose)

    # generate name of the level-1A product
    if config.outfile:
        prod_name = config.outfile
    else:
        prod_name = get_l1a_name(config.l0_list, config.l0_format,
                                 config.file_version, args.select,
                                 coverage_time(science)[0])

    # write L1A product
    try:
        write_lv0_data(config.outdir / prod_name, config.l0_list,
                       config.l0_format, science, nomhk)
    except PermissionError as exc:
        print(f'[FATAL] exception raised with "{exc}".')
        sys.exit(130)
    except RuntimeError as exc:
        print(f'[FATAL] exception raised with "{exc}".')
        sys.exit(131)

    # read PACE navigation information from HKT products
    if config.pace_hkt:
        hkt_nav = read_hkt_nav(config.pace_hkt)
        # select HKT data collocated with Science data
        # - issue a warning if selection is empty
        write_lv0_nav(config.outdir / prod_name, hkt_nav)

    # return with exit status zero
    if config.verbose:
        print(f'[INFO]: Successfully generated: {config.outdir / prod_name}')
    return


# --------------------------------------------------
if __name__ == '__main__':
    main()
