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

import xarray as xr

from pyspex.hkt_io import HKTio
from pyspex.lv0_io import (coverage_time,
                           dump_lv0_data,
                           read_lv0_data,
                           select_lv0_data,
                           write_lv0_data)

# - global parameters ------------------------------
ARG_FORMAT_HELP = """Provide data format of the input file(s):
- raw: CCSDS packages (a.o. ambient calibration);
- st3: CCSDS packages with ITOS and spacewire headers;
- dsb: files recorded on the observatory data storage board;
- default: determine file format from input files.
"""

ARG_INPUT_HELP = """Provide one or more input files:
- raw: if you provide only the file with extension '.H' then all files of the
       same measurement with science and house-keeping data are collected,
       else you have to provide these files yourself;
- st3: in general all measurement data are collected in one file;
- dsb: please provide all files with the data of one measurement.
"""

EPILOG_HELP = """Usage:
  Read inflight Level-0 data and write Level-1A product in directory L1A:

    spx1_level01a.py --outdir L1A <Path>/SPX*.spx

  If the Level-0 data contains Science and diagnostig measurements then use:

    spx1_level01a.py --outdir L1A <Path>/SPX*.spx --select binned
  or
    spx1_level01a.py --outdir L1A <Path>/SPX*.spx --select fullFrame

  Same call but now we add navigation data from HKT products:

    spx1_level01a.py --outdir L1A <Path>/SPX*.spx --pace_hkt <Path>/PACE.20220621T14*.HKT.nc

  Read OCAL Level-0 data and write Level-1A product in directory L1A:

    spx1_level01a.py --outdir L1A <Path>/NomSciCal1_20220123T121801.676167.H

    Note that OCAL science & telemetry data is read from the files:
      <Path>/NomSciCal1_20220123T121801.676167.?
      <Path>/NomSciCal1_20220123T121801.676167.??
      <Path>/NomSciCal1_20220123T121801.676167_hk.?

  Same call but now we are verbose during the data read (no output generated):

    spx1_level01a.py --debug <Path>/NomSciCal1_20220123T121801.676167.H

  Read ST3 Level-0 file and write Level-1A product in directory L1A:

    spx1_level01a.py --outdir L1A <Path>/SCI_20220124_174737_419.ST3

  Same call but now we dump packet header information in ASCII

    spx1_level01a.py --outdir L1A <Path>/SCI_20220124_174737_419.ST3 --dump
"""


# - local functions --------------------------------
def check_input_files(file_list: list, file_format: str) -> tuple:
    """
    Check input files on existence and format

    Parameters
    ----------
    file_list :  list of str
       List of file names
    file_format :  {'auto', 'raw', 'st3', 'dsb'}
       Expected Level-0 file format. Use 'auto' to let this function derive
       the file format from the file-names listed in `file_list`.

    Returns
    -------
    tuple
       file_format, file_list
    file_format : {'raw', 'st3', 'dsb'}
    file_list : list of Path

    Raises
    ------
    FileNotFoundError
       If files are not found on the system.
    TypeError
       If determined file type differs from value supplied by user.
    """
    if len(file_list) == 1 and Path(file_list[0]).suffix == '.H':
        data_dir = Path(file_list[0]).parent
        file_stem = Path(file_list[0]).stem
        file_list = (sorted(data_dir.glob(file_stem + '.[0-9]'))
                     + sorted(data_dir.glob(file_stem + '.?[0-9]'))
                     + sorted(data_dir.glob(file_stem + '_hk.[0-9]')))
        if not file_list:
            raise FileNotFoundError(file_stem)
        if file_format not in ('auto', 'raw'):
            raise TypeError('inconsistent file extensions')

        return 'raw', file_list

    file_format_out = file_format
    file_list_out = []
    for flname in file_list:
        flname = Path(flname)
        if not flname.is_file():
            raise FileNotFoundError(flname)

        if flname.suffix != '.H':
            res = {'.spx': 'dsb',
                   '.ST3': 'st3'}.get(flname.suffix, flname.suffix)
            new_file_format = 'raw' if res[1:].isdigit() else res
            if file_format_out == 'auto':
                file_format_out = new_file_format
            else:
                if file_format_out != new_file_format:
                    raise TypeError('inconsistent file extension')
            file_list_out.append(flname)

    return file_format_out, file_list_out


def read_hkt_nav(file_list: list) -> xr.Dataset:
    """
    Read multiple HKT products and collect data in a Python dictionary
    """
    dim_dict = {'att_': 'att_time',
                'orb_': 'orb_time',
                'tilt': 'tilt_time'}

    res = {}
    for name in file_list:
        hkt = HKTio(name)
        nav = hkt.navigation()
        if not res:
            res = nav.copy()
        else:
            for key1, value in nav.items():
                hdim = dim_dict.get(key1, None)
                res[key1] = xr.concat((res[key1], value), dim=hdim)

    return xr.merge((res['att_'], res['orb_'], res['tilt']),
                    combine_attrs='drop_conflicts')


def write_lv0_nav(l1a_file: str, xds_nav: xr.Dataset):
    """
    Add PACE navigation data to existing Level-1A product
    """
    xds_nav.to_netcdf(l1a_file, group='navigation_data', mode='a')


def get_l1a_name(file_list: list, file_format: str, file_version: int,
                 select: str, sensing_start: datetime) -> str:
    """
    Generate name of Level-1A product based on filename conventions described
    below

    Parameters
    ----------
    file_list :  list of Path
    file_format :  {'raw', 'st3', 'dsb'}
    file_version :  int
    select :  {'all', 'binned', 'fullFrame'}
    sensing_start :  datetime

    Returns
    -------
    str
        Name of Level-1A product

    Notes
    -----

    === Inflight ===
    L1A file name format, following the NASA ... naming convention:
       PACE_SPEXONE[_TTT].YYYYMMDDTHHMMSS.L1A[.Vnn].nc
    where
       TTT is an optional data type (e.g., for the calibration data files)
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       Vnn file-version number (ommited when nn=1)
    for example (file-version = 1):
       [Science Product] PACE_SPEXONE.20230115T123456.L1A.nc
       [Calibration Product] PACE_SPEXONE_CAL.20230115T123456.L1A.nc
       [Dark science Product] PACE_SPEXONE_DARK.20230115T123456.L1A.nc

    === OCAL ===
    L1A file name format:
       SPX1_OCAL_<msm_id>_L1A_YYYYMMDDTHHMMSS_yyyymmddThhmmss_vvvv.nc
    where
       msm_id is the measurement identifier
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       yyyymmddThhmmss is the creation time (UTC) of the product
       vvvv is the version number of the product starting at 0001
    """
    if file_format != 'raw':
        # inflight product name
        prod_type = '_CAL' if select == 'fullFrame' else ''
        prod_ver = '' if file_version==1 else f'.V{file_version:02d}'

        return (f'PACE_SPEXONE{prod_type}'
                f'.{sensing_start.strftime("%Y%m%dT%H%M%S"):15s}.L1A'
                f'.{prod_ver}.nc')

    # OCAL product name
    # determine measurement identifier
    msm_id = file_list[0].stem
    try:
        new_date = datetime.strptime(
            msm_id[-22:], '%y-%j-%H:%M:%S.%f').strftime('%Y%m%dT%H%M%S.%f')
    except ValueError:
        pass
    else:
        msm_id = msm_id[:-22] + new_date

    return (f'SPX1_OCAL_{msm_id}_L1A'
            f'_{sensing_start.strftime("%Y%m%dT%H%M%S"):15s}'
            f'_{datetime.utcnow().strftime("%Y%m%dT%H%M%S"):15s}'
            f'_{file_version:04d}.nc')


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
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Store SPEXone Level-0 data in a Level-1A product',
        epilog=EPILOG_HELP)
    parser.add_argument('--verbose', action='store_true', help='be verbose')
    parser.add_argument('--debug', action='store_true', help='be more verbose')
    parser.add_argument('--dump', action='store_true',
                        help=('dump CCSDS packet headers in ASCII'))
    parser.add_argument('--file_format', type=str, default='auto',
                        choices=('raw', 'st3', 'dsb'), help=ARG_FORMAT_HELP)
    parser.add_argument('--select', default='all',
                        choices=['binned', 'fullFrame'],
                        help='Select "binned" or "fullFrame" readouts')
    parser.add_argument('--outdir', type=Path, default=Path('.'),
                        help='Directory to store the Level-1A product')
    parser.add_argument('--outfile', type=str, default='',
                        help='Output Level-1A product filename')
    parser.add_argument('--file_version', type=int, default=1,
                        help='Provide file version number of level-1A product')
    parser.add_argument('--input_hkt_list', nargs='+', default=None,
                        help='names of PACE HKT products with navigation data')
    parser.add_argument('file_list', nargs='+', help=ARG_INPUT_HELP)
    args = parser.parse_args()

    # check list of input files and detect file format
    try:
        res = check_input_files(args.file_list, args.file_format)
    except FileNotFoundError as exc:
        print(f'[FATAL]: FileNotFoundError exception raised with "{exc}".')
        sys.exit(100)
    except TypeError as exc:
        print(f'[FATAL]: TypeError exception raised with "{exc}".')
        sys.exit(1)
    else:
        args.file_format, args.file_list = res

    # show the user command-line steeings after calling `check_input_files`
    if args.verbose:
        print(args)

    # read level 0 data as Science and TmTC packages
    try:
        res = read_lv0_data(args.file_list, args.file_format,
                            args.debug, args.verbose)
    except ValueError as exc:
        print(f'[FATAL]: ValueError exception raised with "{exc}".')
        sys.exit(100)
    if args.debug:
        return

    # perform an ASCII dump of level 0 headers parameters
    if args.dump:
        dump_lv0_data(args.file_list, args.outdir, *res)
        if args.verbose:
            print(f'Wrote ASCII dump in directory: {args.outdir}')
        return

    # we will not create a Level-1A product without Science data.
    if not res[0]:
        # inform the caller with a warning message and exit status
        print('[WARNING]: no science data found in L0 data, exit')
        sys.exit(110)

    # select Science and NomHK packages from level 0 data
    science, nomhk = select_lv0_data(args.select, res[0], res[1], args.verbose)

    # generate name of the level-1A product
    if args.outfile:
        prod_name = args.outfile
    else:
        prod_name = get_l1a_name(args.file_list, args.file_format,
                                 args.file_version, args.select,
                                 coverage_time(science)[0])

    # write L1A product
    try:
        write_lv0_data(args.outdir / prod_name, args.file_list,
                       args.file_format, science, nomhk)
    except PermissionError as exc:
        print(f'[FATAL] exception raised with "{exc}".')
        sys.exit(130)
    except RuntimeError as exc:
        print(f'[FATAL] exception raised with "{exc}".')
        sys.exit(131)

    # read PACE navigation information from HKT products
    if args.pace_hkt:
        hkt_nav = read_hkt_nav(args.pace_hkt)
        # select HKT data collocated with Science data
        # - issue a warning if selection is empty
        write_lv0_nav(args.outdir / prod_name, hkt_nav)

    # return with exit status zero
    if args.verbose:
        print(f'[INFO]: Successfully generated: {args.outdir / prod_name}')
    return


# --------------------------------------------------
if __name__ == '__main__':
    main()
