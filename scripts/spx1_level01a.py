#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python script to store SPEXone Level-0 data in a new Level-1A product.

Examples
--------
Read CCSDS Level-0 files in current directory, write Level-1A in different
directory:

   spx1_level01a.py --datapath L1A <Path>/NomSciCal1_20220123T121801.676167.H

Note that science & telemetry data is read from:
  <Path>/NomSciCal1_20220123T121801.676167.?
  <Path>/NomSciCal1_20220123T121801.676167.??
  <Path>/NomSciCal1_20220123T121801.676167_hk.?

Read ST3 Level-0 file and write Level-1A in the directory $CWD/L1A:

   spx1_level01a.py --datapath L1A <Path>/SCI_20220124_174737_419.ST3

Read four DSB Level-0 files and write (one) Level-1A in the directory $CWD/L1A:

   spx1_level01a.py --datapath L1A <Path>/SPX00000000[0123].spx

Read ST3 Level-0 file and dump packet header information in ASCII to a file
with suffix replaced by '.dump' in directory 'L1A':

   spx1_level01a.py --datapath L1A <Path>/SCI_20220124_174737_419.ST3 --dump

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from datetime import datetime, timedelta, timezone
from pathlib import Path

from pyspex.lv0_io import (dump_lv0_data,
                           get_science_timestamps,
                           read_lv0_data,
                           select_lv0_data,
                           write_lv0_data)

# - global parameters ------------------------------
EPOCH_1958 = datetime(1958, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=37)
EPOCH_1970 = datetime(1970, 1, 1, tzinfo=timezone.utc)

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


# - local functions --------------------------------
def check_input_files(file_list: list, file_format: str) -> tuple:
    """
    Check input files on existence and format

    Parameters
    ----------
    file_list :  list of Path
    file_format :  {'raw', 'st3', 'dsb'}
    """
    if len(file_list) == 1 and Path(file_list[0]).suffix == '.H':
        data_dir = Path(file_list[0]).parent
        file_stem = Path(file_list[0]).stem
        file_list = (sorted(data_dir.glob(file_stem + '.[0-9]'))
                     + sorted(data_dir.glob(file_stem + '.?[0-9]'))
                     + sorted(data_dir.glob(file_stem + '_hk.[0-9]')))

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
                    raise TypeError('inconsistent file extensions')
            file_list_out.append(flname)

    return file_format_out, file_list_out


def get_l1a_name(file_list: list, file_format: str, file_version: int,
                 select: str, timestamp0: int) -> str:
    """
    Generate name of Level-1A product based on filename conventions described
    below

    Parameters
    ----------
    file_list :  list of Path
    file_format :  {'raw', 'st3', 'dsb'}
    file_version :  int
    select :  {'all', 'binned', 'fullFrame'}
    timestamp0 :  int

    Returns
    -------
    str
        Name of Level-1A product

    Notes
    -----

    === Inflight ===
    L1A file name format, following the NASA ... naming convention:
       PACE_SPEXone[_TTT].YYYYMMDDTHHMMSS.L1A.Vnn.nc
    where
       TTT is an optional data type (e.g., for the calibration data files)
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       nn file-version number
    for example
    [Science Product] PACE_SPEXone.20230115T123456.L1A.V01.nc
    [Calibration Product] PACE_SPEXone_CAL.20230115T123456.L1A.V01.nc
    [Monitoring Products] PACE_SPEXone_DARK.20230115T123456.L1A.V01.nc

    === OCAL ===
    L1A file name format:
       SPX1_OCAL_<msm_id>_L1A_YYYYMMDDTHHMMSS_yyyymmddThhmmss_vvvv.nc
    where
       msm_id is the measurement identifier
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       yyyymmddThhmmss is the creation time (UTC) of the product
       vvvv is the version number of the product starting at 0001
    """
    # set epoch of the timestamps
    epoch = EPOCH_1958 if file_format == 'dsb' else EPOCH_1970

    if file_format != 'raw':
        # inflight product name
        prod_type = '_CAL' if select == 'fullFrame' else ''
        sensing_start = epoch + timedelta(seconds=int(timestamp0))

        return (f'PACE_SPEXone{prod_type}'
                f'.{sensing_start.strftime("%Y%m%dT%H%M%S"):15s}.L1A'
                f'.V{file_version:02d}.nc')

    # OCAL product name
    sensing_start = epoch + timedelta(seconds=int(timestamp0))

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
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='store Level-0 data in a new SPEXone Level-1A product')
    parser.add_argument('--verbose', action='store_true', help='be verbose')
    parser.add_argument('--debug', action='store_true', help='be more verbose')
    parser.add_argument('--dump', action='store_true',
                        help=('dump CCSDS packet headers in ASCII'))
    parser.add_argument('--select', default='all',
                        choices=['binned', 'fullFrame'],
                        help='Select "binned" or "fullFrame" detector-readouts')
    parser.add_argument('--datapath', type=Path, default=Path('.'),
                        help='Directory to store the Level-1A product')
    parser.add_argument('--file_version', type=int, default=1,
                        help='Provide file version number of level-1A product')
    parser.add_argument('--file_format', type=str, default='auto',
                        choices=('raw', 'st3', 'dsb'), help=ARG_FORMAT_HELP)
    # parser.add_argument('--st3_nav', default=None, type=str,
    #                    help='name of ST3 file with navigation data')
    parser.add_argument('file_list', nargs='+', help=ARG_INPUT_HELP)
    args = parser.parse_args()

    # check list of input files and detect file format
    args.file_format, args.file_list = check_input_files(args.file_list,
                                                         args.file_format)
    if args.verbose:
        print(args)

    # read level 0 data
    res = read_lv0_data(args.file_list, args.file_format,
                        args.debug, args.verbose)
    if args.debug:
        return

    # perform an ASCII dump of level 0 headers parameters
    if args.dump:
        dump_lv0_data(args.file_list, args.datapath, *res)
        return

    # select Science and NomHK packages from level 0 data
    science, nomhk = select_lv0_data(args.select, res[0], res[1], args.verbose)

    # generate name of the level-1A product
    prod_name = get_l1a_name(args.file_list, args.file_format,
                             args.file_version, args.select,
                             get_science_timestamps(science[:1])[0])

    # write L1A product
    write_lv0_data(args.datapath / prod_name, args.file_list,
                   args.file_format, science, nomhk)


# --------------------------------------------------
if __name__ == '__main__':
    main()
