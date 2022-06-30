#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python script to store SPEXone Level-0 data in a new Level-1A product.

Examples
--------
Read ST3 Level-0 file in current directory, write Level-1A in different
directory:

   spx1_level01a.py --datapath L1A ./SCI_20220124_174737_419.ST3

Read CCSDS Level-0 files in current directory, write Level-1A in different
directory:

   spx1_level01a.py --datapath L1A ./NomSciCal1_20220123T121801.676167

Note that science data is read from ./NomSciCal1_20220123T121801.676167.?
and ./NomSciCal1_20220123T121801.676167.??, and telemetry data is read
from ./NomSciCal1_20220123T121801.676167_hk.?

Read ST3 Level-0 file in current directory, and write packet header information
to a file with extension '.dump' in directory 'L1A':

   spx1_level01a.py --datapath L1A ./SCI_20220124_174737_419.ST3 --dump

Note that the extension will not be removed, thus the name of the dump file is:
./L1A/SCI_20220124_174737_419.ST3.dump

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from pathlib import Path

from pyspex.lv0_io import (read_lv0_data, dump_lv0_data,
                           select_lv0_data, write_lv0_data)


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

# - local functions --------------------------------
def check_input_files(args) -> tuple:
    """
    Check input files on existence and format
    """
    if len(args.file_list) == 1 and Path(args.file_list[0]).suffix == '.H':
        data_dir = Path(args.file_list[0]).parent
        file_stem = Path(args.file_list[0]).stem
        file_list = (sorted(data_dir.glob(file_stem + '.[0-9]'))
                     + sorted(data_dir.glob(file_stem + '.?[0-9]'))
                     + sorted(data_dir.glob(file_stem + '_hk.[0-9]')))

        return 'raw', file_list

    file_format = args.file_format
    file_list = []
    for flname in args.file_list:
        flname = Path(flname)
        if not flname.is_file():
            raise FileNotFoundError(flname)

        if flname.suffix != '.H':
            res = {'.spx': 'dsb',
                   '.ST3': 'st3'}.get(flname.suffix, flname.suffix)
            new_file_format = 'raw' if res[1:].isdigit() else res
            if file_format == 'auto':
                file_format = new_file_format
            else:
                if file_format != new_file_format:
                    raise TypeError('inconsistent file extensions')
            file_list.append(flname)

    return file_format, file_list


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
    args.file_format, args.file_list = check_input_files(args)
    if args.verbose:
        print(args)

    # read level 0 data
    res = read_lv0_data(args)
    if args.debug:
        return

    # perform an ASCII dump of level 0 headers parameters
    if args.dump:
        dump_lv0_data(args, *res)
        return

    # select Science and NomHK packages from level 0 data
    science, nomhk = select_lv0_data(args, *res)

    # write L1A product
    write_lv0_data(args, science, nomhk)


# --------------------------------------------------
if __name__ == '__main__':
    main()
