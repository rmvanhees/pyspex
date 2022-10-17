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
Create or update SPEXone binning-table CKD.

- Input is binning table information in CSV format.

- The binning-table CKD is written in netCDF4 format.
"""

import argparse
from pathlib import Path

import numpy as np

from pyspex.binning_tables import BinningTables


# - global parameters ------------------------------


# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """
    main program to build a database with SPEXone binning tables

    Create or update SPEXone binning-table CKD.
    Input is binning table information in CSV format.
    The binning-table CKD is written in netCDF4 format.
    """
    parser = argparse.ArgumentParser(
        description='build a database with SPEXone binning tables')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('--validity_start', default=None, type=str,
                        help=('validity start of the binning tables'
                              ' (format: yyyymmddTHHMMSS)'))
    parser.add_argument('--table_id', type=str, default='1',
                        help='start value table_id or comma seperated list')
    parser.add_argument('--lineskip', default=None,
                        help='provide lineskip arrays per binning table')
    parser.add_argument('file_list', nargs='+',
                        help='provide binning table(s) in csv-format')
    args = parser.parse_args()
    if args.verbose:
        print(args)

    # define a table_id per binning table
    table_id_list = args.table_id.split(',')
    if len(table_id_list) == 1:
        offs = int(args.table_id)
        table_id_list = [offs + x for x in range(len(args.file_list))]
    else:
        table_id_list = [int(x) for x in table_id_list]
    if len(table_id_list) != len(args.file_list):
        raise KeyError('you should provide one table_id per binning table')

    # read lineskip data
    lineskip_data = np.loadtxt(args.lineskip, delimiter=',', dtype=np.uint8)
    if lineskip_data.ndim == 1:
        lineskip_data = lineskip_data[np.newaxis, :]
    elif lineskip_data.shape[0] > len(args.file_list):
        lineskip_data = lineskip_data[1:, :]

    # create BinningTable object
    bin_ckd = BinningTables(ckd_dir='.')
    bin_ckd.create_if_needed(args.validity_start)

    # add binning tables
    for ii, flname in enumerate(args.file_list):
        flpath = Path(flname)
        if not flpath.is_file():
            raise FileNotFoundError(flname)
        if flpath.suffix != '.csv':
            continue

        # read csv-file
        table = np.loadtxt(flpath, delimiter=',', dtype=np.uint32)

        # write new binning-table
        bin_ckd.add_table(table_id_list[ii], lineskip_data[ii, :], table)


# --------------------------------------------------
if __name__ == '__main__':
    main()
