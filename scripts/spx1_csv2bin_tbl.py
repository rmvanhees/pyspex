#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Convert SPEXone binning table in csv format to netCDF4

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from pathlib import Path

import numpy as np

# from pys5p.s5p_plot import S5Pplot
# from pys5p.tol_colors import tol_cmap

from pyspex.binning_tables import BinningTables


# - global parameters ------------------------------


# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """
    main program to build a database with SPEXone binning tables
    """
    parser = argparse.ArgumentParser(
        description='build a database with SPEXone binning tables')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='be verbose, default be silent') 
    parser.add_argument('--db_file', default=None,
                        help='append data to existing binning databse')
    parser.add_argument('--table_id', type=str, default='0',
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
    lineskip_data = lineskip_data[1:, :]
    
    # create BinningTable object
    if args.db_file is None:
        bin_ckd = BinningTables(mode='write', ckd_dir='.')
    else:
        bin_ckd = BinningTables(mode=args.db_file, ckd_dir='.')

    # add binning tables
    for ii in range(len(args.file_list)):
        flpath = Path(args.file_list[ii])
        if not flpath.is_file():
            raise FileNotFoundError(args.file_list[ii])
        if flpath.suffix != '.csv':
            continue

        # read csv-file
        table = np.loadtxt(flpath, delimiter=',', dtype=np.uint32)

        # write new binning-table
        bin_ckd.add_table(table_id_list[ii], lineskip_data[ii, :], table)


# --------------------------------------------------
if __name__ == '__main__':
    main()
