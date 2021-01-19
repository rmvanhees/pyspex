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
    main program to generate a netCDF4 file with a SPEXone binning table
    """
    parser = argparse.ArgumentParser(
        description=('create SPEXone L1A product from DEM measurement(s)'))
    parser.add_argument('--figure', default=False, action='store_true',
                        help='generate (PDF) figure of binning table(s)')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('file_list', nargs='+',
                        help=("provide path to SPEXone binning table(s)"
                              " in csv-format"))
    args = parser.parse_args()
    if args.verbose:
        print(args)

    bin_ckd = BinningTables(mode='write', ckd_dir='.')

    # add binning tables
    for flname in args.file_list:
        flpath = Path(flname)
        if not flpath.is_file():
            raise FileNotFoundError(flname)
        if flpath.suffix != '.csv':
            continue

        # read csv-file
        table = np.loadtxt(flname, delimiter=',', dtype=np.uint32)
        bin_ckd.add_table(table)


# --------------------------------------------------
if __name__ == '__main__':
    main()
