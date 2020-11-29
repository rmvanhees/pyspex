#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Add OCAL EGSE information to a SPEXone Level-1A product

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from datetime import datetime, timezone
# from pathlib import Path

import h5py
import numpy as np

# - global parameters ------------------------------
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LEAP_SECONDS = 27


# - local functions --------------------------------
def byte_to_timestamp(str_date: str):
    """
    Helper function for numpy.loadtxt() to convert byte-string to timestamp
    """
    buff = str_date.strip().decode('ascii')
    return datetime.strptime(buff, '%Y-%j-%H:%M:%S.%f').timestamp()


def read_egse(egse_file: str, verbose=False):
    """
    Read EGSE data to numpy compound array
    """
    formats = ('f8',) + 14 * ('f4',) + ('S7',) + 2 * ('i4',)\
        + 5 * ('f4', 'i2',) + 6 * ('i2',) + ('S5', 'i2')
    if verbose:
        print(len(formats), formats)

    with open(egse_file, 'r') as fid:
        line = None
        while not line:
            line = fid.readline().strip()
            names = line.replace('\t', '').split(',')
            names = [x.strip() for x in names]
            while names.count(''):
                names.remove('')
        # Temporary fix
        names = tuple(names[:21])\
            + ('TMC_POS_1 [deg]', 'TMC_MOVING_1',
               'TMC_POS_2 [deg]', 'TMC_MOVING_2')\
               + tuple(names[21:])
        if verbose:
            print(len(names), names)

        data = np.loadtxt(fid, delimiter=',',
                          dtype={'names': names, 'formats': formats},
                          converters={0: byte_to_timestamp})
    return data


# - main function ----------------------------------
def main():
    """
    Main function
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='add OCAL EGSE information to a SPEXone Level-1A product')
    parser.add_argument('file_list', nargs='+',
                        help="provide names of one or more EGSE files (CSV)")
    parser.add_argument('--l1a_file', default=None, type=str,
                        help="SPEXone L1A product to add/replace EGSE info")
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    if args.verbose:
        print(args)

    # open SPEXone Level-1A product to obtain coverage time (UTC)
    if args.l1a_file is not None:
        with h5py.File(args.l1a_file, 'r') as fid:
            buff = datetime.fromisoformat(
                fid.attrs['time_coverage_start'].decode('ascii'))
            print(buff.timestamp())
            buff = datetime.fromisoformat(
                fid.attrs['time_coverage_end'].decode('ascii'))
            print(buff.timestamp())

    # loop over files with EGSE info until we find a match
    for egse_file in args.file_list:
        egse = read_egse(egse_file, verbose=args.verbose)
        print(egse['time'].min(), egse['time'].max())


# --------------------------------------------------
if __name__ == '__main__':
    main()
