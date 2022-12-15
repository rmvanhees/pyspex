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
Add ITOS EGSE information of OCAL measurements to a SPEXone level-1A product.
"""

import argparse

from pathlib import Path

from pyspex.egse_db import add_egse_data, create_egse_db

# - global parameters ------------------------------


# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """Main function."""
    # parse command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', help="be verbose")
    parser.add_argument('--egse_dir', default='Logs', type=Path,
                        help="directory with EGSE data")
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_db = subparsers.add_parser('create_db',
                                      help="create new EGSE database")
    parser_db.add_argument('file_list', nargs='+',
                           help="provide names EGSE files (CSV)")
    parser_db.set_defaults(func=create_egse_db)

    parser_wr = subparsers.add_parser('add',
                                      help=("add EGSE information to a"
                                            " SPEXone level-1A product"))
    parser_wr.add_argument('l1a_file', default=None, type=str,
                           help="SPEXone L1A product")
    parser_wr.set_defaults(func=add_egse_data)
    args = parser.parse_args()
    if args.verbose:
        print(args)

    # call whatever function was selected
    args.func(args)


# --------------------------------------------------
if __name__ == '__main__':
    main()
