#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Add OGSE information of a OCAL measurement to a SPEXone L1A product.

Possible OGSE information:
 * reference diode (Ambient/GSFC polarized, radiometric): the reference diode
 is located in the ... or Grande (GSFC) integrated sphere
 * wavelength monitor (Ambient/GSFC polarized, radiometric, wavelength)
 * wavelength of OPO laser (GSFC spectral-radiometry)

Copyright (c) 2021-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

# from pyspex.lv1_gse import LV1gse
from pyspex.ogse_db import create_ref_diode_db, create_wav_mon_db
# from pyspex.ogse_helios import add_ogse_helios


def create_ogse_db(args):
    """
    Create HDF5 databases of reference diode data and/or
    Avantes fibre spectrometer
    """
    print(args)
    if args.ref_diode:
        create_ref_diode_db(args.ref_diode, verbose=args.verbose)
    if args.wav_mon:
        create_wav_mon_db(args.ref_diode, verbose=args.verbose)


def write_ogse(args):
    """
    Add OGSE data to a SPEXone level-1A product
    """
    print(args)
    if args.ref_diode:
        pass
    if args.avantes:
        pass
    if args.helios:
        pass
    if args.grande:
        pass
    if args.opo_laser:
        pass


# - main function ----------------------------------
def main():
    """
    Main function
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', default=False)

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_db = subparsers.add_parser('create_db',
                                      help="create new OGSE database")
    parser_db.add_argument('--ref_diode', nargs='*', default=[],
                           help="names of reference-diode files")
    parser_db.add_argument('--wav_mon', nargs='*', default=[],
                           help="names of Avantes wavelength-monitor files")
    parser_db.set_defaults(func=create_ogse_db)

    parser_wr = subparsers.add_parser('add',
                                       help=("add OGSE information"
                                             " to a SPEXone Level-1A product"))
    parser_wr.add_argument('--ref_diode', action='store_true',
                           help='add reference-diode data from OGSE database')
    parser_wr.add_argument('--avantes', action='store_true',
                           help=('add Avantes wavelength monitoring'
                                 '  from OGSE database'))
    parser_wr.add_argument('--helios', action='store_true',
                           help='add Helios reference spectrum')
    parser_wr.add_argument('--grande', action='store_true',
                           help='add Grande reference spectrum')
    parser_wr.add_argument('--opo_laser', action='store_true',
                           help='add wavelength of OPO laser')
    parser_wr.add_argument('l1a_file', default=None, type=str,
                           help="SPEXone L1A product")
    parser_wr.set_defaults(func=write_ogse)
    args = parser.parse_args()
    if args.verbose:
        print(args)

    # call whatever function was selected
    args.func(args)


# --------------------------------------------------
if __name__ == '__main__':
    main()
