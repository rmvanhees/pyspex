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
from pathlib import Path

from pyspex.ogse_db import (read_ref_diode, read_wav_mon,
                            add_ogse_ref_diode, add_ogse_wav_mon)
from pyspex.ogse_helios import add_ogse_helios

# - global parameters ------------------------------
DB_REF_DIODE = 'ogse_db_ref_diode.nc'
DB_WAV_MON = 'ogse_db_wave_mon.nc'


# - local functions --------------------------------
def create_ogse_db(args):
    """
    Create databases for reference diode and/or Avantes fibre spectrometer
    """
    if args.ref_diode:
        # read reference-diode data
        xds = read_ref_diode(args.ogse_dir, args.ref_diode, args.verbose)

        # create new database for reference-diode data
        xds.to_netcdf(args.ogse_dir / DB_REF_DIODE,
                      mode='w', format='NETCDF4',
                      group='/gse_data/ReferenceDiode')

    if args.wav_mon:
        # read reference-diode data
        xds = read_wav_mon(args.ogse_dir, args.wav_mon, args.verbose)
        # create new database for reference-diode data
        xds.to_netcdf(args.ogse_dir / DB_WAV_MON,
                      mode='w', format='NETCDF4',
                      group='/gse_data/WaveMonitor')


def write_ogse(args):
    """
    Add OGSE data to a SPEXone level-1A product
    """
    if args.ref_diode:
        add_ogse_ref_diode(args.ogse_dir / DB_REF_DIODE, args.l1a_file)
    if args.avantes:
        add_ogse_wav_mon(args.ogse_dir / DB_WAV_MON, args.l1a_file)
    if args.helios:
        add_ogse_helios(args.l1a_file)
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
    parser.add_argument('--ogse_dir', default='Logs', type=Path,
                        help="directory with OGSE data")
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
