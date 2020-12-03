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

from datetime import datetime, timedelta, timezone
# from pathlib import Path

import h5py
from netCDF4 import Dataset
import numpy as np

from pyspex.lv1_io import LV1mps

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
    # enumerate source status
    ldls_dict = {b'UNPLUGGED': 0, b'Controller Fault': 1, b'Idle': 2,
                 b'Laser ON': 3, b'Lamp ON': 4, b'Missing': 255}

    # enumerate shutter positions
    shutter_dict = {b'CLOSE': 0, b'OPEN': 1, b'Missing': 255}

    # define dtype of the data 
    formats = ('f8',) + 14 * ('f4',) + ('u1',) + 2 * ('i4',)\
        + 5 * ('f4', 'u1',) + 6 * ('u1',) + ('u1',) + ('u1',)
    if verbose:
        print(len(formats), formats)

    with open(egse_file, 'r') as fid:
        line = None
        while not line:
            line = fid.readline().strip()
            fields = line.replace('\t', '').split(',')
            names = []
            units = []
            for field in fields:
                if field == '':
                    continue
                res = field.strip().split(' [')
                names.append(res[0].replace(' nm', 'nm'))
                if len(res) == 2:
                    units.append(res[1].replace('[', '').replace(']', ''))
                else:
                    units.append('')

        # Temporary fix
        names = tuple(names[:21])\
            + ('TMC_POS_1', 'TMC_MOVING_1', 'TMC_POS_2', 'TMC_MOVING_2')\
            + tuple(names[21:])
        units = tuple(units[:21]) + ('deg', '', 'deg', '') + tuple(units[21:])
        if verbose:
            print(len(names), names)
            print(len(units), units)

        data = np.loadtxt(fid, delimiter=',',
                          dtype={'names': names, 'formats': formats},
                          converters={0: byte_to_timestamp,
                                      15: lambda s: ldls_dict[s.strip()],
                                      34: lambda s: shutter_dict[s.strip()]})
    return {'values': data, 'units': units,
            'ldls_dict': ldls_dict, 'shutter_dict': shutter_dict}

def select_egse(l1a_file: str, egse, verbose=False):
    """
    Return indices EGSE records during the measurement in the Level-1A product
    """
    with h5py.File(l1a_file, 'r') as fid:
        ccsds_sec = fid['image_attributes/image_CCSDS_sec'][()]
        ccsds_usec = fid['image_attributes/image_CCSDS_usec'][()]
        ccsds_time = [datetime(1970, 1, 1, tzinfo=timezone.utc)
                      + timedelta(seconds=int(ccsds_sec[0]))
                      + timedelta(microseconds=int(ccsds_usec[0])),
                      datetime(1970, 1, 1, tzinfo=timezone.utc)
                      + timedelta(seconds=int(ccsds_sec[-1]))
                      + timedelta(microseconds=int(ccsds_usec[-1]))]
        image_time = fid['image_attributes/image_time'][()]
        image_time = [datetime(2020, 11, 28, tzinfo=timezone.utc)
                      + timedelta(seconds=float(image_time[0])),
                      datetime(2020, 11, 28, tzinfo=timezone.utc)
                      + timedelta(seconds=float(image_time[-1]))]
        mps = LV1mps(fid['/science_data/detector_telemetry'][0])

    print(ccsds_time)
    print(image_time)
    print(1e-1 * mps.get('REG_NCOADDFRAMES') * mps.get('FTI'))
    print(mps.frame_period * 1e-7)
    print(image_time[0] - timedelta(seconds=float(mps.frame_period * 1e-7)))
    return

def write_egse(l1a_file: str, egse, verbose=False):
    """
    Read EGSE data to SPEXone Level-1A product
    """
    if l1a_file is None:
        fid = Dataset('test_egse.nc', 'w', format='NETCDF4')
        gid = fid.createGroup('gse_data')
    else:
        fid = Dataset(l1a_file, 'r+', format='NETCDF4')
        gid = fid['/gse_data']

    ldls_t = gid.createEnumType('u1','ldls_t',
                                {k.upper(): v
                                 for k, v in egse['ldls_dict'].items()})
    shutter_t = gid.createEnumType('u1','shutter_t',
                                   {k.upper(): v
                                    for k, v in egse['shutter_dict'].items()})
    dim_egse = gid.createDimension("egse_packets", egse['values'].size)
    egse_t = gid.createCompoundType(egse['values'].dtype, 'egse_dtype')
    var_egse = gid.createVariable("egse", egse_t, ("egse_packets",))
    var_egse.long_name = 'EGSE settings'
    var_egse.units = egse['units']
    var_egse.comment = ('DIG_IN_00 is of enumType ldls_t;'
                        ' SHUTTER_STATUS is of enumType shutter_t')
    var_egse[:] = egse['values']

    fid.close()
    

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
            print(buff, buff.timestamp())
            buff = datetime.fromisoformat(
                fid.attrs['time_coverage_end'].decode('ascii'))
            print(buff, buff.timestamp())

    # loop over files with EGSE info until we find a match
    for egse_file in args.file_list:
        egse = read_egse(egse_file, verbose=args.verbose)
    print(egse['values']['time'].min(), egse['values']['time'].max())

    if args.l1a_file is not None:
        select_egse(args.l1a_file, egse, verbose=args.verbose)
        return

    write_egse(args.l1a_file, egse, verbose=args.verbose)


# --------------------------------------------------
if __name__ == '__main__':
    main()
