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

from datetime import datetime, timedelta
from pathlib import Path

import h5py
from netCDF4 import Dataset
import numpy as np

# - global parameters ------------------------------


# - local functions --------------------------------
def get_offs_egse_time(sron_sec: float):
    """
    Get offset between ITOS and EGSE hardware clocks
    """
    date_time_dir = Path('/array/slot1F/spex_one/OCAL/date_stats')
    if not date_time_dir.is_dir():
        date_time_dir = Path('/nfs/SPEXone/OCAL/date_stats')
    if not date_time_dir.is_dir():
        date_time_dir = Path('/data/richardh/SPEXone/OCAL/date_stats')

    with open(date_time_dir / 'cmp_date_egse_itos.txt', 'r') as fid:
        names = fid.readline().strip().lstrip(' #').split('\t\t\t')
        formats = len(names) * ('f8',)
        data = np.loadtxt(fid, dtype={'names': names, 'formats': formats})

    indx = np.argmin(np.abs(data['SRON'] - sron_sec))
    # print('argmin: ', indx,
    #      int(data[indx]['EGSE'] - data[indx]['ITOS']),
    #      int(data[indx]['SRON'] - data[indx]['EGSE']),
    #      int(data[indx]['SRON'] - data[indx]['ITOS']))
    return int(data[indx]['EGSE'] - data[indx]['ITOS'])


def byte_to_timestamp(str_date: str):
    """
    Helper function for numpy.loadtxt() to convert byte-string to timestamp
    """
    buff = str_date.strip().decode('ascii') + '+00:00' # date is in UTC
    return datetime.strptime(buff, '%Y%m%dT%H%M%S.%f%z').timestamp()


def read_egse_old(egse_file: str, verbose=False):
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


def read_egse(egse_file: str, verbose=False):
    """
    Read EGSE data (tab separated values) to numpy compound array
    """
    # enumerate source status
    ldls_dict = {b'UNPLUGGED': 0, b'Controller Fault': 1, b'Idle': 2,
                 b'Laser ON': 3, b'Lamp ON': 4, b'MISSING': 255}

    # enumerate shutter positions
    shutter_dict = {b'CLOSED': 0, b'OPEN': 1, b'PARTIAL': 255}

    # define dtype of the data
    formats = ('f8',) + 14 * ('f4',) + ('u1',) + 2 * ('i4',)\
        + ('f4', 'u1',) + 2 * ('u1',) + 3 * ('f4', 'u1',) + 7 * ('u1',)
    if verbose:
        print(len(formats), formats)

    with open(egse_file, 'r') as fid:
        line = None
        while not line:
            line = fid.readline().strip()
            fields = line.split('\t')
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
                    units.append('1')

        if verbose:
            print(len(names), names)
            print(len(units), units)

        data = np.loadtxt(fid, delimiter='\t',
                          dtype={'names': names, 'formats': formats},
                          converters={0: byte_to_timestamp,
                                      15: lambda s: ldls_dict.get(s.strip(),
                                                                  255),
                                      20: lambda s: shutter_dict.get(s.strip(),
                                                                     255)})
    return {'values': data, 'units': units,
            'ldls_dict': ldls_dict, 'shutter_dict': shutter_dict}

def select_egse(l1a_file: str, egse, verbose=False, offset=90):
    """
    Return indices EGSE records during the measurement in the Level-1A product
    """
    with h5py.File(l1a_file, 'r') as fid:
        # pylint: disable=no-member
        coverage_start = datetime.fromisoformat(
            fid.attrs['time_coverage_start'].decode('ascii'))
        coverage_stop = datetime.fromisoformat(
            fid.attrs['time_coverage_end'].decode('ascii'))

    offset = get_offs_egse_time(coverage_start.timestamp()) - 50
    if verbose:
        print('offset: ', offset)
    coverage_start += timedelta(seconds=offset)
    coverage_stop += timedelta(seconds=offset)
    indx = np.where((egse['values']['time'] >= coverage_start.timestamp())
                    & (egse['values']['time'] <= coverage_stop.timestamp()))[0]
    if indx.size == 0:
        return None

    # perform sanity check
    egse['values'] = egse['values'][indx]
    for param in ['ALT_ANGLE', 'ACT_ANGLE']:
        res_sanity = np.all(np.diff(egse['values'][param]) < 0.01)
        if not res_sanity:
            print('[WARNING] ', param, egse['values'][param])

    return egse

def write_egse(l1a_file: str, egse):
    """
    Read EGSE data to SPEXone Level-1A product
    """
    if l1a_file is None:
        fid = Dataset('test_egse.nc', 'w', format='NETCDF4')
        gid = fid.createGroup('gse_data')
    else:
        fid = Dataset(l1a_file, 'r+', format='NETCDF4')
        gid = fid['/gse_data']

    _ = gid.createEnumType('u1','ldls_t',
                           {k.upper(): v
                            for k, v in egse['ldls_dict'].items()})
    _ = gid.createEnumType('u1','shutter_t',
                           {k.upper(): v
                            for k, v in egse['shutter_dict'].items()})
    _ = gid.createDimension("egse_packets", egse['values'].size)
    egse_t = gid.createCompoundType(egse['values'].dtype, 'egse_dtype')
    var_egse = gid.createVariable("egse", egse_t, ("egse_packets",))
    var_egse.long_name = 'EGSE settings'
    var_egse.units = egse['units']
    var_egse.comment = ('DIG_IN_00 is of enumType ldls_t;'
                        ' SHUTTER_STATUS is of enumType shutter_t')
    var_egse[:] = egse['values']

    if l1a_file is None:
        fid.close()
        return

    # investigate filename
    parts = l1a_file.split('_')
    act_angle = [x.replace('act', '') for x in parts if x.startswith('act')]
    alt_angle = [x.replace('alt', '') for x in parts if x.startswith('alt')]

    view_dict = {'M50DEG': 1, 'M20DEG': 2, '0DEG': 4, 'P20DEG': 8, 'P50DEG': 16}
    parts_type = parts[2].split('-')
    gid['viewport'][:] = view_dict.get(parts_type[min(2, len(parts_type))], 255)

    # write EGSE settings as attributes
    gid.Line_skip_id = ""
    gid.Enabled_lines = np.uint16(2048)
    gid.Binning_table = ""
    gid.Binned_pixels = np.uint32(0)
    gid.Light_source = parts[2]
    gid.FOV_begin = np.nan
    gid.FOV_end = np.nan
    gid.ACT_rotationAngle = np.nan if not act_angle else float(act_angle[0])
    gid.ALT_rotationAngle = np.nan if not alt_angle else float(alt_angle[0])
    gid.ACT_illumination = np.nan
    gid.ALT_illumination = np.nan
    gid.DoLP = 0.
    gid.AoLP = 0.

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

    # loop over files with EGSE info until we find a match
    for egse_file in args.file_list:
        egse = read_egse(egse_file, verbose=args.verbose)

    if args.l1a_file is not None:
        egse = select_egse(args.l1a_file, egse, verbose=args.verbose)
        if egse is None:
            raise FileNotFoundError(
                'could not find EGSE information for the measurements')

    write_egse(args.l1a_file, egse)


# --------------------------------------------------
if __name__ == '__main__':
    main()
