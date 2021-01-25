#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Add OCAL OGSE/EGSE information to a SPEXone Level-1A product

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from datetime import datetime, timedelta, timezone
from pathlib import Path
import pickle

import h5py
from netCDF4 import Dataset
import numpy as np

# - global parameters ------------------------------
# enumerate source status
LDLS_DICT = {b'UNPLUGGED': 0, b'Controller Fault': 1, b'Idle': 2,
             b'Laser ON': 3, b'Lamp ON': 4, b'MISSING': 255}

# enumerate shutter positions
SHUTTER_DICT = {b'CLOSED': 0, b'OPEN': 1, b'PARTIAL': 255}


# - local functions --------------------------------
def byte_to_timestamp(str_date: str):
    """
    Helper function for numpy.loadtxt() to convert byte-string to timestamp
    """
    buff = str_date.strip().decode('ascii') + '+00:00' # date is in UTC
    return datetime.strptime(buff, '%Y%m%dT%H%M%S.%f%z').timestamp()


def read_egse(egse_file: str, verbose=False) -> tuple:
    """
    Read OGSE/EGSE data (tab separated values) to numpy compound array
    """
    # define dtype of the data
    formats = ('f8',) + 14 * ('f4',) + ('u1',) + 2 * ('i4',)\
        + ('f4', 'u1',) + 2 * ('u1',) + 3 * ('f4', 'u1',) + 7 * ('u1',)

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
                names.append(res[0].replace(' nm', 'nm').replace(' ', '_'))
                if len(res) == 2:
                    units.append(res[1].replace('[', '').replace(']', ''))
                else:
                    units.append('1')

        if 'NOMHK_packets_time' in names:
            formats = ('f8',) + formats
            convertors = {0: byte_to_timestamp,
                          1: byte_to_timestamp,
                          16: lambda s: LDLS_DICT.get(s.strip(), 255),
                          21: lambda s: SHUTTER_DICT.get(s.strip(), 255)}
        else:
            convertors = {0: byte_to_timestamp,
                          15: lambda s: LDLS_DICT.get(s.strip(), 255),
                          20: lambda s: SHUTTER_DICT.get(s.strip(), 255)}
        if verbose:
            print(len(names), names)
            print(len(formats), formats)
            print(len(units), units)

        data = np.loadtxt(fid, delimiter='\t', converters=convertors,
                          dtype={'names': names, 'formats': formats})

    return (data, units)


def write_ref_spectrum(gid):
    """
    Write reference spectrum to group /gse_data
    """
    flname = '/nfs/SPEXone/share/ckd/reference_laser_source.pickle'
    if not Path(flname).is_file():
        raise FileNotFoundError('file reference_laser_source.pickle not found')

    with open(flname, 'rb') as fp:
        data = pickle.load(fp)

    _ = gid.createDimension('wavelength', len(data['wavelength']))
    dset = gid.createVariable('wavelength', 'f8', ('wavelength',))
    dset.long_name = 'wavelength of stimulus'
    dset.units = data['units'][0]
    dset[:] = data['wavelength']

    dset = gid.createVariable('signal', 'f8', ('wavelength',))
    dset.long_name = 'signal of stimulus'
    dset.units = data['units'][1]
    dset[:] = data['signal']


def create_db_egse(db_name, egse_data, egse_units):
    """
    Write OGSE/EGSE data to HDF5 database
    """
    time_key = 'ITOS_time' if 'ITOS_time' in egse_data.dtype.names else 'time'

    with Dataset(db_name, 'w', format='NETCDF4') as fid:
        fid.creation_date = \
            datetime.now(timezone.utc).isoformat(timespec='seconds')

        _ = fid.createEnumType('u1','ldls_t',
                               {k.replace(b' ', b'_').upper(): v
                                for k, v in LDLS_DICT.items()})
        _ = fid.createEnumType('u1','shutter_t',
                               {k.upper(): v
                                for k, v in SHUTTER_DICT.items()})
        var_time = fid.createDimension('time', egse_data.size)

        var_time = fid.createVariable('time', 'f8', ('time',),
                                      chunksizes=(512,))
        indx = np.argsort(egse_data[time_key])
        var_time[:] = egse_data[time_key][indx]

        egse_t = fid.createCompoundType(egse_data.dtype, 'egse_dtype')
        var_egse = fid.createVariable('egse', egse_t, ('time',),
                                      chunksizes=(64,))
        var_egse.long_name = 'OGSE/EGSE settings'
        var_egse.units = egse_units
        var_egse.comment = ('DIG_IN_00 is of enumType ldls_t;'
                            ' SHUTTER_STATUS is of enumType shutter_t')
        var_egse[:] = egse_data[indx]


# --------------------------------------------------
def check_egse(egse_data, act_angle, alt_angle):
    """
    Check consistency of OGSE/EGSE information during measurement
    """
    for key, fmt in egse_data.dtype.fields.items():
        if fmt[0] == np.uint8:
            res_sanity = (egse_data[key] == egse_data[key][0]).all()
            if not res_sanity:
                print('[WARNING] ', key, egse_data[key])

    if act_angle:
        if not np.allclose(egse_data['ACT_ANGLE'], act_angle[0], 1e-2):
            print('[WARNING] ', 'ACT_ANGLE', egse_data['ACT_ANGLE'])
    if alt_angle:
        if not np.allclose(egse_data['ALT_ANGLE'], alt_angle[0], 1e-2):
            print('[WARNING] ', 'ALT_ANGLE', egse_data['ALT_ANGLE'])


def select_egse(l1a_file: str, egse_file: str, add_ref_laser_spectra: bool):
    """
    Write OGSE/EGSE records of a measurement to a Level-1A product
    """
    view_dict = {'M50DEG': 1, 'M20DEG': 2, '0DEG': 4, 'P20DEG': 8, 'P50DEG': 16}

    # investigate filename
    parts = l1a_file.split('_')
    vp_parts = parts[2].split('-')

    # default 0, when all viewports are illuminated
    viewport = view_dict.get(vp_parts[min(2, len(vp_parts))], 0)
    act_angle = [float(x.replace('act', ''))
                 for x in parts if x.startswith('act')]
    alt_angle = [float(x.replace('alt', ''))
                 for x in parts if x.startswith('alt')]

    # determine duration of the measurement
    with h5py.File(l1a_file, 'r') as fid:
        # pylint: disable=no-member
        msmt_start = datetime.fromisoformat(
            fid.attrs['time_coverage_start'].decode('ascii'))
        msmt_stop = datetime.fromisoformat(
            fid.attrs['time_coverage_end'].decode('ascii'))
        # print(fid.attrs['time_coverage_start'].decode('ascii'),
        #      fid.attrs['time_coverage_end'].decode('ascii'))
        duration = np.ceil((msmt_stop - msmt_start).total_seconds())

        # use the timestamp in the filename
        # pylint: disable=unsubscriptable-object
        input_file = Path(fid.attrs['input_files'][0]).stem.rstrip('_hk')
        date_str = input_file.split('_')[-1] + "+00:00"
        msmt_start = datetime.strptime(date_str, "%Y%m%dT%H%M%S.%f%z")
        msmt_start = msmt_start.replace(microsecond=0)
        msmt_stop = msmt_start + timedelta(seconds=int(duration))
        # print(msmt_start, msmt_stop)

    # open OGSE/EGSE database
    with Dataset(egse_file, 'r') as fid:
        egse_time = fid['time'][:].data
        indx = np.where((egse_time >= msmt_start.timestamp())
                        & (egse_time <= msmt_stop.timestamp()))[0]
        if indx.size == 0:
            raise RuntimeError('no OGSE/EGSE data found')

        egse_data = fid['egse'][indx[0]:indx[1]+1]
        # perform sanity check
        check_egse(egse_data, act_angle, alt_angle)

        # update Level-1A product with OGSE/EGSE information
        with Dataset(l1a_file, 'r+') as fid2:
            gid = fid2['/gse_data']
            gid['viewport'][:] = viewport

            # add OGSE/EGSE information
            _ = gid.createDimension('time', len(egse_data))
            egse_t = gid.createCompoundType(egse_data.dtype, 'egse_dtype')
            var_egse = gid.createVariable('egse', egse_t, ('time',))
            var_egse.setncatts(fid['egse'].__dict__)
            var_egse[:] = egse_data

            # write reference spectra
            if add_ref_laser_spectra:
                write_ref_spectrum(gid)

            # write sub-set of OGSE/EGSE settings as attributes
            gid.Line_skip_id = ""
            gid.Enabled_lines = np.uint16(2048)
            gid.Binning_table = ""
            gid.Binned_pixels = np.uint32(0)
            gid.Light_source = parts[2]
            gid.FOV_begin = np.nan
            gid.FOV_end = np.nan
            gid.ACT_rotationAngle = np.nan if not act_angle else act_angle[0]
            gid.ALT_rotationAngle = np.nan if not alt_angle else alt_angle[0]
            gid.ACT_illumination = np.nan
            gid.ALT_illumination = np.nan
            gid.DoLP = 0.
            gid.AoLP = 0.


# - main function ----------------------------------
def main():
    """
    Main function
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true', default=False)
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_a = subparsers.add_parser('create_db',
                                     help="create new OGSE/EGSE database")
    parser_a.add_argument('--db_name', default='egse_database.nc', type=str,
                          help="name of OGSE/EGSE database")
    parser_a.add_argument('file_list', nargs='+',
                          help="provide names of one or more EGSE files (CSV)")
    parser_b = subparsers.add_parser('update',
                                     help=("add OGSE/EGSE information"
                                           " to a SPEXone Level-1A product"))
    parser_b.add_argument('--egse_db', default='egse_database.nc', type=str,
                          help="OGSE/EGSE database (HDF5)")
    parser_b.add_argument('--add_ref_laser_spectra', action='store_true',
                          default=True)
    parser_b.add_argument('l1a_file', default=None, type=str,
                          help="SPEXone L1A product")
    args = parser.parse_args()
    if args.verbose:
        print(args)

    if 'file_list' in args:
        egse = None
        for egse_file in args.file_list:
            res = read_egse(egse_file, verbose=args.verbose)
            egse = res[0] if egse is None else np.concatenate((egse, res[0]))
            units = res[1]

        create_db_egse(args.db_name, egse, units)
        return

    select_egse(args.l1a_file, args.egse_db, args.add_ref_laser_spectra)


# --------------------------------------------------
if __name__ == '__main__':
    main()
