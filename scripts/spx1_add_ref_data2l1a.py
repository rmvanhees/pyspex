#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Add OCAL reference detector and avantes information to a SPEXone L1A product

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
from netCDF4 import Dataset


# - local functions --------------------------------
def byte_to_timestamp(str_date: str):
    """
    Helper function for numpy.loadtxt() to convert byte-string to timestamp
    """
    buff = str_date.strip().decode('ascii') + '+00:00'     # date is in UTC
    return datetime.strptime(buff, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()


def read_ref_det(flname: str, verbose=False) -> tuple:
    """
    Read reference detector data to numpy compound array
    (comma separated values)
    """
    tmpfile = Path('/dev/shm/tmp_ref_det.csv')
    data = None
    names = []
    units = []
    usecols = ()

    with open(flname, 'r') as fid:
        while True:
            line = fid.readline().strip()
            if not line.startswith('Unix'):
                continue

            fields = line.split(',')
            for ii, field in enumerate(fields):
                if field == '':
                    continue
                res = field.strip().split(' (')
                if res[0] in names:
                    continue
                names.append(res[0])
                if len(res) == 2:
                    units.append(res[1].replace('(', '').replace(')', ''))
                else:
                    units.append('1')
                usecols += (ii,)
            break

        # define dtype of the data
        names = [xx.replace(' ', '_') for xx in names]
        formats = len(names) * ('f8',)

        if verbose:
            print(len(names), names)
            print(len(formats), formats)
            print(len(units), units)
            print(len(usecols), usecols)
            # return (None, None)

        with open(tmpfile, 'w') as fid_tmp:
            while True:
                line = fid.readline().strip()
                if not line:
                    break
                if line[0] == ',':
                    continue
                fields = line.split(',')
                fid_tmp.write(','.join([x if x else '0'for x in fields]))
                fid_tmp.write('\n')

    with open(tmpfile, 'r') as fid_tmp:
        data = np.loadtxt(fid_tmp, delimiter=',', usecols=usecols,
                          dtype={'names': names, 'formats': formats})
    tmpfile.unlink()
    if verbose:
        print('data :', len(data))

    return (data, units)


def read_avantes(file_list: list, verbose=False) -> tuple:
    """
    Read reference detector data to numpy compound array
    (comma separated values)
    """
    names = ('timestamp', 'spectrum')
    formats = None
    data = None
    wavelength = None

    for flname in file_list:
        with open(flname, 'r') as fid:
            while True:
                line = fid.readline().strip()
                if not line.startswith('Timestamp'):
                    continue

                if wavelength is None:
                    fields = line.split(',')
                    wavelength = np.array(fields[1:], dtype=float)

                    # define dtype of the data
                    formats = ('f8', '{}f8'.format(len(wavelength)))

                    if verbose:
                        print(len(names), names)
                        print(len(formats), formats)
                        # return (None, None)
                break

            res = np.loadtxt(fid, delimiter=',',
                             converters={0: byte_to_timestamp},
                             dtype={'names': names, 'formats': formats})

        if verbose:
            print('data :', len(res))

        data = res if data is None else np.concatenate((data, res))

    return (data, wavelength)


def create_db_source(db_name, ref_data, ref_units, avantes_data, avantes_wv):
    """
    Write Reference detector and Avantes data to HDF5 database
    """
    with Dataset(db_name, 'w', format='NETCDF4') as fid:
        fid.creation_date = \
            datetime.now(timezone.utc).isoformat(timespec='seconds')

        # write Reference Detector data
        time_key = 'Unix_Time'
        gid = fid.createGroup("ReferenceDetector")
        dset = gid.createDimension('time', ref_data.size)
        dset = gid.createVariable('time', 'f8', ('time',),
                                  chunksizes=(512,))
        indx = np.argsort(ref_data[time_key])
        dset[:] = ref_data[time_key][indx]

        ref_t = gid.createCompoundType(ref_data.dtype, 'ref_dtype')
        dset = gid.createVariable('ref_det', ref_t, ('time',),
                                  chunksizes=(64,))
        dset.long_name = 'Reference detector data'
        dset.units = ref_units
        dset[:] = ref_data[indx]

        # write Avantes data
        time_key = 'timestamp'
        gid = fid.createGroup("Avantes")
        dset = gid.createDimension('time', avantes_data.size)
        dset = gid.createVariable('time', 'f8', ('time',),
                                  chunksizes=(512,))
        indx = np.argsort(avantes_data[time_key])
        dset[:] = avantes_data[time_key][indx]
        dset = gid.createDimension('wave', avantes_wv.size)
        dset = gid.createVariable('wave', 'f8', ('wave',))
        dset[:] = avantes_wv
        avantes_t = gid.createCompoundType(avantes_data.dtype, 'avantes_dtype')
        dset = gid.createVariable('avantes', avantes_t, ('time',),
                                  chunksizes=(64,))
        dset.long_name = 'Avantes data'
        dset[:] = avantes_data[indx]


# --------------------------------------------------
def write_ref_data(l1a_file: str, ref_db: str):
    """
    Select reference data taken during a measurement and add to a L1A product
    """
    # determine duration of the measurement (ITOS clock)
    with h5py.File(l1a_file, 'r') as fid:
        # pylint: disable=no-member
        msmt_start = datetime.fromisoformat(
            fid.attrs['time_coverage_start'].decode('ascii'))
        msmt_stop = datetime.fromisoformat(
            fid.attrs['time_coverage_end'].decode('ascii'))
        print(fid.attrs['time_coverage_start'].decode('ascii'),
              fid.attrs['time_coverage_end'].decode('ascii'))

    # correct msmt_start and msmt_stop to SRON clock
    if Path('/array/slot1F/spex_one/OCAL/date_stats').is_dir():
        data_dir = Path('/array/slot1F/spex_one/OCAL/date_stats')
    elif Path('/data/richardh/SPEXone/OCAL/date_stats').is_dir():
        data_dir = Path('/data/richardh/SPEXone/OCAL/date_stats')
    else:
        data_dir = Path('/nfs/SPEXone/OCAL/date_stats')
    print(data_dir / 'cmp_date_egse_itos2.txt')
    with open(data_dir / 'cmp_date_egse_itos2_clean.txt', 'r') as fid:
        names = fid.readline().strip().lstrip(' #').split('\t\t\t')
        formats = len(names) * ('f8',)
        print(len(names), names)
        cmp_date = np.loadtxt(fid, dtype={'names': names, 'formats': formats})

    cmp_date['ITOS'] -= 0.35
    indx = np.argsort(np.abs(cmp_date['ITOS'] - msmt_start.timestamp()))[0]
    t_diff = cmp_date['SRON(1)'][indx] - cmp_date['ITOS'][indx]
    print('T_shogun - T_itos [sec]:', t_diff)

    # open database with reference data (SRON clock)
    with Dataset(ref_db, 'r') as fid:
        gid = fid['ReferenceDetector']
        ref_time = gid['time'][:] - t_diff
        indx = np.where((ref_time >= msmt_start.timestamp())
                        & (ref_time <= msmt_stop.timestamp()))[0]
        if indx.size == 0:
            raise RuntimeError('no reference detector data found')
        # print(indx)

        ref_time = gid['time'][indx[0]:indx[-1]+1] - t_diff
        ref_det = gid['ref_det'][indx[0]:indx[-1]+1]

        gid = fid['Avantes']
        avantes_time = gid['time'][:] - t_diff
        indx = np.where((avantes_time >= msmt_start.timestamp())
                        & (avantes_time <= msmt_stop.timestamp()))[0]
        if indx.size == 0:
            raise RuntimeError('no Avantes data found')
        print(indx)

        avantes_time = gid['time'][indx[0]:indx[-1]+1] - t_diff
        avantes = gid['avantes'][indx[0]:indx[-1]+1]
        # print('avantes ', avantes.shape)
        avantes_wv = gid['wave'][:]
        # print('avantes_wv ', avantes_wv.shape)

        # update Level-1A product with OGSE/EGSE information
        with Dataset(l1a_file, 'r+') as fid2:
            gid = fid2['/gse_data']
            subgid = gid.createGroup('ReferenceDetector')
            _ = subgid.createDimension('time', len(ref_det))
            dset = subgid.createVariable('time', 'f8', ('time',))
            # print(dset.shape, ref_time.shape)
            dset[:] = ref_time

            ref_t = subgid.createCompoundType(ref_det.dtype, 'ref_dtype')
            dset = subgid.createVariable('ref_det', ref_t, ('time',))
            dset.setncatts(fid['/ReferenceDetector/ref_det'].__dict__)
            dset[:] = ref_det

            subgid = gid.createGroup('Avantes')
            _ = subgid.createDimension('time', len(avantes))
            dset = subgid.createVariable('time', 'f8', ('time',))
            dset[:] = avantes_time

            avantes_t = subgid.createCompoundType(avantes.dtype,
                                                  'avantes_dtype')
            dset = subgid.createVariable('avantes', avantes_t, ('time',))
            dset.setncatts(fid['/Avantes/avantes'].__dict__)
            dset[:] = avantes
            _ = subgid.createDimension('wave', len(avantes_wv))
            dset = subgid.createVariable('wave', 'f8', ('wave',))
            dset[:] = avantes_wv


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
                                     help="create new reference database")
    parser_a.add_argument('--db_name', default='reference_db.nc', type=str,
                          help="name of reference database (HDF5)")
    parser_a.add_argument('--avantes', nargs='+',
                          help="provide name of avantes files (CSV)")
    parser_a.add_argument('--ref_det', default=None,
                          help="provide name of reference detector file (CSV)")
    parser_b = subparsers.add_parser('update',
                                     help=("add reference information"
                                           " to a SPEXone Level-1A product"))
    parser_b.add_argument('--db_name', default='reference_db.nc', type=str,
                          help="name of reference database (HDF5)")
    parser_b.add_argument('l1a_file', default=None, type=str,
                          help="SPEXone L1A product")
    args = parser.parse_args()
    if args.verbose:
        print(args)

    if 'avantes' in args:
        # read reference detector data
        ref_det, ref_units = read_ref_det(args.ref_det, args.verbose)
        # read avantes data
        avantes, avantes_wv = read_avantes(args.avantes, args.verbose)
        # store reference data in HDF5 database
        create_db_source(args.db_name, ref_det, ref_units, avantes, avantes_wv)
    else:
        write_ref_data(args.l1a_file, args.db_name)


# --------------------------------------------------
if __name__ == '__main__':
    main()
