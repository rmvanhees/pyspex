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
This module contain routines to read reference diode measurements and
wavelength monitor data. These data are supposed to be written to a HDF5
database. From which collocated data can be added to a SPEXone Level-1A
product.
"""
__all__ = ['read_ref_diode', 'read_wav_mon',
           'add_ogse_ref_diode', 'add_ogse_wav_mon']

from datetime import datetime
from io import StringIO
from pathlib import Path

import h5py
import numpy as np
from xarray import DataArray, Dataset, open_dataset

# - global parameters ------------------------------


# ---------- CREATE OGSE DATABASES ----------
def read_ref_diode(ogse_dir: Path, file_list: list, verbose=False) -> Dataset:
    """
    Read reference diode data into a xarray.Dataset.
    (input: comma separated values)
    """
    data = None
    unit_dict = {"Unix Time (s)": ("seconds", 'f8'),
                 "Excel Time (d)": ("days", 'f4'),
                 "Normalized Time (s)": ("seconds", 'f8'),
                 "amps": ("A", 'f4'), "scaled": ("1", 'f4'),
                 "last_zero": ("seconds", 'f8'), "lamp_on": ("flag", 'b'),
                 "voltage": ("V", 'f4'), "current": ("A", 'f4')}

    fields_to_skip = ('', 'averaging', 'lamp_on_counter', 'record_timestamp',
                      'scale', 'tempC')

    for flname in file_list:
        names = []
        units = []
        formats = []
        usecols = ()
        all_valid_lines = ''
        with open(ogse_dir / flname, 'r', encoding='ascii') as fid:
            while True:
                line = fid.readline().strip()
                if not line.startswith('Unix'):
                    continue

                fields = line.split(',')
                for ii, field in enumerate(fields):
                    if field in fields_to_skip:
                        continue
                    res = field.strip().split(' (')
                    res[0] = res[0].replace(' ', '_')
                    if res[0] in names:
                        continue
                    names.append(res[0])
                    units.append(unit_dict[field][0])
                    formats.append(unit_dict[field][1])
                    usecols += (ii,)
                break

            if verbose:
                print(len(names), names)
                print(len(units), units)
                print(len(formats), formats)
                print(len(usecols), usecols)
                # return (data, units)

            while True:
                line = fid.readline()
                if not line:
                    break
                if ',,,,' not in line:
                    all_valid_lines += line.replace(',,', ',0,')

        # read data (skip first line)
        res = np.loadtxt(StringIO(all_valid_lines), delimiter=',',
                         skiprows=1, usecols=usecols,
                         converters={0: lambda s: float(s) - 3600},
                         dtype={'names': names, 'formats': formats})
        data = res if data is None else np.concatenate((data, res))

    if verbose:
        print('data :', len(data))

    time_key = 'Unix_Time'
    res = {}
    res['time'] = DataArray(
        data[time_key], coords={'time': data[time_key]},
        attrs={'longname': 'time', 'units': 'seconds since 1970-1-1 0:0:0'})
    for ii, key in enumerate(names):
        if key == time_key:
            continue

        res[key] = DataArray(data[key], coords={'time': data[time_key]},
                             attrs={'longname': key, 'units': units[ii]})

    xds = Dataset(res).sortby('time')
    xds.attrs = {'source': 'Calibrated diode inside integrated sphere',
                 'comment': 'Generated on SRON clean-room tablet'}
    return xds


# ---------------
def read_wav_mon(ogse_dir: Path, file_list: list, verbose=False) -> Dataset:
    """
    Read wavelength monitor data into a xarray.Dataset.
    (input comma separated values)
    """
    def byte_to_timestamp(str_date: str) -> datetime:
        """
        Helper function for numpy.loadtxt()
        convert byte-string to timestamp (UTC)

        Parameters
        ----------
        str_date : byte
            isoformated date-time string with timezone CET

        Returns
        -------
        timestamp :  float
            Unix timestamp in UTC
        """
        buff = str_date.strip().decode('ascii') + '+01:00'
        return datetime.strptime(buff, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()

    names = ('timestamp', 'spectrum')
    formats = None
    data = None
    wavelength = None
    t_intg = None
    n_avg = None

    for flname in file_list:
        with open(ogse_dir / flname, 'r', encoding='ascii') as fid:
            while True:
                line = fid.readline().strip()

                if line.startswith('Integration'):
                    fields = line.split(':')
                    scalar_t_intg = float(fields[1])
                    continue

                if line.startswith('Averaging'):
                    fields = line.split(':')
                    scalar_n_avg = int(fields[1])
                    continue

                if line.startswith('Timestamp'):
                    fields = line.split(',')
                    wavelength = np.array(fields[1:], dtype=float)

                    # define dtype of the data
                    formats = ('f8', f'{len(wavelength)}f8')
                    break

            if verbose:
                print(f'Integration time: {t_intg}ms')
                print(f'Averaging Nr.: {n_avg}')
                print(len(names), names)
                print(len(formats), formats)
                # return (None, None)

            res = np.loadtxt(fid, delimiter=',',
                             converters={0: byte_to_timestamp},
                             dtype={'names': names, 'formats': formats})

        if verbose:
            print('data :', len(res))

        if data is None:
            data = res
            t_intg = np.full(res.size, scalar_t_intg)
            n_avg = np.full(res.size, scalar_n_avg)
        else:
            data = np.concatenate((data, res))
            t_intg = np.concatenate((t_intg, np.full(res.size, scalar_t_intg)))
            n_avg = np.concatenate((n_avg, np.full(res.size, scalar_n_avg)))

    time_key = 'timestamp'
    res = {}
    res['time'] = DataArray(
        data[time_key], coords={'time': data[time_key]},
        attrs={'longname': 'time', 'units': 'seconds since 1970-1-1 0:0:0'})
    res['wavelength'] = DataArray(
        wavelength, coords={'wavelength': wavelength},
        attrs={'longname': 'wavelength grid', 'units': 'nm'})
    res['t_intg'] = DataArray(
        t_intg.astype('i2'), dims=['time'],
        attrs={'longname': 'Integration time', 'units': 'nm'})
    res['n_avg'] = DataArray(
        n_avg.astype('i2'), dims=['time'],
        attrs={'longname': 'Averaging number', 'units': '1'})
    res['spectrum'] = DataArray(
        data['spectrum'], dims=['time', 'wavelength'],
        attrs={'longname': 'radiance spectrum', 'units': 'W/(m^2.sr.nm)'})

    xds = Dataset(res).sortby('time')
    xds.attrs = {'source': 'Avantes fibre spectrometer',
                 'comment': 'Generated on SRON clean-room tablet'}
    return xds


# ----- SELECT OGSE DATA FROM DATABASE AND ADD TO L1A PRODUCT -----
def read_date_stats() -> tuple:
    """
    Read output of program 'date' executed at freckle (ITOS) and shogun (SRON)
    """
    if Path('/array/slot1F/spex_one/OCAL/date_stats').is_dir():
        data_dir = Path('/array/slot2B/spex_ocal/ambient/date_stats')
    else:
        data_dir = Path('/nfs/SPEXone/ocal/ambient/date_stats')
    flname = data_dir / 'cmp_date_egse_itos2.txt'

    all_valid_lines = ''
    with open(flname, 'r', encoding='ascii') as fid:
        names = fid.readline().strip().lstrip(' #').split('\t\t\t')
        formats = len(names) * ('f8',)

        while True:
            line = fid.readline()
            if not line:
                break
            fields = line.split('\t')
            if '' not in fields:
                all_valid_lines += line

    res = np.loadtxt(StringIO(all_valid_lines),
                     dtype={'names': names, 'formats': formats})

    # a constant difference between the SRON/ITOS timestamps is introduced
    # by the calls of 'date' via 'ssh'. This difference is estimated at 350ms.
    t_itos = (1000 * res['ITOS'] - 350).astype(int)
    t_sron = (1000 * res['SRON(1)']).astype(int)

    return t_itos.astype('datetime64[ms]'), t_sron.astype('datetime64[ms]')


def clock_offset(l1a_file: Path) -> float:
    """
    Derive offset between msmt_start/msmt_stop and the SRON clock
    """
    # determine duration of the measurement (ITOS clock)
    with h5py.File(l1a_file, 'r') as fid:
        # pylint: disable=unsubscriptable-object
        res = fid.attrs['input_files']
        if isinstance(res, bytes):
            input_file = Path(res.decode('ascii')).stem.rstrip('_hk')
        else:
            input_file = Path(res[0]).stem.rstrip('_hk')
        # pylint: disable=no-member
        msmt_start = np.datetime64(
            fid.attrs['time_coverage_start'].decode('ascii').split('+')[0])
        msmt_stop = np.datetime64(
            fid.attrs['time_coverage_end'].decode('ascii').split('+')[0])

    duration = (msmt_stop - msmt_start).astype('timedelta64[s]') + 1
    # print('duration: ', duration)

    # use the timestamp in the filename to correct ICU time
    date_start = datetime.strptime(input_file.split('_')[-1],
                                   "%Y%m%dT%H%M%S.%f")
    msmt_start = np.datetime64(date_start.isoformat()).astype('datetime64[s]')
    msmt_stop = msmt_start + duration
    # print('msmt: ', msmt_start, msmt_stop)

    # correct msmt_start and msmt_stop to SRON clock
    t_itos, t_sron = read_date_stats()
    indx = np.argsort(np.abs(t_itos - t_sron))[0]
    t_diff = t_sron[indx] - t_itos[indx]
    # print('T_sron - T_itos:', t_diff)

    return msmt_start, msmt_stop, t_diff


def add_ogse_ref_diode(ref_db: Path, l1a_file: Path) -> None:
    """
    Select reference data taken during a measurement and add to a L1A product
    """
    # msmt_start and msmt_stop are generated with the ITOS clock
    msmt_start, msmt_stop, t_diff = clock_offset(l1a_file)

    # ref_time is generated with the SRON clock
    xds = open_dataset(ref_db, group='/gse_data/ReferenceDiode')
    ref_time = xds['time'].values - t_diff
    indx = np.where((ref_time >= msmt_start) & (ref_time <= msmt_stop))[0]
    if indx.size == 0:
        print('ReferenceDiode',
              ref_time.min(), msmt_start, msmt_stop, ref_time.max())
        print('[WARNING]: no reference-diode data found')
        return

    # update Level-1A product with OGSE/EGSE information
    xds = xds.isel(time=indx)
    xds.to_netcdf(l1a_file, mode='r+', format='NETCDF4',
                  group='/gse_data/ReferenceDiode')


def add_ogse_wav_mon(ref_db: Path, l1a_file: Path) -> None:
    """
    Select reference data taken during a measurement and add to a L1A product
    """
    # msmt_start and msmt_stop are generated with the ITOS clock
    msmt_start, msmt_stop, t_diff = clock_offset(l1a_file)

    # ref_time is generated with the SRON clock
    xds = open_dataset(ref_db, group='/gse_data/WaveMonitor')
    ref_time = xds['time'].values - t_diff
    mask = ((ref_time >= msmt_start) & (ref_time <= msmt_stop))
    if mask.sum() == 0:
        print('WaveMonitor',
              ref_time.min(), msmt_start, msmt_stop, ref_time.max())
        print('[WARNING]: no wavelength monitoring data found')
        return

    # update Level-1A product with OGSE/EGSE information
    xds = xds.isel(time=mask.nonzero()[0])
    xds.to_netcdf(l1a_file, mode='r+', format='NETCDF4',
                  group='/gse_data/WaveMonitor')


def __test():
    """Small function to test this module.
    """
    ogse_dir = Path('/data/richardh/SPEXone/ambient/polarimetric/calibration/'
                    'light_level/Logs/')

    file_list = ['POLARIMETRIC_LIGHT_LEVELS_ref_det_session174.csv']
    print('---------- SHOW DATASET ----------')
    print(read_ref_diode(ogse_dir, file_list, verbose=True))
    # print('---------- WRITE DATASET ----------')
    # create_ref_diode_db(ogse_dir, file_list)

    file_list = ['POLARIMETRIC_LIGHTLEVELS_3_Avantes_20210211T165045.csv']
    print('---------- SHOW DATASET ----------')
    print(read_wav_mon(ogse_dir, file_list, verbose=True))
    # print('---------- WRITE DATASET ----------')
    # create_wav_mon_db(ogse_dir, file_list)


# --------------------------------------------------
if __name__ == '__main__':
    __test()
