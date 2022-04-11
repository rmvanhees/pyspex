"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

...

Copyright (c) 2020-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime
from io import StringIO

import numpy as np
import xarray as xr


# ---------- CREATE OGSE databases ----------
def read_ref_diode(file_list: list, verbose=False) -> xr.Dataset:
    """
    Read reference diode data to numpy compound array
    (input: comma separated values)
    """
    data = None
    units = []
    unit_dict = {"Unix Time (s)": ("seconds", 'f8'),
                 "Excel Time (d)": ("days", 'f4'),
                 "Normalized Time (s)": ("seconds", 'f8'),
                 "amps": ("A", 'f4'), "scaled": ("1", 'f4'),
                 "last_zero": ("seconds", 'f8'), "lamp_on": ("flag", 'b'),
                 "voltage": ("V", 'f4'), "current": ("A", 'f4')}

    fields_to_skip = ('', 'averaging', 'lamp_on_counter', 'record_timestamp',
                      'scale', 'tempC')

    names = []
    formats = []
    usecols = ()
    for flname in file_list:
        all_valid_lines = ''
        with open(flname, 'r', encoding='ascii') as fid:
            while True:
                line = fid.readline().strip()
                if not line.startswith('Unix'):
                    continue

                if not names:
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
    res['time'] = xr.DataArray(
        data[time_key], coords={'time': data[time_key]},
        attrs={'longname': 'time', 'units': 'seconds since 1970-1-1 0:0:0'})
    for ii, key in enumerate(names):
        if key == time_key:
            continue

        res[key] = xr.DataArray(data[key], coords={'time': data[time_key]},
                                attrs={'longname': key, 'units': units[ii]})

    xds = xr.Dataset(res).sortby('time')
    xds.attrs = {'source': 'Calibrated diode inside integrated sphere',
                 'comment': 'Generated on SRON clean-room tablet'}
    return xds


# ---------------
def create_ref_diode_db(file_list: list, verbose=False):
    """
    Write reference diode and wavelength monitor data to HDF5 database
    """
    # read reference-diode data
    xds = read_ref_diode(file_list, verbose)

    # create new database for reference-diode data
    xds.to_netcdf('ogse_ref_diode.nc', mode='w', format='NETCDF4',
                  group='/gse_data/ReferenceDiode')


# ---------------
def read_wav_mon(file_list: list, verbose=False) -> xr.Dataset:
    """
    Read wavelength monitor data to numpy compound array
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
        with open(flname, 'r', encoding='ascii') as fid:
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
    res['time'] = xr.DataArray(
        data[time_key], coords={'time': data[time_key]},
        attrs={'longname': 'time', 'units': 'seconds since 1970-1-1 0:0:0'})
    res['wavelength'] = xr.DataArray(
        wavelength, coords={'wavelength': wavelength},
        attrs={'longname': 'wavelength grid', 'units': 'nm'})
    res['t_intg'] = xr.DataArray(
        t_intg.astype('i2'), dims=['time'],
        attrs={'longname': 'Integration time', 'units': 'nm'})
    res['n_avg'] = xr.DataArray(
        n_avg.astype('i2'), dims=['time'],
        attrs={'longname': 'Averaging number', 'units': '1'})
    res['spectrum'] = xr.DataArray(
        data['spectrum'], dims=['time', 'wavelength'],
        attrs={'longname': 'radiance spectrum', 'units': 'W/(m^2 sr nm'})

    xds = xr.Dataset(res).sortby('time')
    xds.attrs = {'source': 'Avantes fibre spectrometer',
                 'comment': 'Generated on SRON clean-room tablet'}
    return xds


# ---------------
def create_wav_mon_db(file_list: list, verbose=False):
    """
    Write Avantes fibre spectrometer data to HDF5 database
    """
    # read reference-diode data
    xds = read_wav_mon(file_list, verbose)

    # create new database for reference-diode data
    xds.to_netcdf('ogse_wave_mon.nc', mode='w', format='NETCDF4',
                  group='/gse_data/WaveMonitor')


# --------------------------------------------------
def test():
    """
    Test module
    """
    data_dir = '/data/richardh/SPEXone/ambient/polarimetric/calibration/'\
        'light_level/Logs/'

    file_list = [data_dir
                 + 'POLARIMETRIC_LIGHT_LEVELS_ref_det_session174.csv']
    print('---------- SHOW DATASET ----------')
    print(read_ref_diode(file_list, verbose=True))
    print('---------- WRITE DATASET ----------')
    create_ref_diode_db(file_list)

    file_list = [data_dir
                 + 'POLARIMETRIC_LIGHTLEVELS_3_Avantes_20210211T165045.csv']
    print('---------- SHOW DATASET ----------')
    print(read_wav_mon(file_list, verbose=True))
    print('---------- WRITE DATASET ----------')
    create_wav_mon_db(file_list)


if __name__ == '__main__':
    test()
