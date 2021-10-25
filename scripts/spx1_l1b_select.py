#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Create new SPEXone Level-1B product with selected data from original

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from datetime import datetime
from pathlib import Path

import h5py

from pyspex.lv1_io import L1Bio


# --------------------------------------------------
def main():
    """
    Main function of this module
    """
    parser = argparse.ArgumentParser(
        description=('Copy selected data from one SPEXone L1B product'
                     ' into a new SPEXone L1B product'))
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('--mps_id', nargs='*', type=int, default=None,
                        help='select on MPS-ID')
    # parser.add_argument('--time', nargs=2, default=None,
    #                    help='select on image time [start, end]')
    # parser.add_argument('--', default=None, help='')
    parser.add_argument('--out', default='.',
                        help=('name of directory to store the new Level-1B'
                              ' product, default: current working directory'))
    parser.add_argument('l1b_product', default=None,
                        help='name of SPEXone Level-1B product')
    args = parser.parse_args()
    if args.verbose:
        print(args)

    l1b_product = Path(args.l1b_product)
    if not l1b_product.is_file():
        raise FileNotFoundError(f'File {args.l1b_product} does not exist')
    # ToDo: check if SPEXone Level-1B product
    # ToDo: implement check on data product

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(mode=0o755, parents=True)

    # ----- read data from orignal product -----
    # pylint: disable=no-member, unsubscriptable-object
    with h5py.File(l1b_product) as fid:
        # group BIN_ATTRIBUTES
        ref_date = None
        image_time = fid['/BIN_ATTRIBUTES/image_time'][:]
        units = fid['/BIN_ATTRIBUTES/image_time'].attrs['units']
        if len(units) > 14:
            ref_date = datetime.fromisoformat(units[14:])
        elif fid.attrs['time_coverage_start'] != b'Unknown':
            ref_date = datetime.fromisoformat(
                fid.attrs['time_coverage_start'].decode('ascii'))
            ref_date = ref_date.replace(hour=0, minute=0, second=0)

        # group GEOLOCATION_DATA
        altitude = fid['/GEOLOCATION_DATA/altitude'][:]
        latitude = fid['/GEOLOCATION_DATA/latitude'][:]
        longitude = fid['/GEOLOCATION_DATA/longitude'][:]
        sensor_azi = fid['/GEOLOCATION_DATA/sensor_azimuth'][:]
        sensor_zen = fid['/GEOLOCATION_DATA/sensor_zenith'][:]
        solar_azi = fid['/GEOLOCATION_DATA/solar_azimuth'][:]
        solar_zen = fid['/GEOLOCATION_DATA/solar_zenith'][:]

        # group OBSERVATION_DATA
        intens_val = fid['/OBSERVATION_DATA/I'][:]
        intens_noise = fid['/OBSERVATION_DATA/I_noise'][:]
        aolp_val = fid['/OBSERVATION_DATA/AoLP'][:]
        aolp_noise = fid['/OBSERVATION_DATA/AoLP_noise'][:]
        dolp_val = fid['/OBSERVATION_DATA/DoLP'][:]
        dolp_noise = fid['/OBSERVATION_DATA/DoLP_noise'][:]
        q_val = fid['/OBSERVATION_DATA/q'][:]
        q_noise = fid['/OBSERVATION_DATA/q_noise'][:]
        u_val = fid['/OBSERVATION_DATA/u'][:]
        u_noise = fid['/OBSERVATION_DATA/u_noise'][:]

        # group SENSOR_VIEWS_BANDS
        group = 'SENSOR_VIEWS_BANDS' if 'SENSOR_VIEWS_BANDS' in fid \
            else 'SENSOR_VIEW_BANDS'
        view_indx = fid[f'/{group}/viewport_index'][:]
        intens_bands = fid[f'/{group}/intensity_bandpasses'][:]
        intens_wav = fid[f'/{group}/intensity_wavelengths'][:]
        intens_f0 = fid[f'/{group}/intensity_f0'][:]
        polar_bands = fid[f'/{group}/polarization_bandpasses'][:]
        polar_wav = fid[f'/{group}/polarization_wavelengths'][:]
        polar_f0 = fid[f'/{group}/polarization_f0'][:]
        view_angles = fid[f'/{group}/view_angles'][:]

        # global attributes
        global_attrs = {}
        check_attrs = ("history", "sun_earth_distance",
                       "terrain_data_source",
                       "spectral_response_function",
                       "systematic_uncertainty_model")
        for key in sorted(check_attrs):
            if key in fid.attrs:
                if isinstance(fid.attrs[key], h5py.Empty):
                    global_attrs[key] = ""
                else:
                    global_attrs[key] = fid.attrs[key].decode('ascii')

    dims = {'bins_along_track':  latitude.shape[0],
            'spatial_samples_per_image':  latitude.shape[1],
            'intensity_bands_per_view':  intens_bands.shape[1],
            'polarization_bands_per_view':  polar_bands.shape[1]}

    # ----- perform data selection -----
    # ToDo: implement data selection

    # ----- now we can update the name of the output product -----
    # - because the production time has changed
    # - and when coverage time is changed
    if (out_dir / l1b_product.name).is_file() \
       and l1b_product.samefile(out_dir / l1b_product.name):
        raise OSError('Output will overwrite original product')

    # ----- write new output product with selected data -----
    print('dims: ', dims)
    print('ref_date: ', ref_date)
    with L1Bio(out_dir / l1b_product.name, ref_date=ref_date, dims=dims) as l1b:
        # group BIN_ATTRIBUTES
        l1b.set_dset('/BIN_ATTRIBUTES/image_time', image_time)

        # group GEOLOCATION_DATA
        l1b.set_dset('/GEOLOCATION_DATA/altitude', altitude)
        l1b.set_dset('/GEOLOCATION_DATA/latitude', latitude)
        l1b.set_dset('/GEOLOCATION_DATA/longitude', longitude)
        l1b.set_dset('/GEOLOCATION_DATA/sensor_azimuth', sensor_azi)
        l1b.set_dset('/GEOLOCATION_DATA/sensor_zenith', sensor_zen)
        l1b.set_dset('/GEOLOCATION_DATA/solar_azimuth', solar_azi)
        l1b.set_dset('/GEOLOCATION_DATA/solar_zenith', solar_zen)

        # group OBSERVATION_DATA
        l1b.set_dset('/OBSERVATION_DATA/I', intens_val)
        l1b.set_dset('/OBSERVATION_DATA/I_noise', intens_noise)
        l1b.set_dset('/OBSERVATION_DATA/AoLP', aolp_val)
        l1b.set_dset('/OBSERVATION_DATA/AoLP_noise', aolp_noise)
        l1b.set_dset('/OBSERVATION_DATA/DoLP', dolp_val)
        l1b.set_dset('/OBSERVATION_DATA/DoLP_noise', dolp_noise)
        l1b.set_dset('/OBSERVATION_DATA/Q_over_I', q_val)
        l1b.set_dset('/OBSERVATION_DATA/Q_over_I_noise', q_noise)
        l1b.set_dset('/OBSERVATION_DATA/U_over_I', u_val)
        l1b.set_dset('/OBSERVATION_DATA/U_over_I_noise', u_noise)

        # group SENSOR_VIEWS_BANDS
        l1b.set_dset('/SENSOR_VIEWS_BANDS/viewport_index', view_indx)
        l1b.set_dset('/SENSOR_VIEWS_BANDS/intensity_bandpasses', intens_bands)
        l1b.set_dset('/SENSOR_VIEWS_BANDS/intensity_wavelengths', intens_wav)
        l1b.set_dset('/SENSOR_VIEWS_BANDS/intensity_F0', intens_f0)
        l1b.set_dset('/SENSOR_VIEWS_BANDS/polarization_bandpasses', polar_bands)
        l1b.set_dset('/SENSOR_VIEWS_BANDS/polarization_wavelengths', polar_wav)
        l1b.set_dset('/SENSOR_VIEWS_BANDS/polarization_F0', polar_f0)
        l1b.set_dset('/SENSOR_VIEWS_BANDS/view_angles', view_angles)

        # write global attributes
        l1b.fill_global_attrs(inflight=True)
        for key, value in global_attrs.items():
            l1b.set_attr(key, value)


# --------------------------------------------------
if __name__ == '__main__':
    main()
