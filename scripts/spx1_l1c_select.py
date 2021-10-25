#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Create new SPEXone Level-1C product with selected data from original

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from datetime import datetime
from pathlib import Path

import h5py

from pyspex.lv1_io import L1Cio


# --------------------------------------------------
def main():
    """
    Main function of this module
    """
    parser = argparse.ArgumentParser(
        description=('Copy selected data from one SPEXone L1C product'
                     ' into a new SPEXone L1C product'))
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('--mps_id', nargs='*', type=int, default=None,
                        help='select on MPS-ID')
    # parser.add_argument('--time', nargs=2, default=None,
    #                    help='select on image time [start, end]')
    # parser.add_argument('--', default=None, help='')
    parser.add_argument('--out', default='.',
                        help=('name of directory to store the new Level-1C'
                              ' product, default: current working directory'))
    parser.add_argument('l1c_products', default=None, nargs='+',
                        help=('names of SPEXone/OCI Level-1C products'
                              'expect one SPEXone L1C product'
                              ' and one or more OCI L1C granuals'))
    args = parser.parse_args()
    if args.verbose:
        print(args)

    for name in args.l1c_products:
        l1c_product = Path(name)
        if not l1c_product.is_file():
            raise FileNotFoundError(f'File {name} does not exist')
        # ToDo: check SPEXone/OCI products
        # ToDo: store in variables spx1_product:str, oci_products:list
        # ToDo: implement check on data product

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(mode=0o755, parents=True)

    # ----- read data from orignal product -----
    # ToDo: read data into dict or xarray.Dataset per product
    # ToDo: combine OCI granuals into one dict or xarray.Dataset
    # pylint: disable=no-member, unsubscriptable-object
    with h5py.File(l1c_product[0]) as fid:
        # group BIN_ATTRIBUTES
        ref_date = None
        nadir_view_time = fid['/bin_attributes/nadir_view_time'][:]
        units = fid['/bin_attributes/nadir_view_time'].attrs['units']
        if len(units) > 14:
            ref_date = datetime.fromisoformat(units[14:])
        elif fid.attrs['time_coverage_start'] != b'Unknown':
            ref_date = datetime.fromisoformat(
                fid.attrs['time_coverage_start'].decode('ascii')[:-1])
            ref_date = ref_date.replace(hour=0, minute=0, second=0)
        view_time_offs = fid['/bin_attributes/view_time_offsets'][:]

        # group GEOLOCATION_DATA
        altitude = fid['/geolocation_data/altitude'][:]
        altitude_var = fid['/geolocation_data/altitude_variability'][:]
        latitude = fid['/geolocation_data/latitude'][:]
        longitude = fid['/geolocation_data/longitude'][:]
        sensor_azi = fid['/geolocation_data/sensor_azimuth'][:]
        sensor_zen = fid['/geolocation_data/sensor_zenith'][:]
        solar_azi = fid['/geolocation_data/solar_azimuth'][:]
        solar_zen = fid['/geolocation_data/solar_zenith'][:]

        # group OBSERVATION_DATA
        obs_per_view = fid['/observation_data/obs_per_view'][:]
        qc_bits = fid['/observation_data/QC_bitwise'][:]
        qc_val = fid['/observation_data/QC'][:]
        qc_pol_bits = fid['/observation_data/QC_polsample_bitwise'][:]
        qc_pol = fid['/observation_data/QC_polsample'][:]
        intens_val = fid['/observation_data/I'][:]
        intens_noise = fid['/observation_data/I_noise'][:]
        intens_pol = fid['/observation_data/I_polsample'][:]
        intens_pol_noise = fid['/observation_data/I_polsample_noise'][:]
        aolp_val = fid['/observation_data/AOLP'][:]
        aolp_noise = fid['/observation_data/AOLP_noise'][:]
        dolp_val = fid['/observation_data/DOLP'][:]
        dolp_noise = fid['/observation_data/DOLP_noise'][:]
        q_val = fid['/observation_data/Q_over_I'][:]
        q_noise = fid['/observation_data/Q_over_I_noise'][:]
        u_val = fid['/observation_data/U_over_I'][:]
        u_noise = fid['/observation_data/U_over_I_noise'][:]

        # group SENSOR_VIEWS_BANDS
        intens_bands = fid['/sensor_views_bands/intensity_bandpasses'][:]
        intens_wav = fid['/sensor_views_bands/intensity_wavelengths'][:]
        intens_f0 = fid['/sensor_views_bands/intensity_F0'][:]
        polar_bands = fid['/sensor_views_bands/polarization_bandpasses'][:]
        polar_wav = fid['/sensor_views_bands/polarization_wavelengths'][:]
        polar_f0 = fid['/sensor_views_bands/polarization_F0'][:]
        view_angles = fid['/sensor_views_bands/view_angles'][:]

        # global attributes
        global_attrs = {}
        check_attrs = ("history", "sun_earth_distance",
                       "terrain_data_source",
                       "spectral_response_function",
                       "systematic_uncertainty_model",
                       "bin_size_at_nadir")  # "nadir_bin"
        for key in sorted(check_attrs):
            if key in fid.attrs:
                if isinstance(fid.attrs[key], h5py.Empty):
                    global_attrs[key] = ""
                else:
                    global_attrs[key] = fid.attrs[key].decode('ascii')

    dims = {'bins_along_track':  latitude.shape[0],
            'bins_across_track':  latitude.shape[1],
            'intensity_bands_per_view':  intens_bands.shape[1],
            'polarization_bands_per_view':  polar_bands.shape[1]}

    # ----- perform data selection -----
    # ToDo: implement data selection

    # ----- now we can update the name of the output product -----
    # - because the production time has changed
    # - and when coverage time is changed
    if (out_dir / l1c_product.name).is_file() \
       and l1c_product.samefile(out_dir / l1c_product.name):
        raise OSError('Output will overwrite original product')

    # ----- write new output product with selected data -----
    print('dims: ', dims)
    with L1Cio(out_dir / l1c_product.name, ref_date=ref_date, dims=dims) as l1c:
        # group BIN_ATTRIBUTES
        l1c.set_dset('/BIN_ATTRIBUTES/nadir_view_time', nadir_view_time)
        l1c.set_dset('/BIN_ATTRIBUTES/view_time_offsets', view_time_offs)

        # group GEOLOCATION_DATA
        l1c.set_dset('/GEOLOCATION_DATA/altitude', altitude)
        l1c.set_dset('/GEOLOCATION_DATA/altitude_variability', altitude_var)
        l1c.set_dset('/GEOLOCATION_DATA/latitude', latitude)
        l1c.set_dset('/GEOLOCATION_DATA/longitude', longitude)
        l1c.set_dset('/GEOLOCATION_DATA/sensor_azimuth', sensor_azi)
        l1c.set_dset('/GEOLOCATION_DATA/sensor_zenith', sensor_zen)
        l1c.set_dset('/GEOLOCATION_DATA/solar_azimuth', solar_azi)
        l1c.set_dset('/GEOLOCATION_DATA/solar_zenith', solar_zen)

        # group OBSERVATION_DATA
        l1c.set_dset('/OBSERVATION_DATA/obs_per_view', obs_per_view)
        l1c.set_dset('/OBSERVATION_DATA/QC_bitwise', qc_bits.astype('u4'))
        l1c.set_dset('/OBSERVATION_DATA/QC', qc_val)
        l1c.set_dset('/OBSERVATION_DATA/QC_polsample_bitwise',
                     qc_pol_bits.astype('u4'))
        l1c.set_dset('/OBSERVATION_DATA/QC_polsample', qc_pol)
        l1c.set_dset('/OBSERVATION_DATA/I', intens_val)
        l1c.set_dset('/OBSERVATION_DATA/I_noise', intens_noise)
        l1c.set_dset('/OBSERVATION_DATA/I_polsample', intens_pol)
        l1c.set_dset('/OBSERVATION_DATA/I_polsample_noise', intens_pol_noise)
        l1c.set_dset('/OBSERVATION_DATA/AoLP', aolp_val)
        l1c.set_dset('/OBSERVATION_DATA/AoLP_noise', aolp_noise)
        l1c.set_dset('/OBSERVATION_DATA/DoLP', dolp_val)
        l1c.set_dset('/OBSERVATION_DATA/DoLP_noise', dolp_noise)
        l1c.set_dset('/OBSERVATION_DATA/Q_over_I', q_val)
        l1c.set_dset('/OBSERVATION_DATA/Q_over_I_noise', q_noise)
        l1c.set_dset('/OBSERVATION_DATA/U_over_I', u_val)
        l1c.set_dset('/OBSERVATION_DATA/U_over_I_noise', u_noise)

        # group SENSOR_VIEWS_BANDS
        l1c.set_dset('/SENSOR_VIEWS_BANDS/intensity_bandpasses', intens_bands)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/intensity_wavelengths', intens_wav)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/intensity_F0', intens_f0)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/polarization_bandpasses', polar_bands)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/polarization_wavelengths', polar_wav)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/polarization_F0', polar_f0)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/view_angles', view_angles)

        # write global attributes
        l1c.fill_global_attrs(inflight=True)
        for key, value in global_attrs.items():
            l1c.set_attr(key, value)


# --------------------------------------------------
if __name__ == '__main__':
    main()
