#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Create a valid PACE SPEXone Level-1B product to estimate the size of a
science orbit product

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime, timezone

import numpy as np

from pyspex.lv1_io import L1Bio

# - global parameters ------------------------------


# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """
    Generate a SPEXone inflight Level-1B product
    """
    start_time = datetime(2023, 1, 15, 12, 34, 56, tzinfo=timezone.utc)
    midnight = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    duration = 2880  # seconds

    # define product: name, level, version, start of time-coverage
    prod_level = 'L1B'
    bin_size = '2.5km'
    prod_version = 'V1.0'
    l1b_flname = 'PACE_SPEX.{:s}.{:s}.{:s}.{:s}.nc'.format(
        start_time.strftime('%Y%m%dT%H%M%S'),
        prod_level, bin_size, prod_version)

    with L1Bio(l1b_flname, dims={}) as l1b:
        secnds = (1 + np.arange(3 * duration, dtype=float)) / 3\
            + (start_time - midnight).total_seconds()

        # get dimension sizes
        n_view = l1b.get_dim('number_of_views')
        n_intens = l1b.get_dim('intensity_bands_per_view')
        n_polar = l1b.get_dim('polarization_bands_per_view')
        n_spatial = l1b.get_dim('spatial_samples_per_image')

        # group SENSOR_VIEWS_BANDS
        l1b.set_dset('/SENSOR_VIEWS_BANDS/view_angles',
                     np.array([-58., -30., 0., 30., 58.]))
        l1b.set_dset('/SENSOR_VIEWS_BANDS/viewport_index',
                     np.arange(n_spatial, dtype='u1') // 40)
        l1b.set_dset('/SENSOR_VIEWS_BANDS/intensity_wavelengths',
                     np.arange(n_view * n_intens).reshape(n_view, n_intens))
        l1b.set_dset('/SENSOR_VIEWS_BANDS/intensity_bandpasses',
                     np.arange(n_view * n_intens).reshape(n_view, n_intens))
        l1b.set_dset('/SENSOR_VIEWS_BANDS/polarization_wavelengths',
                     np.arange(n_view * n_polar).reshape(n_view, n_polar))
        l1b.set_dset('/SENSOR_VIEWS_BANDS/polarization_bandpasses',
                     np.arange(n_view * n_polar).reshape(n_view, n_polar))
        l1b.set_dset('/SENSOR_VIEWS_BANDS/intensity_f0',
                     np.arange(n_view * n_intens).reshape(n_view, n_intens))
        l1b.set_dset('/SENSOR_VIEWS_BANDS/polarization_f0',
                     np.arange(n_view * n_polar).reshape(n_view, n_polar))

        for ii in range(3 * duration):
            # group BIN_ATTRIBUTES
            l1b.set_dset('/BIN_ATTRIBUTES/image_time', [secnds[ii]])

            # group GEOLOCATION_DATA
            data = np.zeros((1, n_spatial), dtype=float) \
                + ii / duration
            l1b.set_dset('/GEOLOCATION_DATA/latitude', data)
            l1b.set_dset('/GEOLOCATION_DATA/longitude', data)
            l1b.set_dset('/GEOLOCATION_DATA/altitude', data)
            l1b.set_dset('/GEOLOCATION_DATA/altitude_variability', data)
            l1b.set_dset('/GEOLOCATION_DATA/solar_azimuth', data)
            l1b.set_dset('/GEOLOCATION_DATA/solar_zenith', data)
            l1b.set_dset('/GEOLOCATION_DATA/sensor_azimuth', data)
            l1b.set_dset('/GEOLOCATION_DATA/sensor_zenith', data)

            # group OBSERVATION_DATA
            data = np.ones((1, n_spatial, n_intens), dtype=float) \
                + ii / duration
            l1b.set_dset('/OBSERVATION_DATA/I', data)
            l1b.set_dset('/OBSERVATION_DATA/I_noise', data)
            data = np.ones((1, n_spatial, n_polar), dtype=float) \
                + ii / duration
            l1b.set_dset('/OBSERVATION_DATA/q', data)
            l1b.set_dset('/OBSERVATION_DATA/q_noise', data)
            l1b.set_dset('/OBSERVATION_DATA/u', data)
            l1b.set_dset('/OBSERVATION_DATA/u_noise', data)
            l1b.set_dset('/OBSERVATION_DATA/AoLP', data)
            l1b.set_dset('/OBSERVATION_DATA/AoLP_noise', data)
            l1b.set_dset('/OBSERVATION_DATA/DoLP', data)
            l1b.set_dset('/OBSERVATION_DATA/DoLP_noise', data)

        # Global attributes
        l1b.fill_global_attrs(orbit=12345, bin_size=bin_size)


# --------------------------------------------------
if __name__ == '__main__':
    main()
