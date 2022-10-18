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
Defines the format of a SPEXone Level-1B product.
"""
__all__ = ['init_l1b']

import datetime

from netCDF4 import Dataset

# - global parameters ------------------------------
ORBIT_DURATION = 5904  # seconds

# - local functions --------------------------------


# - main function ----------------------------------
# pylint: disable=too-many-statements
def init_l1b(l1b_flname: str, ref_date: datetime.date, dims: dict) -> None:
    """
    Create an empty PACE SPEX Level-1B product

    Parameters
    ----------
    l1b_flname : string
       Name of Level-1B product
    ref_date : datetime.date
       Date of the first detector image
    dims :   dictionary
       Provide length of the Level-1B dimensions
       Default values::

          bins_along_track: 400
          spatial_samples_per_image: 200
          intensity_bands_per_view: 50
          polarization_bands_per_view: 50

    """
    # check function parameters
    if not isinstance(dims, dict):
        raise TypeError("dims should be a dictionary")

    # initialize dimensions
    n_views = 5
    n_bins_along = 400
    n_spatial_samples = 200
    n_intens_bands = 50
    n_polar_bands = 50

    if 'bins_along_track' in dims:
        n_bins_along = dims['bins_along_track']
    if 'spatial_samples_per_image' in dims:
        n_spatial_samples = dims['spatial_samples_per_image']
    if 'intensity_bands_per_view' in dims:
        n_intens_bands = dims['intensity_bands_per_view']
    if 'polarization_bands_per_view' in dims:
        n_polar_bands = dims['polarization_bands_per_view']

    # create/overwrite netCDF4 product
    rootgrp = Dataset(l1b_flname, "w")

    # create global dimensions
    _ = rootgrp.createDimension('number_of_views', n_views)
    _ = rootgrp.createDimension('bins_along_track', n_bins_along)
    _ = rootgrp.createDimension('spatial_samples_per_image', n_spatial_samples)
    _ = rootgrp.createDimension('intensity_bands_per_view', n_intens_bands)
    _ = rootgrp.createDimension('polarization_bands_per_view', n_polar_bands)

    # create groups and all variables with attributes
    sgrp = rootgrp.createGroup('BIN_ATTRIBUTES')
    chunksizes = None if n_bins_along is not None else (512,)
    dset = sgrp.createVariable('image_time', 'f8', ('bins_along_track',),
                               chunksizes=chunksizes, fill_value=-32767)
    dset.long_name = "Image time"
    dset.description = "Integration start time in seconds of day"
    if ref_date is None:
        dset.units = "second since midnight"
    else:
        dset.units = f"seconds since {ref_date.isoformat()} 00:00:00"
        dset.year = f"{ref_date.year}"
        dset.month = f"{ref_date.month}"
        dset.day = f"{ref_date.day}"
    dset.valid_min = 0
    dset.valid_max = 86400 + ORBIT_DURATION

    # -------------------------
    sgrp = rootgrp.createGroup('GEOLOCATION_DATA')
    chunksizes = (None if n_bins_along is not None
                  else (128, n_spatial_samples))
    dset = sgrp.createVariable('altitude', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.standard_name = 'altitude'
    dset.long_name = "height above mean sea level"
    dset.units = "m"
    dset.positive = "up"
    dset.axis = "Z"
    dset = sgrp.createVariable('latitude', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.standard_name = 'latitude'
    dset.long_name = 'latitude'
    dset.valid_min = -90
    dset.valid_max = 90
    dset.units = 'degrees_north'
    dset = sgrp.createVariable('longitude', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.standard_name = 'longitude'
    dset.long_name = 'longitude'
    dset.valid_min = -180
    dset.valid_max = 180
    dset.units = 'degrees_east'
    dset = sgrp.createVariable('sensor_azimuth', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.long_name = 'sensor azimuth angle'
    dset.units = 'degree'
    dset = sgrp.createVariable('sensor_zenith', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.long_name = 'sensor zenith angle'
    dset.units = 'degree'
    dset = sgrp.createVariable('solar_azimuth', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.long_name = 'solar azimuth angle'
    dset.units = 'degree'
    dset = sgrp.createVariable('solar_zenith', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.long_name = 'solar zenith angle'
    dset.units = 'degree'

    # -------------------------
    sgrp = rootgrp.createGroup('OBSERVATION_DATA')
    chunksizes = None if n_bins_along is not None else \
        (8, n_spatial_samples, n_intens_bands)
    dset = sgrp.createVariable('I', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'intensity_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'I Stokes vector component'
    dset.units = 'W/(m^2.sr.um)'
    dset = sgrp.createVariable('I_noise', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'intensity_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of I'
    dset.units = 'W/(m^2.sr.um)'

    chunksizes = None if n_bins_along is not None else \
        (64, n_spatial_samples, n_polar_bands)
    dset = sgrp.createVariable('Q_over_I', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'Q over I (little q) Stokes vector component'
    dset.units = '1'
    dset = sgrp.createVariable('Q_over_I_noise', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of Q_over_I'
    dset.units = '1'
    dset = sgrp.createVariable('U_over_I', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'U over I (little u) Stokes vector component'
    dset.units = '1'
    dset = sgrp.createVariable('U_over_I_noise', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of U_over_I'
    dset.units = '1'
    dset = sgrp.createVariable('AoLP', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'angle of linear polarization'
    dset.units = 'degree'
    dset = sgrp.createVariable('AoLP_noise', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of AoLP'
    dset.units = 'degree'
    dset = sgrp.createVariable('DoLP', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'degree of linear polarization'
    dset.units = '1'
    dset = sgrp.createVariable('DoLP_noise', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of DoLP'
    dset.units = '1'

    # -------------------------
    sgrp = rootgrp.createGroup('SENSOR_VIEWS_BANDS')
    dset = sgrp.createVariable("viewport_index", "u1",
                               ("spatial_samples_per_image",))
    dset.long_name = "index of viewport"
    dset.valid_min = 0
    dset.valid_max = 4
    dset.comment = "Contains indices to each viewport for all spatial samples."
    dset = sgrp.createVariable('view_angles', 'f4', ('number_of_views',))
    dset.long_name = 'along track view zenith angles at sensor'
    dset.units = 'degree'
    dset.comment = ('view_angles is defined at the sensor, as it provides'
                    ' a swath independent value at TOA.')
    dset = sgrp.createVariable('intensity_wavelengths', 'f4',
                               ('number_of_views', 'intensity_bands_per_view'))
    dset.long_name = 'wavelength at center of intensity bands'
    dset.units = 'nm'
    dset = sgrp.createVariable('intensity_bandpasses', 'f4',
                               ('number_of_views', 'intensity_bands_per_view'))
    dset.long_name = 'FWHM of intensity bands'
    dset.units = 'nm'
    dset = sgrp.createVariable('intensity_F0', 'f4',
                               ('number_of_views', 'intensity_bands_per_view'))
    dset.long_name = 'Solar irradiance on intensity wavelength grid'
    dset.units = 'W/m^2'
    dset.comment = ('Spectral response function convolved mean solar flux'
                    ' at each intensity band and view.')
    dset = sgrp.createVariable('polarization_wavelengths', 'f4',
                               ('number_of_views',
                                'polarization_bands_per_view'))
    dset.long_name = 'wavelength at center of polarization bands'
    dset.units = 'nm'
    dset = sgrp.createVariable('polarization_bandpasses', 'f4',
                               ('number_of_views',
                                'polarization_bands_per_view'))
    dset.long_name = 'FWHM of polarization bands'
    dset.units = 'nm'
    dset = sgrp.createVariable('polarization_F0', 'f4',
                               ('number_of_views',
                                'polarization_bands_per_view'))
    dset.long_name = 'Solar irradiance on polarization wavelength grid'
    dset.units = 'W/m^2'
    dset.comment = ('Spectral response function convolved mean solar flux'
                    ' at each polarization band and view.')

    return rootgrp
