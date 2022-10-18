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
Defines the format of a SPEXone Level-1C product.
"""
__all__ = ['init_l1c']

import datetime

from netCDF4 import Dataset
import numpy as np


# - global parameters ------------------------------
ORBIT_DURATION = 5904  # seconds

# - local functions --------------------------------


# - main function ----------------------------------
# pylint: disable=too-many-statements
def init_l1c(l1c_flname: str, ref_date: datetime.date, dims: dict) -> None:
    """
    Create an empty PACE SPEX Level-1C product

    Parameters
    ----------
    l1c_flname : string
       Name of Level-1C product
    ref_date : datetime.date
       Date of the first detector image
    dims :   dictionary
       Provide length of the Level-1C dimensions
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
    n_bins_across = 20
    n_intens_bands = 50
    n_polar_bands = 50

    if 'bins_across_track' in dims:
        n_bins_across = dims['bins_across_track']
    if 'bins_along_track' in dims:
        n_bins_along = dims['bins_along_track']
    if 'intensity_bands_per_view' in dims:
        n_intens_bands = dims['intensity_bands_per_view']
    if 'polarization_bands_per_view' in dims:
        n_polar_bands = dims['polarization_bands_per_view']

    # create/overwrite netCDF4 product
    rootgrp = Dataset(l1c_flname, "w")

    # create global dimensions
    _ = rootgrp.createDimension('intensity_bands_per_view', n_intens_bands)
    _ = rootgrp.createDimension('pol_bands_per_view', n_polar_bands)
    _ = rootgrp.createDimension('bins_across_track', n_bins_across)
    _ = rootgrp.createDimension('bins_along_track', n_bins_along)
    _ = rootgrp.createDimension('number_of_views', n_views)

    # create groups and all variables with attributes
    sgrp = rootgrp.createGroup('BIN_ATTRIBUTES')
    chunksizes = None if n_bins_along is not None else (512,)
    dset = sgrp.createVariable('nadir_view_time', 'f8',
                               ('bins_along_track',), chunksizes=chunksizes)
    dset.long_name = 'Nadir view time'
    dset.description = "time when bin was viewed at nadir"
    dset.valid_min = 0
    dset.valid_max = 86400.999999
    if ref_date is None:
        dset.units = "second since midnight"
    else:
        dset.units = f"seconds since {ref_date.isoformat()} 00:00:00"
        dset.year = f"{ref_date.year}"
        dset.month = f"{ref_date.month}"
        dset.day = f"{ref_date.day}"
    dset.valid_min = 0
    dset.valid_max = 86400 + ORBIT_DURATION
    chunksizes = None if n_bins_along is not None else (512, n_bins_across)
    dset = sgrp.createVariable('view_time_offsets', 'f8',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views'), chunksizes=chunksizes)
    dset.long_name = "time offsets of views"
    dset.description = "offset of views wrt nadir view"
    dset.valid_min = -200
    dset.valid_max = 200
    dset.units = 'second'

    # -------------------------
    sgrp = rootgrp.createGroup('GEOLOCATION_DATA')
    dset = sgrp.createVariable('altitude', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.standard_name = 'altitude'
    dset.long_name = "altitude at bin locations"
    dset.units = "m"
    dset.positive = "up"
    dset.axis = "Z"
    dset = sgrp.createVariable('altitude_variability', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.long_name = 'RMS variability of altitude at bin locations'
    dset.units = 'm'
    dset = sgrp.createVariable('latitude', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.standard_name = 'latitude'
    dset.long_name = 'latitudes of bin locations'
    dset.valid_min = -90
    dset.valid_max = 90
    dset.units = 'degrees_north'
    dset = sgrp.createVariable('longitude', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.standard_name = 'longitude'
    dset.long_name = 'longitude of bin locations'
    dset.valid_min = -180
    dset.valid_max = 180
    dset.units = 'degrees_east'
    chunksizes = None if n_bins_along is not None \
        else (512, n_bins_across, n_views)
    dset = sgrp.createVariable('sensor_azimuth', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views'), chunksizes=chunksizes)
    dset.long_name = 'sensor azimuth angle at bin locations'
    dset.comment = 'clockwise from north'
    dset.units = 'degree'
    dset = sgrp.createVariable('sensor_zenith', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views'), chunksizes=chunksizes)
    dset.long_name = 'sensor zenith angle at bin locations'
    dset.units = 'degree'
    dset = sgrp.createVariable('solar_azimuth', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views'), chunksizes=chunksizes)
    dset.long_name = 'solar azimuth angle at bin locations'
    dset.comment = 'clockwise from north'
    dset.units = 'degree'
    dset = sgrp.createVariable('solar_zenith', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views'), chunksizes=chunksizes)
    dset.long_name = 'solar zenith angle at bin locations'
    dset.units = 'degree'

    # -------------------------
    sgrp = rootgrp.createGroup('OBSERVATION_DATA')
    chunksizes = None if n_bins_along is not None else \
        (32, n_bins_across, n_views)
    dset = sgrp.createVariable('obs_per_view', 'i2',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views'), chunksizes=chunksizes)
    dset.long_name = 'observations contributing to bin from each view'
    dset.valid_min = 0
    dset.units = '1'
    dset.comment = "Observations contributing to bin from each view"
    chunksizes = None if n_bins_along is not None \
        else (256, n_bins_across, n_views, n_polar_bands)
    dset = sgrp.createVariable('AoLP', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'angle of linear polarization'
    dset.units = 'degree'
    dset = sgrp.createVariable('AoLP_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of AOLP in bin'
    dset.units = '1'
    dset = sgrp.createVariable('DoLP', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'degree of linear polarization'
    dset.units = '1'
    dset = sgrp.createVariable('DoLP_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of DOLP in bin'
    dset.units = '1'
    chunksizes = None if n_bins_along is not None \
        else (32, n_bins_across, n_views, n_intens_bands)
    dset = sgrp.createVariable('I', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'intensity_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'I Stokes vector component'
    dset.units = 'W/(m^2.sr.um)'
    dset = sgrp.createVariable('I_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'intensity_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of I in bin'
    dset.units = 'W/(m^2.sr.um)'
    chunksizes = None if n_bins_along is not None \
        else (256, n_bins_across, n_views, n_polar_bands)
    dset = sgrp.createVariable('I_polsample', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = \
        'I Stokes vector component at polarization band spectal sampling'
    dset.units = 'W/(m^2.sr.um)'
    dset = sgrp.createVariable('I_polsample_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of I_polsample in bin'
    dset.units = 'W/(m^2.sr.um)'
    chunksizes = None if n_bins_along is not None \
        else (32, n_bins_across, n_views, n_intens_bands)
    dset = sgrp.createVariable('QC', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'intensity_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'quality indicator'
    dset.valid_min = 0
    dset.valid_max = 10
    dset.units = '1'
    dset = sgrp.createVariable('QC_bitwise', 'u4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'intensity_bands_per_view'),
                               fill_value=2 ** 32 - 1,
                               chunksizes=chunksizes)
    dset.long_name = 'quality flags'
    dset.valid_min = np.uint32(0)
    dset.valid_max = np.uint32(2 ** 31)
    dset.units = '1'
    chunksizes = None if n_bins_along is not None else \
        (256, n_bins_across, n_views, n_polar_bands)
    dset = sgrp.createVariable('QC_polsample', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'quality indicator at polarization band spectal sampling'
    dset.valid_min = 0
    dset.valid_max = 10
    dset.units = '1'
    dset = sgrp.createVariable('QC_polsample_bitwise', 'u4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               fill_value=2 ** 32 - 1,
                               chunksizes=chunksizes)
    dset.long_name = 'quality flags at polarization band spectal sampling'
    dset.valid_min = np.uint32(0)
    dset.valid_max = np.uint32(2 ** 31)
    dset.units = '1'
    dset = sgrp.createVariable('Q_over_I', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'Q_over_I (little q) Stokes vector component'
    dset.units = '1'
    dset = sgrp.createVariable('Q_over_I_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of Q_over_I in bin'
    dset.units = '1'
    dset = sgrp.createVariable('U_over_I', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'U_over_I (little u) Stokes vector component'
    dset.units = '1'
    dset = sgrp.createVariable('U_over_I_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'pol_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'random noise of U_over_I in bin'
    dset.units = '1'

    # -------------------------
    sgrp = rootgrp.createGroup('SENSOR_VIEWS_BANDS')
    dset = sgrp.createVariable('intensity_bandpasses', 'f4',
                               ('number_of_views', 'intensity_bands_per_view'))
    dset.long_name = 'intensity field bandpasses at each view'
    dset.units = 'nm'
    dset = sgrp.createVariable('intensity_F0', 'f4',
                               ('number_of_views', 'intensity_bands_per_view'))
    dset.long_name = 'intensity band solar irradiance'
    dset.units = 'W/m^2'
    dset = sgrp.createVariable('intensity_wavelengths', 'f4',
                               ('number_of_views', 'intensity_bands_per_view'))
    dset.long_name = 'intensity field center wavelengths at each view'
    dset.units = 'nm'
    dset = sgrp.createVariable('polarization_bandpasses', 'f4',
                               ('number_of_views', 'pol_bands_per_view'))
    dset.long_name = 'polarization field bandpasses at each view'
    dset.units = 'nm'
    dset = sgrp.createVariable('polarization_F0', 'f4',
                               ('number_of_views', 'pol_bands_per_view'))
    dset.long_name = 'polarization band solar irradiance'
    dset.units = 'W/m^2'
    dset.comment = ('Spectral response function convolved mean solar flux'
                    ' at each polarization band and view.')
    dset = sgrp.createVariable('polarization_wavelengths', 'f4',
                               ('number_of_views', 'pol_bands_per_view'))
    dset.long_name = 'polarization field wavelengths at each view'
    dset.units = 'nm'
    dset.comment = ('Spectral response function convolved mean solar flux'
                    ' at each intensity band and view.')
    dset = sgrp.createVariable('view_angles', 'f4', ('number_of_views',))
    dset.long_name = 'along-track view angles for sensor'
    dset.units = 'degree'
    dset.comment = ('view_angles is defined at the sensor, as it provides'
                    ' a swath independent value at TOA.')

    return rootgrp
