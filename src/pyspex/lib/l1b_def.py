"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation of the PACE SPEX Level-1B product

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from netCDF4 import Dataset

# - global parameters ------------------------------


# - local functions --------------------------------


# - main function ----------------------------------
def init_l1b(l1b_flname: str, dims: dict):
    """
    Create an empty PACE SPEX Level-1B product

    Parameters
    ----------
    l1b_flname : string
       Name of Level-1B product
    dims :   dictionary
       Provide length of the Level-1B dimensions
       Default values:
            number_of_images : None     # number of image frames
            spatial_samples : 200     # sum spatial bins of all viewports
    """
    # check function parameters
    if not isinstance(dims, dict):
        raise TypeError("dims should be a dictionary")

    # initialize dimensions
    number_of_images = None
    spatial_samples = 200
    n_views = 5
    n_bins_intens = 400
    n_bins_polar = 50

    if 'number_of_images' in dims:
        number_of_images = dims['number_of_images']
    if 'samples_per_image' in dims:
        spatial_samples = dims['spatial_samples']

    # create/overwrite netCDF4 product
    rootgrp = Dataset(l1b_flname, "w")

    # create global dimensions
    _ = rootgrp.createDimension('number_of_views', n_views)
    _ = rootgrp.createDimension('spatial_samples_per_image', spatial_samples)
    _ = rootgrp.createDimension('intensity_bands_per_view', n_bins_intens)
    _ = rootgrp.createDimension('polarization_bands_per_view', n_bins_polar)
    _ = rootgrp.createDimension('bins_along_track', number_of_images)

    # create groups and all variables with attributes
    sgrp = rootgrp.createGroup('SENSOR_VIEWS_BANDS')
    dset = sgrp.createVariable("viewport_index", "u1",
                               ("spatial_samples_per_image",))
    dset.long_name = "index of viewport"
    dset.valid_min = 0
    dset.valid_max = 4
    dset.comment = "Contains indices to each viewport for all spatial samples."
    dset = sgrp.createVariable('view_angles', 'f4', ('number_of_views',))
    dset.long_name = 'along track view zenith angles at sensor'
    dset.units = 'degrees'
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
    dset = sgrp.createVariable('intensity_f0', 'f4',
                               ('number_of_views', 'intensity_bands_per_view'))
    dset.long_name = 'Solar irradiance on intensity wavelength grid'
    dset.units = 'W.m-2'
    dset.comment = ('Spectral response function convolved mean solar flux'
                    ' at each intensity band and view.')
    dset = sgrp.createVariable('polarization_f0', 'f4',
                               ('number_of_views',
                                'polarization_bands_per_view'))
    dset.long_name = 'Solar irradiance on polarization wavelength grid'
    dset.units = 'W.m-2'
    dset.comment = ('Spectral response function convolved mean solar flux'
                    ' at each polarization band and view.')

    sgrp = rootgrp.createGroup('BIN_ATTRIBUTES')
    chunksizes = None if number_of_images is not None else (512,)
    dset = sgrp.createVariable('image_time', 'f8', ('bins_along_track',),
                               chunksizes=chunksizes)
    dset.long_name = 'image time (seconds of day)'
    dset.valid_min = 0
    dset.valid_max = 86400.999999
    dset.units = 'seconds'
    dset.reference = 'yyyy-mm-ddT00:00:00'

    sgrp = rootgrp.createGroup('GEOLOCATION_DATA')
    chunksizes = (None if number_of_images is not None
                  else (128, spatial_samples))
    dset = sgrp.createVariable('latitude', 'f4', ('bins_along_track',
                                                  'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.standard_name = 'latitude'
    dset.long_name = 'latitude'
    dset.valid_min = -90
    dset.valid_max = 90
    dset.units = 'degrees_north'
    dset = sgrp.createVariable('longitude', 'f4', ('bins_along_track',
                                                   'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.standard_name = 'longitude'
    dset.long_name = 'longitude'
    dset.valid_min = -180
    dset.valid_max = 180
    dset.units = 'degrees_east'
    dset = sgrp.createVariable('altitude', 'f4', ('bins_along_track',
                                                  'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.standard_name = 'altitude'
    dset.long_name = "height above mean sea level"
    dset.units = "meters"
    dset.positive = "up"
    dset.axis = "Z"
    dset = sgrp.createVariable('altitude_variability', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.long_name = 'altitude (stdev)'
    dset.units = 'meters'
    dset = sgrp.createVariable('sensor_azimuth', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.long_name = 'sensor azimuth angle'
    dset.units = 'degrees'
    dset = sgrp.createVariable('sensor_zenith', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.long_name = 'sensor zenith angle'
    dset.units = 'degrees'
    dset = sgrp.createVariable('solar_azimuth', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.long_name = 'solar azimuth angle'
    dset.units = 'degrees'
    dset = sgrp.createVariable('solar_zenith', 'f4',
                               ('bins_along_track',
                                'spatial_samples_per_image'),
                               chunksizes=chunksizes)
    dset.long_name = 'solar zenith angle'
    dset.units = 'degrees'

    sgrp = rootgrp.createGroup('OBSERVATION_DATA')
    chunksizes = None if number_of_images is not None else \
        (8, spatial_samples, n_bins_intens)
    dset = sgrp.createVariable('I', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'intensity_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'I Stokes component'
    dset.units = 'W.m-2.sr-1.um-1'
    dset = sgrp.createVariable('I_noise', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'intensity_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of I'
    dset.units = 'W.m-2.sr-1.um-1'

    chunksizes = None if number_of_images is not None else \
        (64, spatial_samples, n_bins_polar)
    dset = sgrp.createVariable('q', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'Q over I Stokes component'
    dset.units = '1'
    dset = sgrp.createVariable('q_noise', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of Q over I'
    dset.units = '1'
    dset = sgrp.createVariable('u', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'U over I Stokes component'
    dset.units = '1'
    dset = sgrp.createVariable('u_noise', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of U over I'
    dset.units = '1'
    dset = sgrp.createVariable('AoLP', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'angle of linear polarization'
    dset.units = 'degrees'
    dset = sgrp.createVariable('AoLP_noise', 'f4',
                               ('bins_along_track', 'spatial_samples_per_image',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of AoLP'
    dset.units = 'degrees'
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
    dset.long_name = 'noise of DoLP'
    dset.units = '1'

    return rootgrp


# --------------------------------------------------
if __name__ == '__main__':
    fid = init_l1b('PACE_SPEX.20230115T123456.L1B.2.5km.V01.nc', {})
    fid.fill_global_attrs(orbit=12345, bin_size='2.5km')
    fid.close()
