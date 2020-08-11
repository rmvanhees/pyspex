"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation of the PACE SPEX Level-1C product

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from netCDF4 import Dataset

# - global parameters ------------------------------


# - local functions --------------------------------


# - main function ----------------------------------
def init_l1c(l1c_flname: str, dims: dict):
    """
    Create an empty PACE SPEX Level-1C product

    Parameters
    ----------
    l1c_flname : string
       Name of Level-1C product
    dims :   dictionary
       Provide length of the Level-1B dimensions
       Default values:
            number_of_images : None     # number of image frames
    """
    # check function parameters
    if not isinstance(dims, dict):
        raise TypeError("dims should be a dictionary")

    # initialize dimensions
    number_of_images = None
    n_views = 5
    n_bins_intens = 400
    n_bins_polar = 50
    n_bins_across = 40

    if 'number_of_images' in dims:
        number_of_images = dims['number_of_images']

    # create/overwrite netCDF4 product
    rootgrp = Dataset(l1c_flname, "w")

    # create global dimensions
    _ = rootgrp.createDimension('number_of_views', n_views)
    _ = rootgrp.createDimension('intensity_bands_per_view', n_bins_intens)
    _ = rootgrp.createDimension('polarization_bands_per_view', n_bins_polar)
    _ = rootgrp.createDimension('bins_across_track', n_bins_across)
    _ = rootgrp.createDimension('bins_along_track', number_of_images)

    # create groups and all variables with attributes
    sgrp = rootgrp.createGroup('SENSOR_VIEWS_BANDS')
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
    dset = sgrp.createVariable('nadir_view_time', 'f8', ('bins_along_track',),
                               chunksizes=chunksizes)
    dset.long_name = 'nadir time (seconds of day)'
    dset.valid_min = 0
    dset.valid_max = 86400.999999
    dset.units = 'seconds'
    dset.reference = 'yyyy-mm-ddT00:00:00'
    chunksizes = None if number_of_images is not None else (512, n_bins_across)
    dset = sgrp.createVariable('view_time_offsets', 'f8',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.long_name = 'offset to nadir view time'
    dset.units = 'seconds'

    sgrp = rootgrp.createGroup('GEOLOCATION_DATA')
    dset = sgrp.createVariable('latitude', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.standard_name = 'latitude'
    dset.long_name = 'latitude'
    dset.valid_min = -90
    dset.valid_max = 90
    dset.units = 'degrees_north'
    dset = sgrp.createVariable('longitude', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.standard_name = 'longitude'
    dset.long_name = 'longitude'
    dset.valid_min = -180
    dset.valid_max = 180
    dset.units = 'degrees_east'
    dset = sgrp.createVariable('altitude', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.standard_name = 'altitude'
    dset.long_name = "height above mean sea level"
    dset.units = "meters"
    dset.positive = "up"
    dset.axis = "Z"
    dset = sgrp.createVariable('altitude_variability', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.long_name = 'altitude (stdev)'
    dset.units = 'meters'
    dset = sgrp.createVariable('solar_azimuth', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.long_name = 'solar azimuth angle'
    dset.units = 'degrees'
    dset = sgrp.createVariable('solar_zenith', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.long_name = 'solar zenith angle'
    dset.units = 'degrees'

    chunksizes = None if number_of_images is not None else \
        (512, n_bins_across, n_views)
    dset = sgrp.createVariable('sensor_azimuth', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views'),
                               chunksizes=chunksizes)
    dset.long_name = 'sensor azimuth angle'
    dset.units = 'degrees'
    dset = sgrp.createVariable('sensor_zenith', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views'),
                               chunksizes=chunksizes)
    dset.long_name = 'sensor zenith angle'
    dset.units = 'degrees'

    chunksizes = None if number_of_images is not None else (512, 40, 5)
    sgrp = rootgrp.createGroup('OBSERVATION_DATA')
    dset = sgrp.createVariable('obs_per_view', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views'),
                               chunksizes=chunksizes)
    dset.long_name = 'observations per view'
    dset.units = '1'
    dset.comment = "Observations contributing to bin from each view"

    chunksizes = None if number_of_images is not None else \
        (32, n_bins_across, n_views, n_bins_intens)
    dset = sgrp.createVariable('I', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'intensity_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'I Stokes component'
    dset.units = 'W.m-2.sr-1.um-1'
    dset = sgrp.createVariable('I_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views', 'intensity_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of I'
    dset.units = 'W.m-2.sr-1.um-1'

    chunksizes = None if number_of_images is not None else \
        (256, n_bins_across, n_views, n_bins_polar)
    dset = sgrp.createVariable('I_polsample', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'I at polarimeter sampling'
    dset.units = 'W.m-2.sr-1.um-1'
    dset = sgrp.createVariable('I_polsample_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of I at polarimeter sampling'
    dset.units = 'W.m-2.sr-1.um-1'
    dset = sgrp.createVariable('Q', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'Q Stokes component'
    dset.units = 'W.m-2.sr-1.um-1'
    dset = sgrp.createVariable('Q_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of Q'
    dset.units = 'W.m-2.sr-1.um-1'
    dset = sgrp.createVariable('U', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'U Stokes component'
    dset.units = 'W.m-2.sr-1.um-1'
    dset = sgrp.createVariable('U_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of U'
    dset.units = 'W.m-2.sr-1.um-1'
    dset = sgrp.createVariable('q', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'Q over I Stokes component'
    dset.units = '1'
    dset = sgrp.createVariable('q_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of q'
    dset.units = '1'
    dset = sgrp.createVariable('u', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'U over I Stokes component'
    dset.units = '1'
    dset = sgrp.createVariable('u_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of u'
    dset.units = '1'
    dset = sgrp.createVariable('DoLP', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'degree of linear polarization'
    dset.units = '1'
    dset = sgrp.createVariable('DoLP_noise', 'f4',
                               ('bins_along_track', 'bins_across_track',
                                'number_of_views',
                                'polarization_bands_per_view'),
                               chunksizes=chunksizes)
    dset.long_name = 'noise of DoLP'
    dset.units = '1'

    return rootgrp


# --------------------------------------------------
if __name__ == '__main__':
    fid = init_l1c('PACE_SPEX.20230115T123456.L1C.5km.V01.nc', {})
    fid.fill_global_attrs(orbit=12345, bin_size='5km')
    fid.close()
