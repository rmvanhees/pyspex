"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation of the PACE SPEX Level-1C product

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime
from pathlib import Path

from netCDF4 import Dataset

# - global parameters ------------------------------


# - local functions --------------------------------
def global_attributes() -> dict:
    """
    Return dictionary with required global attributes for the L1C format
    """
    return {
        'title': 'PACE SPEX Level-1C data',
        'instrument': 'SPEX',
        'processing_version': 'V1.0',
        'conventions': 'CF-1.6',
        'institution': ('NASA Goddard Space Flight Center,'
                        ' Ocean Biology Processing Group'),
        'license': ('http://science.nasa.gov/earth-science'
                    '/earth-science-data/data-information-policy/'),
        'naming_authority': 'gov.nasa.gsfc.sci.oceancolor',
        'keyword_vocabulary': ('NASA Global Change Master Directory (GCMD)'
                               ' Science Keywords'),
        'stdname_vocabulary': ('NetCDF Climate and Forecast (CF)'
                               ' Metadata Convention'),
        'creator_name': 'NASA/GSFC',
        'creator_email': 'data@oceancolor.gsfc.nasa.gov',
        'creator_url': 'http://oceancolor.gsfc.nasa.gov',
        'project': 'PACE Project',
        'publisher_name': 'NASA/GSFC',
        'publisher_email': 'data@oceancolor.gsfc.nasa.gov',
        'publisher_url': 'http://oceancolor.gsfc.nasa.gov',
        'processing_level': 'L1C',
        'cdm_data_type': 'swath',
        'history': '',
        'cdl_version_date': '2020-02-20',
        'startdirection': 'Ascending',
        'enddirection': 'Ascending',
        'time_coverage_start': 'yyyy-mm-ddThh:mm:ss.sssZ',
        'time_coverage_end': 'yyyy-mm-ddThh:mm:ss.sssZ',
        'date_created': datetime.utcnow().isoformat(timespec='milliseconds'),
        'sun_earth_distance': 1.0,
        'terrain_data_source': '<Source of terrain data used for aggregation>',
        'spectral_response_function': ('<Points to documentation'
                                       ' containing this information>'),
        'systematic_uncertainty_model': ('<Models (equations) for systematic'
                                         ' uncertainty for I, DoLP, Q, U>'),
        'nadir_bin': 200,
        'bin_size_at_nadir': '5 km'
    }


# - main function ----------------------------------
def init_l1c(l1c_flname: str, orbit_number=-1, number_of_images=None):
    """
    Create an empty PACE SPEX Level-1C product

    Parameters
    ----------
    l1c_flname : string
       Name of Level-1C product
    orbit_number: int
       Orbit revolution counter, default=-1
    number_of_images: int
       Number of images used as input to generate the L1B product.
       Default is None, then this dimension is UNLIMITED.
     main program to create an empty L1C in-flight product
    """
    # size of the fixed dimensions
    n_views = 5
    n_bins_intens = 400
    n_bins_polar = 50
    n_bins_across = 40

    # create/overwrite netCDF4 product
    rootgrp = Dataset(l1c_flname, "w")

    # create global dimensions
    _ = rootgrp.createDimension('number_of_views', n_views)
    _ = rootgrp.createDimension('intensity_bands_per_view', n_bins_intens)
    _ = rootgrp.createDimension('polarization_bands_per_view', n_bins_polar)
    _ = rootgrp.createDimension('bins_across_track', n_bins_across)
    _ = rootgrp.createDimension('bins_along_track', number_of_images)

    # create groups and all variables with attributes
    sgrp = rootgrp.createGroup('SENSOR_VIEW_BANDS')
    dset = sgrp.createVariable('view_angles', 'f4', ('number_of_views',))
    dset.long_name = 'along track view zenith angles at sensor'
    dset.units = 'degrees'
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
    dset.long_name = 'spectral response function of intensity bands'
    dset.units = 'W.m-2'
    dset = sgrp.createVariable('polarization_f0', 'f4',
                               ('number_of_views',
                                'polarization_bands_per_view'))
    dset.long_name = 'spectral response function of polarization bands'
    dset.units = 'W.m-2'

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
    dset.units = "m"
    dset.positive = "up"
    dset.axis = "Z"
    dset = sgrp.createVariable('altitude_variability', 'f4',
                               ('bins_along_track', 'bins_across_track'),
                               chunksizes=chunksizes)
    dset.long_name = 'altitude (stdev)'
    dset.units = 'm'
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

    # add global attributes
    dict_attrs = global_attributes()
    dict_attrs['product_name'] = Path(l1c_flname).name
    dict_attrs['orbit_number'] = orbit_number
    dict_attrs['nadir_bin'] = n_bins_intens // 2
    for key in sorted(dict_attrs.keys()):
        rootgrp.setncattr(key, dict_attrs[key])

    rootgrp.close()

# --------------------------------------------------
if __name__ == '__main__':
    init_l1c('PACE_SPEX.20230115T123456.L1C.5km.V01.nc', orbit_number=12345)
