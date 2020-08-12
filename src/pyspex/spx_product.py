"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Generate product name

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime

import re

# - global parameters ------------------------------


# - local functions --------------------------------


# - main functions ---------------------------------
# pylint: disable=too-many-arguments
def prod_name(utc_sensing_start, msm_id=None,
              file_class='TEST', data_type='CA', level='L1A',
              orbit=None, version_number=1):
    """
    Return name of SPEXone product

    Parameters
    ----------
    utc_sensing_start: datetime.datetime
       Sensing/Validity start
    msm_id : string, optional
       Provide identifier for measurement, OCAL only
    file_class : string
       File class: OPER, CONS, TEST, ...
    data_type: string
       Product type as SPX1_<data_type>, where data type can be:
       CA (calibration data), RA (radiance data), ...
    level : string
       Level of data in product, format: ^L[012][ABCX_]$ (3 characters)
    orbit: integer, optional
       Revolution counter, zero for on-ground measurements
    version_number: integer
       Version number of the product starting at 0001

    Notes
    -----
    The general format that applies to all in-flight SPEXone products:

       PACE_CCCC_SPX1_PP_LLL_OOOOO_yyyymmddThhmmss_YYYYMMDDTHHMMSS_vvvvv.nc

    The general format that applies to all on-ground SPEXone products:

       SPX1_OCAL_<msm_id>_LLL_yyyymmddThhmmss_YYYYMMDDTHHMMSS_vvvvv.nc
    """
    # check parameters
    if re.search(r"^L[012][ABCX_]$", level) is None:
        raise ValueError("parameter: level")

    if not isinstance(version_number, int):
        raise ValueError("parameter: version_number")

    # define product ID
    if msm_id is None:
        if orbit is not None and not isinstance(orbit, int):
            raise ValueError("parameter: orbit")

        if file_class not in ('OPER', 'CONS', 'TEST'):
            raise ValueError("parameter: file_class")

        if data_type not in ('CA', 'RA'):
            raise ValueError("parameter: data_type")

        product_id = 'PACE_{:4s}_SPX1_{:2s}_{:3s}_{:05d}'.format(
            file_class, data_type, level, orbit)
    else:
        product_id = 'SPX1_OCAL_{:s}_L1A'.format(msm_id)

    # define string of sensing start as yyyymmddThhmmss
    sensing_start = utc_sensing_start.strftime("%Y%m%dT%H%M%S")

    # define instance ID
    return '{}_{:15s}_{:15s}_{:04d}.nc'.format(
        product_id, sensing_start,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"), version_number)
