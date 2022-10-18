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
Generate file-name of the SPEXone Level-1A products.
"""
__all__ = ['prod_name']

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
       Level of data in product, format: '^L[012][X_BAC]$' (3 characters)
    orbit: integer, optional
       Revolution counter, zero for on-ground measurements
    version_number: integer
       Version number of the product starting at 0001

    Notes
    -----
    The general format that applies to all in-flight SPEXone products:

    - [Science Product] PACE_SPEXone.yyyymmddThhmmss.LLL.VVV.nc
    - [Calibration Product] PACE_SPEXone_CAL.yyyymmddThhmmss.LLL.VVV.nc
    - [Monitoring Product] PACE_SPEXone_TTTTT.yyyymmddThhmmss.LLL.VVV.nc

    A Near Real-Time identifier can be added if required

    The general format that applies to all on-ground SPEXone products:

    - SPX1_OCAL_<msm_id>_LLL_yyyymmddThhmmss_YYYYMMDDTHHMMSS_vvvvv.nc

    """
    # check parameters
    if re.search(r'^L[012][ABCX_]$', level) is None:
        raise ValueError("parameter: level")

    if not isinstance(version_number, int):
        raise ValueError("parameter: version_number")

    # define string of sensing start as yyyymmddThhmmss
    sensing_start = utc_sensing_start.strftime("%Y%m%dT%H%M%S")

    # in-flight product when no MSM identifier is provided
    if msm_id is None:
        if orbit is not None and not isinstance(orbit, int):
            raise ValueError("parameter: orbit")

        if file_class not in ('OPER', 'CONS', 'TEST'):
            raise ValueError("parameter: file_class")

        # should be None, 'CAL' or 'MON*'
        if data_type is not None:
            return (f'PACE_SPEXone_{data_type}.{sensing_start:15s}'
                    f'.{level:3s}.V{version_number:02d}.nc')

        return (f'PACE_SPEXone.{sensing_start:15s}'
                f'.{level:3s}.V{version_number:02d}.nc')

    # on-ground product
    return (f'SPX1_OCAL_{msm_id}_L1A_{sensing_start:15s}'
            f'_{datetime.utcnow().strftime("%Y%m%dT%H%M%S"):15s}'
            f'_{version_number:04d}.nc')
