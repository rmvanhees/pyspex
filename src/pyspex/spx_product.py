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
__all__ = ['get_l1a_name']

from datetime import datetime

from . import version


# - main function ----------------------------------
# pylint: disable=too-many-arguments
def get_l1a_name(msm_id: str, utc_sensing_start: datetime | None) -> str:
    """
    Return name of SPEXone product

    Parameters
    ----------
    msm_id : string, optional
       Provide identifier for measurement, OCAL only

    Notes
    -----
    L1A file name format:
       SPX1_OCAL_<msm_id>[_YYYYMMDDTHHMMSS]_L1A_vvvvvvv.nc
    where
       msm_id is the measurement identifier
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       vvvvvvv is the git-hash string of the pyspex repository
    """
    # define string of sensing start as yyyymmddThhmmss
    if utc_sensing_start is None:
        return f'SPX1_OCAL_{msm_id}_L1A_{version.get(githash=True)}.nc'

    sensing_start = utc_sensing_start.strftime("%Y%m%dT%H%M%S")

    return (f'SPX1_OCAL_{msm_id}_{sensing_start}'
            f'_L1A_{version.get(githash=True)}.nc')
