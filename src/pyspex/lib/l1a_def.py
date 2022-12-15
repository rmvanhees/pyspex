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
Defines the format of a SPEXone Level-1A product.
"""
__all__ = ['init_l1a']

import datetime
import numpy as np

from netCDF4 import Dataset

from .tmtc_def import tmtc_dtype

# - global parameters ------------------------------
ORBIT_DURATION = 5904  # seconds


# - local functions --------------------------------
def attrs_sec_per_day(dset, ref_date: datetime.date) -> None:
    """
    Add CF attributes to a dataset holding 'seconds of day'

    Parameters
    ----------
    dset : h5py.Dataset
       Variable containing a timestamp as seconds since referencce date
    ref_date : datetime.date
       Reference date

    Examples
    --------
    Update the attributes of variable 'time':

    >>> ref_date = datetime.date(2022, 03, 21)
    >>> dset = sgrp.createVariable('image_time', 'f8', ('number_of_images',),
    >>>                            fill_value=-32767)
    >>> dset.long_name = "image time"
    >>> dset.description = "Integration start time in seconds of day."
    >>> attrs_sec_per_day(dset, ref_date)

    In CDL the variable `time` will be defined as::

       double time(number_of_scans) ;
          time:_FillValue = -32767. ;
          time:long_name = "time" ;
          time:units = "seconds since 2022-03-21 00:00:00" ;
          time:description = "Earth view mid time in seconds of day" ;
          time:year = 2022 ;
          time:month = 3 ;
          time:day = 21 ;
          time:valid_min = 0. ;
          time:valid_max = 86401. ;

    Note that '_FillValue', 'long_name' and 'description' are not set by
    this function.
    """
    dset.units = f"seconds since {ref_date.isoformat()} 00:00:00"
    dset.year = f"{ref_date.year}"
    dset.month = f"{ref_date.month}"
    dset.day = f"{ref_date.day}"
    dset.valid_min = 0
    dset.valid_max = 86400 + ORBIT_DURATION


# - main function ----------------------------------
# pylint: disable=too-many-statements
def init_l1a(l1a_flname: str, ref_date: datetime.date, dims: dict) -> None:
    """
    Create an empty SPEXone Level-1A product (on-ground or in-flight)

    Parameters
    ----------
    l1a_flname : string
       Name of L1A product
    ref_date :  datetime.date
       Date of the first detector image
    dims :   dictionary
       Provide length of the Level-1A dimensions. Default values::

          number_of_images : None     # number of image frames
          samples_per_image : None    # depends on binning table
          hk_packets : None           # number of HK tlm-packets (1 Hz)

    Notes
    -----
    Original CDL definition is from F. S. Patt (GSFC), 08-Feb-2019
    """
    # check function parameters
    if not isinstance(dims, dict):
        raise TypeError("dims should be a dictionary")

    # initialize dimensions
    number_img = dims.get('number_of_images', None)
    img_samples = dims.get('samples_per_image', None)
    hk_packets = dims.get('hk_packets', None)

    # create/overwrite netCDF4 product
    try:
        rootgrp = Dataset(l1a_flname, 'w')
    except Exception as exc:
        raise Exception(f'Failed to create netCDF4 file {l1a_flname}') from exc

    # - define global dimensions
    _ = rootgrp.createDimension('number_of_images', number_img)
    _ = rootgrp.createDimension('samples_per_image', img_samples)
    _ = rootgrp.createDimension('hk_packets', hk_packets)

    # - define group /image_attributs and its datasets
    sgrp = rootgrp.createGroup('/image_attributes')
    dset = sgrp.createVariable('icu_time_sec', 'u4', ('number_of_images',))
    dset.long_name = "ICU time stamp (seconds)"
    dset.description = "Science TM parameter ICU_TIME_SEC."
    dset.valid_min = np.uint32(1956528000)  # year 2020
    dset.valid_max = np.uint32(2493072000)  # year 2037
    dset.units = "seconds since 1958-01-01 00:00:00 TAI"
    dset = sgrp.createVariable('icu_time_subsec', 'u2', ('number_of_images',))
    dset.long_name = "ICU time stamp (sub-seconds)"
    dset.description = "Science TM parameter ICU_TIME_SUBSEC."
    dset.valid_min = np.uint16(0)
    dset.valid_max = np.uint16(0xFFFF)
    dset.units = "1/65536 s"

    dset = sgrp.createVariable('image_time', 'f8', ('number_of_images',),
                               fill_value=-32767)
    dset.long_name = "image time"
    dset.description = "Integration start time in seconds of day."
    attrs_sec_per_day(dset, ref_date)
    dset = sgrp.createVariable('image_ID', 'i4', ('number_of_images',))
    dset.long_name = "image counter from power-up"
    dset.valid_min = np.int32(0)
    dset.valid_max = np.int32(0x7FFFFFFF)
    dset = sgrp.createVariable('binning_table', 'u1', ('number_of_images',))
    dset.long_name = "binning-table ID"
    dset.valid_min = np.uint8(0)
    dset.valid_max = np.uint8(0xFF)
    dset = sgrp.createVariable('digital_offset', 'i2', ('number_of_images',))
    dset.long_name = "digital offset"
    dset.units = "1"
    dset = sgrp.createVariable('nr_coadditions', 'u2', ('number_of_images',),
                               fill_value=0)
    dset.long_name = "number of coadditions"
    dset.valid_min = np.int32(1)
    dset.units = "1"
    dset = sgrp.createVariable('exposure_time', 'f8', ('number_of_images',),
                               fill_value=0)
    dset.long_name = "exposure time"
    dset.units = "s"

    # - define group /science_data and its datasets
    sgrp = rootgrp.createGroup('/science_data')
    chunksizes = None if number_img is not None else (1, img_samples)
    dset = sgrp.createVariable('detector_images', 'u2',
                               ('number_of_images', 'samples_per_image'),
                               chunksizes=chunksizes, fill_value=0xFFFF)
    dset.long_name = "detector pixel values"
    dset.valid_min = np.uint16(0)
    dset.valid_max = np.uint16(0xFFFE)
    dset.units = "counts"
    hk_dtype = rootgrp.createCompoundType(tmtc_dtype(0x350), 'science_dtype')
    dset = sgrp.createVariable('detector_telemetry', hk_dtype,
                               dimensions=('number_of_images',))
    dset.long_name = "SPEX science telemetry"
    dset.comment = "A subset of MPS and housekeeping parameters."

    # - define group /engineering_data and its datasets
    sgrp = rootgrp.createGroup('/engineering_data')
    dset = sgrp.createVariable('HK_tlm_time', 'f8', ('hk_packets',),
                               fill_value=-32767)
    dset.long_name = "HK telemetry packet time"
    dset.description = "Packaging time in seconds of day."
    attrs_sec_per_day(dset, ref_date)
    hk_dtype = rootgrp.createCompoundType(tmtc_dtype(0x320), 'nomhk_dtype')
    dset = sgrp.createVariable('NomHK_telemetry', hk_dtype, ('hk_packets',))
    dset.long_name = "SPEX nominal-HK telemetry"
    dset.comment = "An extended subset of the housekeeping parameters."
    dset = sgrp.createVariable('temp_detector', 'f4', ('hk_packets',))
    dset.long_name = "detector temperature"
    dset.comment = "TS1 DEM Temperature (nominal)."
    dset.valid_min = 260
    dset.valid_max = 300
    dset.units = "K"
    dset = sgrp.createVariable('temp_housing', 'f4', ('hk_packets',))
    dset.long_name = "housing temperature"
    dset.comment = "TS2 Housing Temperature (nominal)."
    dset.valid_min = 260
    dset.valid_max = 300
    dset.units = "K"
    dset = sgrp.createVariable('temp_radiator', 'f4', ('hk_packets',))
    dset.long_name = "radiator temperature"
    dset.comment = "TS3 Radiator Temperature (nominal)."
    dset.valid_min = 260
    dset.valid_max = 300
    dset.units = "K"
    # hk_dtype = rootgrp.createCompoundType(tmtc_dtype(0x322)), 'demhk_dtype')
    # dset = sgrp.createVariable('DemHK_telemetry', hk_dtype, ('hk_packets',))
    # dset.long_name = "SPEX detector-HK telemetry"
    # dset.comment = "DEM housekeeping parameters."

    # - define group /navigation_data, but leaf it's content empty
    # The actual navigation data will be copied from the HKT products.
    sgrp = rootgrp.createGroup('/navigation_data')

    return rootgrp
