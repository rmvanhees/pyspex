#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Defines the format of a SPEXone Level-1A product."""

from __future__ import annotations

__all__ = ["init_l1a"]

from typing import TYPE_CHECKING

import numpy as np

# pylint: disable=no-name-in-module
from netCDF4 import Dataset, Variable

from .tmtc_def import tmtc_dtype

if TYPE_CHECKING:
    import datetime as dt

# - global parameters ------------------------------
ORBIT_DURATION = 5904  # seconds


# - local functions --------------------------------
def attrs_sec_per_day(dset: Variable, ref_date: dt.datetime) -> None:
    """Add CF attributes to a dataset holding 'seconds of day'.

    Parameters
    ----------
    dset : Variable
       NetCDF4 variable containing a timestamp as seconds since reference date
    ref_date : dt.datetime
       Reference date

    Examples
    --------
    Update the attributes of variable 'time':

      ref_date = dt.datetime(2022, 3, 21, tzinfo=dt.timezone.utc)
      dset = sgrp.createVariable('image_time', 'f8', ('number_of_images',),
                                fill_value=-32767)
      dset.long_name = "Image time"
      dset.description = "Integration start time in seconds of day."
      attrs_sec_per_day(dset, ref_date)

    In CDL the variable `time` will be defined as::

       double time(number_of_scans) ;
          time:_FillValue = -32767 ;
          time:long_name = "Time" ;
          time:units = "seconds since 2022-03-21 00:00:00" ;
          time:description = "Earth view mid-time in seconds of day" ;
          time:year = 2022 ;
          time:month = 3 ;
          time:day = 21 ;
          time:valid_min = 0 ;
          time:valid_max = 86401 ;

    Note that '_FillValue', 'long_name' and 'description' are not set by
    this function.

    """
    dset.units = f"seconds since {ref_date.strftime('%Y-%m-%d %H:%M:%S')}"
    dset.year = f"{ref_date.year}"
    dset.month = f"{ref_date.month}"
    dset.day = f"{ref_date.day}"
    dset.valid_min = 0
    dset.valid_max = 86400 + ORBIT_DURATION


def image_attributes(rootgrp: Dataset, ref_date: dt.datetime) -> None:
    """Define group /image_attributes and its datasets."""
    sgrp = rootgrp.createGroup("/image_attributes")
    dset = sgrp.createVariable("icu_time_sec", "u4", ("number_of_images",))
    dset.long_name = "ICU time stamp (seconds)"
    dset.description = "Science TM parameter ICU_TIME_SEC."
    dset.valid_min = np.uint32(1956528000)  # year 2020
    dset.valid_max = np.uint32(2493072000)  # year 2037
    dset.units = "seconds since 1958-01-01 00:00:00 TAI"
    dset = sgrp.createVariable("icu_time_subsec", "u2", ("number_of_images",))
    dset.long_name = "ICU time stamp (sub-seconds)"
    dset.description = "Science TM parameter ICU_TIME_SUBSEC."
    dset.valid_min = np.uint16(0)
    dset.valid_max = np.uint16(0xFFFF)
    dset.units = "1/65536 s"
    dset = sgrp.createVariable(
        "image_time", "f8", ("number_of_images",), fill_value=-32767
    )
    dset.long_name = "Image time"
    dset.description = "Integration start time in seconds of day."
    attrs_sec_per_day(dset, ref_date)
    dset = sgrp.createVariable(
        "timedelta_centre", "f8", ("number_of_images",), fill_value=-32767
    )
    dset.long_name = "Time-delta to centre of integration time"
    dset.description = "Add this offset to image-time (MPS specific)."
    dset.units = "s"
    dset = sgrp.createVariable(
        "nr_coadditions", "u2", ("number_of_images",), fill_value=0
    )
    dset.long_name = "Number of coadditions"
    dset.valid_min = np.int32(1)
    # dset.units = "1"
    dset = sgrp.createVariable(
        "exposure_time", "f8", ("number_of_images",), fill_value=0
    )
    dset.long_name = "Exposure time"
    dset.units = "s"
    dset = sgrp.createVariable("image_ID", "i4", ("number_of_images",))
    dset.long_name = "Image counter from power-up"
    dset.valid_min = np.int32(0)
    dset.valid_max = np.int32(0x7FFFFFFF)
    dset = sgrp.createVariable("binning_table", "u1", ("number_of_images",))
    dset.long_name = "Binning-table ID"
    dset.valid_min = np.uint8(0)
    dset.valid_max = np.uint8(0xFF)
    dset = sgrp.createVariable("digital_offset", "i2", ("number_of_images",))
    dset.long_name = "Digital offset"
    dset.units = "1"


def get_chunksizes(ydim: int, compression: bool) -> tuple[int, int]:
    """Obtain chunksizes for dataset: /science_data/science_data."""
    # I did some extensive testing.
    # - Without compression (chunked vs contiguous):
    #   * Writing to a contiguous dataset is faster (10-20%)
    #   * Reading one image is about as fast for chunked and contiguous storage
    #   * Reading a pixel image is much faster for chunked storage (2-8x)
    # - With compression (always chunked):
    #   * Writing takes 3x as long compared to without compression,
    #     but saves about > 40% on disk storage
    #   * Reading of compressed data is much slower than uncompressed data
    #   * The performance when reading one detector image is acceptable,
    #     however reading one pixel image is really slow (specially full-frame).
    # Therefore, these are the best choices for the variable `chunksizes`.
    return (
        (20, ydim)
        if ydim < 1048576
        else (1, min(512 * 1024, ydim))
        if compression
        else (1, ydim)
    )


def science_data(
    rootgrp: Dataset, compression: bool, chunksizes: tuple[int, int]
) -> None:
    """Define group /science_data and its datasets."""
    sgrp = rootgrp.createGroup("/science_data")
    dset = sgrp.createVariable(
        "detector_images",
        "u2",
        ("number_of_images", "samples_per_image"),
        compression="zlib" if compression else None,
        complevel=1,
        chunksizes=chunksizes,
        fill_value=0xFFFF,
    )
    dset.long_name = "Detector pixel values"
    dset.valid_min = np.uint16(0)
    dset.valid_max = np.uint16(0xFFFE)
    dset.units = "counts"
    hk_dtype = rootgrp.createCompoundType(tmtc_dtype(0x350), "science_dtype")
    dset = sgrp.createVariable(
        "detector_telemetry", hk_dtype, dimensions=("number_of_images",)
    )
    dset.long_name = "SPEXone Science telemetry"
    dset.comment = "A subset of MPS and housekeeping parameters."


def engineering_data(rootgrp: Dataset, ref_date: dt.datetime) -> None:
    """Define group /engineering_data and its datasets."""
    sgrp = rootgrp.createGroup("/engineering_data")
    dset = sgrp.createVariable("HK_tlm_time", "f8", ("hk_packets",), fill_value=-32767)
    dset.long_name = "HK telemetry packet time"
    dset.description = "Packaging time in seconds of day."
    attrs_sec_per_day(dset, ref_date)
    hk_dtype = rootgrp.createCompoundType(tmtc_dtype(0x320), "nomhk_dtype")
    dset = sgrp.createVariable("NomHK_telemetry", hk_dtype, ("hk_packets",))
    dset.long_name = "SPEXone nominal-HK telemetry"
    dset.comment = "An extended subset of the housekeeping parameters."
    dset = sgrp.createVariable("temp_detector", "f4", ("hk_packets",))
    dset.long_name = "`Detector temperature"
    dset.comment = "TS1 DEM Temperature (nominal)."
    dset.valid_min = np.float32(17.83)
    dset.valid_max = np.float32(18.83)
    dset.units = "degC"
    dset = sgrp.createVariable("temp_housing", "f4", ("hk_packets",))
    dset.long_name = "Housing temperature"
    dset.comment = "TS2 Housing Temperature (nominal)."
    dset.valid_min = np.float32(19.11)
    dset.valid_max = np.float32(20.11)
    dset.units = "degC"
    dset = sgrp.createVariable("temp_radiator", "f4", ("hk_packets",))
    dset.long_name = "Radiator temperature"
    dset.comment = "TS3 Radiator Temperature (nominal)."
    dset.valid_min = -2
    dset.valid_max = 3
    dset.units = "degC"
    # hk_dtype = rootgrp.createCompoundType(tmtc_dtype(0x322)), 'demhk_dtype')
    # dset = sgrp.createVariable('DemHK_telemetry', hk_dtype, ('hk_packets',))
    # dset.long_name = "SPEXone detector-HK telemetry"
    # dset.comment = "DEM housekeeping parameters."


# - main function ----------------------------------
def init_l1a(
    l1a_flname: str, ref_date: dt.datetime, dims: dict, compression: bool = False
) -> Dataset:
    """Create an empty SPEXone Level-1A product (on-ground or in-flight).

    Parameters
    ----------
    l1a_flname : str
       Name of L1A product
    ref_date :  dt.datetime
       Date of the first detector image (UTC)
    dims :  dict
       Provide length of the Level-1A dimensions. Default values::

          number_of_images : None     # number of image frames
          samples_per_image : None    # depends on binning table
          hk_packets : None           # number of HK tlm-packets (1 Hz)

    compression : bool, default=False
       Use compression on dataset /science_data/detector_images.

    Notes
    -----
    The optional groups '/gse_data' and '/navigation_data' are not created
    by this script.

    Original CDL definition is from F. S. Patt (GSFC), 08-Feb-2019

    """
    # check function parameters
    if not isinstance(dims, dict):
        raise TypeError("dims should be a dictionary")

    # initialize dimensions
    number_img = dims.get("number_of_images")
    img_samples = dims.get("samples_per_image")
    hk_packets = dims.get("hk_packets")

    # create/overwrite netCDF4 product
    try:
        rootgrp = Dataset(l1a_flname, "w")
    except Exception as exc:
        raise Exception(f"Failed to create netCDF4 file {l1a_flname}") from exc

    # - define global dimensions
    _ = rootgrp.createDimension("number_of_images", number_img)
    _ = rootgrp.createDimension("samples_per_image", img_samples)
    _ = rootgrp.createDimension("hk_packets", hk_packets)

    # - define the various HDF54/netCDF4 groups and their datasets
    image_attributes(rootgrp, ref_date)
    chunksizes = get_chunksizes(img_samples, compression)
    science_data(rootgrp, compression, chunksizes)
    engineering_data(rootgrp, ref_date)
    sgrp = rootgrp.createGroup("/processing_control")
    _ = sgrp.createGroup("input_parameters")

    return rootgrp
