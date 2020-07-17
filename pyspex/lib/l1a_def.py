"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation of the PACE SPEX Level-1A product (inflight and on-ground)

Copyright (c) 2019 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from pyspex.lib.tmtc_def import tmtc_def

# - global parameters ------------------------------


# - local functions --------------------------------
def global_attrs(inflight, origin=None) -> dict:
    """
    Generate dictionary with all required global attributes
    """
    if origin is None:
        origin = 'NASA' if inflight else 'SRON'

    res = {}
    if not inflight:
        res['title'] = "SPEXone Level-1A data"
        res['cdm_data_type'] = "on-ground calibration"
        res['data_collect_mode'] = "on-ground"
    else:
        res['title'] = "PACE SPEX Level-1A data"
        res['cdm_data_type'] = "swath"
        res['startdirection'] = "Ascending"
        res['enddirection'] = "Ascending"
        res['data_collect_mode'] = "Earth Collect"
        res['bin_size_at_nadir'] = '2.5 km'

    # Independed attributes (OCAL/ICAL, NASA/SRON ...)
    res['instrument'] = "SPEX"
    res['processing_version'] = "V1.0"
    res['conventions'] = "CF-1.6"
    res['cdl_version_date'] = "2020-02-20"
    res['history'] = ""
    res['license'] = ("http://science.nasa.gov/earth-science/"
                      "earth-science-data/data-information-policy/")
    res['naming_authority'] = "gov.nasa.gsfc.sci.oceancolor"
    res['keywords_vocabulary'] = ("NASA Global Change Master Directory (GCMD)"
                                  " Science Keywords")
    res['stdname_vocabulary'] = ("NetCDF Climate and Forecast (CF)"
                                 " Metadata Convention")
    res['project'] = "PACE Project"
    res['processing_level'] = "L1A"

    if origin == 'SRON':
        res['institution'] = "SRON Netherlands Institute for Space Research"
        res['creator_name'] = "SRON/Earth Science"
        res['creator_email'] = "spex_mpc@sron.nl"
        res['creator_url'] = "https://www.sron.nl/earth"
        res['publisher_name'] = "SRON/Earth Science"
        res['publisher_email'] = "spex_mpc@sron.nl"
        res['publisher_url'] = "https://www.sron.nl/earth"
    else:
        res['institution'] = ("NASA Goddard Space Flight Center,"
                              " Ocean Biology Processing Group")
        res['creator_name'] = "NASA/GSFC"
        res['creator_email'] = "data@oceancolor.gsfc.nasa.gov"
        res['creator_url'] = "http://oceancolor.gsfc.nasa.gov"
        res['publisher_name'] = "NASA/GSFC"
        res['publisher_email'] = "data@oceancolor.gsfc.nasa.gov"
        res['publisher_url'] = "http://oceancolor.gsfc.nasa.gov"

    res['time_coverage_start'] = 'yyyy-mm-ddThh:mm:ss.sssZ'
    res['time_coverage_end'] = 'yyyy-mm-ddThh:mm:ss.sssZ'
    res['date_created'] = datetime.utcnow().isoformat(timespec='milliseconds')

    return res


# -------------------------
def create_group_gse_data(fid, n_wave=None):
    """
    Define datasets and attributes in the group /gse_data
    """
    gid = fid.createGroup('/gse_data')

    dset = gid.createVariable('viewport', 'u1')
    dset.long_name = "viewport status"
    dset.valid_range = np.array([0, 16], dtype='u1')
    dset.comment = "bitmask: 1, 2, 4, 8, 16"
    dset[:] = 0x0  # initialize to default value: all viewports used

    if n_wave is not None:
        gid.createDimension('wavelength', n_wave)

        dset = gid.createVariable('wavelength', 'f8', ('wavelength',))
        dset.long_name = "wavelength of stimulus"
        dset.units = 'nm'

        dset = gid.createVariable('signal', 'f8', ('wavelength',))
        dset.long_name = "signal of stimulus"
        dset.units = 'photons.s-1.nm-1.m-2)'


# -------------------------
def init_l1a(l1a_flname: str, dims: dict,
             orbit_number=-1, inflight=False) -> None:
    """
    Create an empty OCAL SPEXone or inflight PACE SPEX Level-1A product

    Parameters
    ----------
    l1a_flname : string
       Name of L1A product
    dims :   dictionary
       Provide alternative lengths for a dimension
       Default values:
            number_of_images : None     # number of image frames
            samples_per_image : 184000  # depends on binning table
            SC_records : None           # space-craft navigation records (1 Hz)
            hk_packets : None           # number of HK tlm-packets ( Hz)
            wavelength : None
    inflight:  boolean, optional
       True for in-flight measurements
       False for on-ground measurements (Default)

    Notes
    -----
    Original design Frederick S. Patt (Goddard Space Flight Center), 08-Feb-2019
    Modified for on-ground calibration measurements, Richard M. van Hees (SRON)
    """
    # check function parameters
    if not isinstance(dims, dict):
        raise TypeError("dims should be a dictionary")

    # initialize dimensions
    number_img = None
    img_samples = 184000
    hk_packets = None
    sc_records = None
    n_wave = None

    if 'number_of_images' in dims:
        number_img = dims['number_of_images']
    if 'samples_per_image' in dims:
        img_samples = dims['samples_per_image']
    if 'SC_records' in dims:
        sc_records = dims['SC_records']
    if 'hk_packets' in dims:
        hk_packets = dims['hk_packets']
    if 'wavelength' in dims:
        n_wave = dims['wavelength']

    with Dataset(l1a_flname, 'w') as fid:
        # - define dimensions -------------------------
        _ = fid.createDimension('number_of_images', number_img)
        _ = fid.createDimension('samples_per_image', img_samples)
        _ = fid.createDimension('hk_packets', hk_packets)
        _ = fid.createDimension('SC_records', sc_records)
        _ = fid.createDimension('quaternion_elements', 4)
        _ = fid.createDimension('vector_elements', 3)

        # - define group /image_attributs and its datasets
        gid = fid.createGroup('/image_attributes')
        dset = gid.createVariable('image_time', 'f8',
                                  ('number_of_images',))
        dset.long_name = "image time (seconds of day)"
        dset.valid_min = 0
        dset.valid_max = 86400.999999
        dset.reference = "yyyy-mm-dd"
        dset.units = "seconds"
        dset = gid.createVariable('image_CCSDS_sec', 'u4',
                                  ('number_of_images',))
        dset.long_name = "image CCSDS time (seconds since 1970)"
        dset.valid_min = np.uint32(1577500000)  # year 2020
        dset.valid_max = np.uint32(2050000000)  # year 2035
        dset.units = "seconds"
        dset = gid.createVariable('image_CCSDS_usec', 'i4',
                                  ('number_of_images',))
        dset.long_name = "image CCSDS time (microseconds)"
        dset.valid_min = np.int32(0)
        dset.valid_max = np.int32(999999)
        dset.units = "microseconds"
        dset = gid.createVariable('image_ID', 'i4',
                                  ('number_of_images',))
        dset.long_name = "image counter from power-up"
        dset.valid_min = np.int32(0)
        dset.valid_max = np.int32(0x7FFFFFFF)
        dset = gid.createVariable('binning_table', 'u1',
                                  ('number_of_images',))
        dset.long_name = "binning-table ID"
        dset.valid_min = np.uint8(0)
        dset.valid_max = np.uint8(0xFF)
        dset = gid.createVariable('digital_offset', 'i2',
                                  ('number_of_images',))
        dset.long_name = "digital offset"
        dset.units = "1"
        dset = gid.createVariable('nr_coadditions', 'u2',
                                  ('number_of_images',))
        dset.long_name = "number of coadditions"
        dset.units = "1"
        dset = gid.createVariable('exposure_time', 'f8',
                                  ('number_of_images',))
        dset.long_name = "exposure time"
        dset.units = "seconds"

        # - define group /science_data and its datasets
        gid = fid.createGroup('/science_data')
        chunksizes = None if number_img is not None else (1, img_samples)
        dset = gid.createVariable('detector_images', 'u2',
                                  ('number_of_images', 'samples_per_image'),
                                  chunksizes=chunksizes, fill_value=0)
        dset.long_name = "Image data from detector"
        dset.valid_min = np.uint16(0)
        dset.valid_max = np.uint16(0xFFFF)
        dset.units = "counts"
        mps_dtype = fid.createCompoundType(np.dtype(tmtc_def(0x350)),
                                           'mps_dtype')
        dset = gid.createVariable('detector_telemetry', mps_dtype,
                                  ('number_of_images',))
        dset.long_name = "SPEX science telemetry"
        dset.comment = "Measurement Parameter Settings"

        # - define group /engineering_data and its datasets
        gid = fid.createGroup('/engineering_data')
        dset = gid.createVariable('HK_tlm_time', 'f8', ('hk_packets',))
        dset.long_name = "HK telemetry packet time (seconds of day)"
        dset.valid_min = 0
        dset.valid_max = 86400.999999
        dset.units = "seconds"
        hk_dtype = fid.createCompoundType(np.dtype(tmtc_def(0x320)),
                                          'hk_dtype')
        dset = gid.createVariable('HK_telemetry', hk_dtype, ('hk_packets',))
        dset.long_name = "SPEX nominal-HK telemetry"
        dset = gid.createVariable('temp_optics', 'f4', ('hk_packets',))
        dset.long_name = "Optics temperature"
        dset.valid_min = 260
        dset.valid_max = 300
        dset.units = "K"
        dset = gid.createVariable('temp_detector', 'f4', ('hk_packets',))
        dset.long_name = "Detector temperature"
        dset.valid_min = 260
        dset.valid_max = 300
        dset.units = "K"

        # - define group /navigation_data and its datasets
        gid = fid.createGroup('/navigation_data')
        dset = gid.createVariable('adstate', 'u1',
                                  ('SC_records',), fill_value=255)
        dset.long_name = "Current ADCS State"
        dset.flag_values = np.array([0, 1, 2, 3, 4, 5], dtype='u1')
        dset.flag_meanings = "Wait Detumple AcqSun Point Delta Earth"
        dset = gid.createVariable('att_time', 'f8', ('SC_records',))
        dset.long_name = "Attitude sample time (seconds of day)"
        dset.valid_min = 0.0
        dset.valid_max = 86400.999999
        dset.units = "seconds"
        chunksizes = None if sc_records is not None else (256, 4)
        dset = gid.createVariable('att_quat', 'f4',
                                  ('SC_records', 'quaternion_elements'),
                                  chunksizes=chunksizes)
        dset.long_name = "Attitude quaternions (J2000 to spacecraft)"
        dset.valid_min = -1
        dset.valid_max = 1
        dset.units = "1"
        dset = gid.createVariable('orb_time', 'f8', ('SC_records',))
        dset.long_name = "Orbit vector time (seconds of day)"
        dset.valid_min = 0
        dset.valid_max = 86400.999999
        dset.units = "seconds"
        chunksizes = None if sc_records is not None else (340, 3)
        dset = gid.createVariable('orb_pos', 'f4',
                                  ('SC_records', 'vector_elements'),
                                  chunksizes=chunksizes)
        dset.long_name = "Orbit positions vectors (J2000)"
        dset.valid_min = -7200000
        dset.valid_max = 7200000
        dset.units = "meters"
        dset = gid.createVariable('orb_vel', 'f4',
                                  ('SC_records', 'vector_elements'),
                                  chunksizes=chunksizes)
        dset.long_name = "Orbit velocity vectors (J2000)"
        dset.valid_min = -7600
        dset.valid_max = 7600
        dset.units = "meters s-1"

        if not inflight:
            create_group_gse_data(fid, n_wave)

        # - define global attributes ------------------
        dict_attrs = global_attrs(inflight)
        dict_attrs['product_name'] = Path(l1a_flname).name
        dict_attrs['orbit_number'] = orbit_number
        for key in sorted(dict_attrs.keys()):
            fid.setncattr(key, dict_attrs[key])


# --------------------------------------------------
if __name__ == '__main__':
    init_l1a('PACE_SPEX.20230115T123456.L1A.2.5km.V01.nc', {},
             inflight=True, orbit_number=12345)

    init_l1a('SPX1_OCAL_msm_id_L1A_20190306T123456_20200310T195841_0001.nc',
             {'samples_per_image': 2048**2, 'wavelength': 2048},
             inflight=False, orbit_number=-1)
