#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Contains the classes `L1Aio`, L1Bio and `L1Cio` to write
PACE/SPEXone data in resp. Level-1A, Level-1B or Level-1C format.
"""
from __future__ import annotations
__all__ = ['L1Aio', 'L1Bio', 'L1Cio', 'write_l1a']

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath

import netCDF4 as nc4
import numpy as np

from .hkt_io import read_hkt_nav, write_hkt_nav
from .lib.attrs_def import attrs_def
from .lib.l1a_def import init_l1a
from .lib.l1b_def import init_l1b
from .lib.l1c_def import init_l1c
from .lv0_io import (coverage_time, hk_sec_of_day, img_sec_of_day,
                     nomhk_timestamps, science_timestamps, select_nomhk,
                     select_science)
from .tm_science import TMscience
from .version import pyspex_version

# - global parameters -------------------
ONE_DAY = 24 * 60 * 60


# - local functions ---------------------
def frac_poly(xx_in: np.ndarray,
              coefs: tuple[float, float, float, float, float] | None = None):
    """Temperature [K] calibration derived by Paul Tol (2020-10-21).

    Parameters
    ----------
    xx_in :  ndarray
    coefs :  tuple, default=None
      Coefficients of fractional polynomial: r0, r1, r2, r3, r4

    Returns
    -------
    ndarray, dtype float
    """
    xdata = xx_in.astype(float)

    if coefs is None:
        coefs = (273.15 + 21.19, 6.97828e+7,
                 -3.53275e-25, 7.79625e-31, -4.6505E-32)

    return (coefs[0]
            + coefs[1] / xdata
            + coefs[2] * xdata ** 4
            + (coefs[3] + coefs[4] * np.log(xdata)) * xdata ** 5)


def get_l1a_name(config: dataclass, mode: str,
                 sensing_start: datetime) -> str:
    """
    Generate name of Level-1A product based on filename conventions described
    below

    Parameters
    ----------
    config :  dataclass
       Settings for the L0->l1A processing.
    mode :  {'all', 'full', 'binned'}, default='all'
       Select Science packages with full-frame image or binned images
    sensing_start :  datetime
       Start date/time of the first detector frame

    Returns
    -------
    str
        Name of Level-1A product.

    Notes
    -----

    === Inflight ===
    L1A file name format, following the NASA ... naming convention:
       PACE_SPEXONE[_TTT].YYYYMMDDTHHMMSS.L1A[.Vnn].nc
    where
       TTT is an optional data type (e.g., for the calibration data files)
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       Vnn file-version number (ommited when nn=1)
    for example (file-version=1):
       [Science Product] PACE_SPEXONE.20230115T123456.L1A.nc
       [Calibration Product] PACE_SPEXONE_CAL.20230115T123456.L1A.nc
       [Dark science Product] PACE_SPEXONE_DARK.20230115T123456.L1A.nc

    === OCAL ===
    L1A file name format:
       SPX1_OCAL_<msm_id>[_YYYYMMDDTHHMMSS]_L1A_vvvvvvv.nc
    where
       msm_id is the measurement identifier
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       vvvvvvv is the git-hash string of the pyspex repository
    """
    if config.outfile:
        return config.outfile

    if config.l0_format != 'raw':
        if config.eclipse is None:
            subtype = '_OCAL'
        elif not config.eclipse:
            subtype = ''
        else:
            subtype = '_CAL' if mode == 'full' else '_DARK'

        prod_ver = '' if config.file_version == 1\
            else f'.V{config.file_version:02d}'

        return (f'PACE_SPEXONE{subtype}'
                f'.{sensing_start.strftime("%Y%m%dT%H%M%S"):15s}.L1A'
                f'{prod_ver}.nc')

    # OCAL product name
    # determine measurement identifier
    msm_id = config.l0_list[0].stem
    try:
        new_date = datetime.strptime(
            msm_id[-22:], '%y-%j-%H:%M:%S.%f').strftime('%Y%m%dT%H%M%S.%f')
    except ValueError:
        pass
    else:
        msm_id = msm_id[:-22] + new_date

    return f'SPX1_OCAL_{msm_id}_L1A_{pyspex_version(githash=True)}.nc'


def write_lv0_data(prod_name: str, config: dataclass, nomhk: np.ndarray,
                   science: np.ndarray, images: np.ndarray) -> None:
    """
    Write level 0 packages to a level-1A product

    Parameters
    ----------
    prod_name :  str
       Name of the Level-1A product.
    config :  dataclass
       Settings for the L0->l1A processing.
    nomhk : np.ndarray
       L0 nominal housekeeping data.
    science : np.ndarray
       L0 telementry without detector data.
    images :  np.ndarray
       L0 detector read-outs
    """
    # Define data dimensions
    dims = {'number_of_images': images.shape[0],
            'samples_per_image': images.shape[1],
            'hk_packets': nomhk.size}

    # Preprocess the timestamps to be stored in the L1A product
    # [Science TM]
    #  - the ICU time stamps are not altered!
    #  - the variable 'image_time' holds sec_of_day and needs a reference day
    #    in addition, the data is corrected for the start of integration time.
    # [nomHK TM]
    #  - the CCSDS time stamps are not altered!
    #  - the variable 'nomhk_time' holds sec_of_day and needs a reference day
    # Only the DSB files use EPOCH 1958 (TAI) all other use EPOCH 1970 (UTC)
    img_sec, img_subsec = science_timestamps(science)
    ref_date, img_time = img_sec_of_day(img_sec, img_subsec, science['hk'])

    # Generate and fill L1A product
    with L1Aio(config.outdir / prod_name, ref_date.date(), dims,
               compression=config.compression) as l1a:
        # write image data, detector telemetry and image attributes
        l1a.fill_science(images, science['hk'],
                         np.bitwise_and(science['hdr']['sequence'], 0x3fff))
        l1a.set_dset('/image_attributes/icu_time_sec', img_sec)
        # modify attribute units for non-DSB products
        if config.l0_format != 'dsb':
            l1a.set_attr('valid_min', np.uint32(1577800000),
                         ds_name='/image_attributes/icu_time_sec')
            l1a.set_attr('valid_max', np.uint32(1735700000),
                         ds_name='/image_attributes/icu_time_sec')
            l1a.set_attr('units', "seconds since 1970-01-01 00:00:00",
                         ds_name='/image_attributes/icu_time_sec')
        l1a.set_dset('/image_attributes/icu_time_subsec', img_subsec)
        l1a.set_dset('/image_attributes/image_time', img_time)

        # write engineering data
        if nomhk.size > 0:
            l1a.fill_nomhk(nomhk['hk'])
            nomhk_sec, nomhk_subsec = nomhk_timestamps(nomhk)
            hk_time = hk_sec_of_day(nomhk_sec, nomhk_subsec, ref_date)
            l1a.set_dset('/engineering_data/HK_tlm_time', hk_time)

        # write global attributes
        if nomhk.size > 0:
            l1a.set_attr('icu_sw_version', f'0x{nomhk["hk"]["ICUSWVER"][0]:x}')
        if config.l0_format == 'raw':
            l1a.fill_global_attrs(inflight=False)
        else:
            l1a.fill_global_attrs(inflight=True)
        tstamp = coverage_time(science)
        l1a.set_attr('time_coverage_start',
                     tstamp[0].isoformat(timespec='milliseconds'))
        l1a.set_attr('time_coverage_end',
                     tstamp[1].isoformat(timespec='milliseconds'))
        l1a.set_attr('input_files', [x.name for x in config.l0_list])

    # add processor_configuration
    if config.yaml_fl:
        with nc4.Dataset(config.outdir / prod_name, 'r+') as fid:
            dset = fid.createVariable('processor_configuration', str)
            dset.comment = ('Configuration parameters used during'
                            ' the processor run that produced this file.')
            dset[0] = ''.join(
                [s for s in config.yaml_fl.open(encoding='ascii').readlines()
                 if not (s == '\n' or s.startswith('#'))])


# - high-level write function -----------
def write_l1a(config, science_in: tuple[np.ndarray], nomhk_in: tuple[np.ndarray]) -> None:
    """Write Level-1A product.

    Parameters
    ----------
    config :  dataclass
       Settings for the L0->l1A processing.
    science_in : tuple of np.ndarray
       L0 detector data.
    nomhk_in : tuple of np.ndarray
       L0 nominal housekeeping data.
    """
    def reject_tm(tm_sec, array):
        """reject corrupted timestamps.

        Notes
        -----
        This function is implemented for the day-in-a-life test at Godard.
        Date: Jan, 2023
        """
        indx = (np.abs(tm_sec[1:] - tm_sec[:-1]) > ONE_DAY).nonzero()[0]
        print(f'[WARNING] found large jumps between timestamps at {indx}')
        if len(indx) == 1:
            array = array[:indx[0]+1] \
                if indx[0] > array.size // 2 else array[indx[0]+1:]
        elif len(indx) == 2:
            if indx[1] - indx[0] > array.size // 2:
                array = array[indx[0]+1:indx[1]+1]
            else:
                mask = np.ones(array.size, dtype=bool)
                mask[indx[0]+1:indx[1]+1] = False
                array = array[mask]
        else:
            print('[WARNING] cound not reject corrupted timestamps')
        return array

    if config.eclipse is None:
        # these are "OCAL data" try to write all data to one L1A product.
        mode_list = ['all']
    elif not config.eclipse:
        # these are "Science data": always binned data
        mode_list = ['binned']
    else:
        # this can be "Dark data": binned data using "Science mode" MPSes
        # and/or "Calibration data”: full frame data in "Diagonstic mode".
        mode_list = ['binned', 'full']

    # print(mode_list, len(science_in), type(science_in),
    #       len(nomhk_in), type(nomhk_in))
    for mode in mode_list:
        # selected L0 data-packages
        # and group Science packages to detector-frames
        science, images = select_science(science_in, mode)
        if science.size == 0:
            continue

        mps_list = [int(i) for i in np.unique(science['hk']['MPS_ID'])]
        if config.verbose:
            print(f'[INFO]: list of unique MPS {mps_list}')
        nomhk = select_nomhk(nomhk_in, mps_list)

        # reject corrupted timestamps in science data
        img_sec, _ = science_timestamps(science)
        img_sec = img_sec.astype(int)
        if np.any(np.abs(img_sec[1:] - img_sec[:-1]) > ONE_DAY):
            science = reject_tm(img_sec, science)

        # reject corrupted timestamps in house-keeping data
        hk_sec, _ = nomhk_timestamps(nomhk)
        hk_sec = hk_sec.astype(int)
        if np.any(np.abs(hk_sec[1:] - hk_sec[:-1]) > ONE_DAY):
            nomhk = reject_tm(hk_sec, nomhk)

        # write L1A product
        prod_name = get_l1a_name(config, mode, coverage_time(science)[0])
        try:
            write_lv0_data(prod_name, config, nomhk, science, images)
        except (PermissionError, RuntimeError) as exc:
            raise RuntimeError from exc

        # add PACE navigation information from HKT products
        if config.hkt_list:
            hkt_nav = read_hkt_nav(config.hkt_list)
            # select HKT data collocated with Science data
            # - issue a warning if selection is empty
            write_hkt_nav(config.outdir / prod_name, hkt_nav)

        if config.verbose:
            print(f'[INFO]: Successfully generated: {prod_name}')


# - class LV1io -------------------------
class Lv1io:
    """
    Generic class to create SPEXone Level-1 products

    Parameters
    ----------
    product :  str
       Name of the SPEXone Level-1 product
    ref_date :  datetime.date()
       Date of the first detector image
    dims :  dict
       Dimensions of the datasets (differs for L1A, L1B, L1C)
    compression : bool, default=False
       Use compression on dataset /science_data/detector_images [L1A, only]

    Notes
    -----
    The engineering data should be extended, suggestions:
    * Temperatures of a.o. detector, FEE, optica, obm, telescope
    * Instrument settings: exposure time, dead time, frame time, co-adding, ...
    """
    product: Path
    processing_level = 'unknown'
    dset_stored = {}

    def __init__(self, product: str, ref_date: datetime.date,
                 dims: dict, compression=False):
        """Initialize access to a SPEXone Level-1 product.
        """
        self.product = Path(product)
        self.fid = None

        # initialize private class-attributes
        self.__epoch = ref_date

        # initialize Level-1 product
        if self.processing_level == 'L1A':
            self.fid = init_l1a(product, ref_date, dims, compression)
        elif self.processing_level == 'L1B':
            self.fid = init_l1b(product, ref_date, dims)
        elif self.processing_level == 'L1C':
            self.fid = init_l1c(product, ref_date, dims)
        else:
            raise KeyError('valid processing levels are: L1A, L1B or L1C')
        for key in self.dset_stored:
            self.dset_stored[key] = 0

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return f'{class_name}({self.product!r})'

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __enter__(self):
        """Method called to initiate the context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Method called when exiting the context manager.
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self) -> None:
        """Close all resources (currently a placeholder function).
        """
        if self.fid is None:
            return

        self.fid.close()
        self.fid = None

    # ---------- PUBLIC FUNCTIONS ----------
    @property
    def epoch(self) -> datetime:
        """Provide epoch for SPEXone.
        """
        return self.__epoch

    def get_dim(self, name: str):
        """Get size of a netCDF4 dimension.
        """
        return self.fid.dimensions[name].size

    # ----- ATTRIBUTES --------------------
    def get_attr(self, name: str, ds_name=None):
        """Read data of an attribute.

        Global or attached to a group or variable.

        Parameters
        ----------
        name : string
           name of the attribute
        ds_name : string, default=None
           name of dataset to which the attribute is attached

        Returns
        -------
        scalar or array_like
           value of attribute 'name', global or attached to dataset 'ds_name'
        """
        if ds_name is None:
            res = self.fid.getncattr(name)
        else:
            if ds_name not in self.fid.groups \
               and ds_name not in self.fid.variables:
                return None
            res = self.fid[ds_name].getncattr(name)

        if isinstance(res, bytes):
            return res.decode('ascii')

        return res

    def set_attr(self, name: str, value, ds_name=None) -> None:
        """Write data to an attribute.

        Global or attached to a group or variable.

        Parameters
        ----------
        name : string
           name of the attribute
        value : scalar, array_like
           value or values to be written
        ds_name : string, default=None
           name of group or dataset to which the attribute is attached
           **Use group name without starting '/'**
        """
        if ds_name is None:
            if isinstance(value, str):
                self.fid.setncattr(name, np.string_(value))
            else:
                self.fid.setncattr(name, value)
        else:
            grp_name = str(PurePosixPath(ds_name).parent)
            var_name = str(PurePosixPath(ds_name).name)
            if grp_name != '.':
                if var_name not in self.fid[grp_name].groups \
                   and var_name not in self.fid[grp_name].variables:
                    raise KeyError(f'ds_name {ds_name} not present in product')
            else:
                if var_name not in self.fid.groups \
                   and var_name not in self.fid.variables:
                    raise KeyError(f'ds_name {ds_name} not present in product')

            if isinstance(value, str):
                self.fid[ds_name].setncattr(name, np.string_(value))
            else:
                self.fid[ds_name].setncattr(name, value)

    # ----- VARIABLES --------------------
    def get_dset(self, name: str):
        """Read data of a netCDF4 variable.

        Parameters
        ----------
        name : string
           name of dataset

        Returns
        -------
        scalar or array_like
           value of dataset 'name'
        """
        grp_name = str(PurePosixPath(name).parent)
        var_name = str(PurePosixPath(name).name)
        if grp_name != '.':
            if var_name not in self.fid[grp_name].variables:
                raise KeyError(f'dataset {name} not present in Level-1 product')
        else:
            if var_name not in self.fid.variables:
                raise KeyError(f'dataset {name} not present in Level-1 product')

        return self.fid[name][:]

    def set_dset(self, name: str, value) -> None:
        """Write data to a netCDF4 variable.

        Parameters
        ----------
        name : string
           Name of Level-1 dataset
        value : scalar or array_like
           Value or values to be written
        """
        value = np.asarray(value)
        grp_name = str(PurePosixPath(name).parent)
        var_name = str(PurePosixPath(name).name)
        if grp_name != '.':
            if var_name not in self.fid[grp_name].variables:
                raise KeyError(f'dataset {name} not present in Level-1 product')
        else:
            if var_name not in self.fid.variables:
                raise KeyError(f'dataset {name} not present in Level-1 product')

        self.fid[name][...] = value
        self.dset_stored[name] += 1 if value.shape == () else value.shape[0]

    # -------------------------
    def fill_global_attrs(self, orbit=-1,
                          bin_size=None,
                          inflight=False) -> None:
        """Define global attributes in the SPEXone Level-1 products.

        Parameters
        ----------
        orbit :  int, default=-1
           Orbit revolution counter
        bin_size :  str, default=None
           Size of the nadir footprint (cross-track), include unit: e.g. '5km'
        inflight :  bool, default=False
           Measurements performed on-ground or inflight
        """
        dict_attrs = attrs_def(self.processing_level, inflight)
        dict_attrs['product_name'] = self.product.name
        dict_attrs['orbit_number'] = orbit
        if bin_size is not None:
            dict_attrs['bin_size_at_nadir'] = bin_size

        for key, value in dict_attrs.items():
            if value is not None:
                self.fid.setncattr(key, value)


# - class L1Aio -------------------------
class L1Aio(Lv1io):
    """
    This class can be used to create a SPEXone Level-1A product

    Parameters
    ----------
    product :  str
       Name of the SPEXone Level-1A product
    ref_date :  datetime.date()
       Date of the first detector image
    dims :  dict
       Dimensions of the datasets, default values::

          number_of_images : None     # number of image frames
          samples_per_image : 184000  # depends on binning table
          hk_packets : None           # number of HK tlm-packets
          wavelength : None

    compression : bool, default=False
       Use compression on dataset /science_data/detector_images
    """
    processing_level = 'L1A'
    dset_stored = {
        '/science_data/detector_images': 0,
        '/science_data/detector_telemetry': 0,
        '/image_attributes/binning_table': 0,
        '/image_attributes/digital_offset': 0,
        '/image_attributes/nr_coadditions': 0,
        '/image_attributes/exposure_time': 0,
        '/image_attributes/icu_time_sec': 0,
        '/image_attributes/icu_time_subsec': 0,
        '/image_attributes/image_time': 0,
        '/image_attributes/image_ID': 0,
        '/engineering_data/NomHK_telemetry': 0,
        # '/engineering_data/DemHK_telemetry': 0,
        '/engineering_data/temp_detector': 0,
        '/engineering_data/temp_housing': 0,
        '/engineering_data/temp_radiator': 0,
        '/engineering_data/HK_tlm_time': 0
    }

    def close(self):
        """Close product and check if required datasets are filled with data.
        """
        if self.fid is None:
            return

        # check if atleast one dataset is updated
        if self.fid.dimensions['number_of_images'].size == 0:
            self.fid.close()
            self.fid = None
            return

        # check of all required dataset their sizes
        self.check_stored(allow_empty=True)
        self.fid.close()
        self.fid = None

    # -------------------------
    def check_stored(self, allow_empty=False):
        """Check variables with the same first dimension have equal sizes.

        Parameters
        ----------
        allow_empty :  bool, default=False
        """
        warn_str = ('SPEX Level-1A format check [WARNING]:'
                    ' size of variable "{:s}" is wrong, only {:d} elements')

        # check image datasets
        dim_sz = self.get_dim('number_of_images')
        res = []
        key_list = [x for x in self.dset_stored
                    if (x.startswith('/science_data')
                        or x.startswith('/image_attributes'))]
        for key in key_list:
            res.append(self.dset_stored[key])
        res = np.array(res)
        if allow_empty:
            indx = ((res > 0) & (res != dim_sz)).nonzero()[0]
        else:
            indx = (res != dim_sz).nonzero()[0]
        for ii in indx:
            print(warn_str.format(key_list[ii], res[ii]))

        # check house-keeping datasets
        dim_sz = self.get_dim('hk_packets')
        key_list = [x for x in self.dset_stored
                    if x.startswith('/engineering_data')]
        res = []
        for key in key_list:
            res.append(self.dset_stored[key])
        res = np.array(res)
        if allow_empty:
            indx = ((res > 0) & (res != dim_sz)).nonzero()[0]
        else:
            indx = (res != dim_sz).nonzero()[0]
        for ii in indx:
            print(warn_str.format(key_list[ii], res[ii]))

    # ---------- PUBLIC FUNCTIONS ----------
    def fill_science(self, img_data, img_hk, img_id) -> None:
        """Write Science data and housekeeping telemetry (Science).

        Parameters
        ----------
        img_data : numpy array (uint16)
           Detector image data
        img_hk : numpy array ()
           Structured array with all Science telemetry parameters
        img_id : numpy array (uint16)
           Detector frame counter modulo 0x3FFF

        Notes
        -----
        Adds detector_telemetry data to the group /science_data

        Parameters: binning_table, digital_offset, exposure_time
        and nr_coadditions are extracted from the telemetry packets and writen
        in the group /image_attributes
        """
        if len(img_hk) == 0:
            return

        self.set_dset('/science_data/detector_images', img_data)
        self.set_dset('/science_data/detector_telemetry', img_hk)
        self.set_dset('/image_attributes/image_ID', img_id)

        tm_sc = TMscience(img_hk)
        self.set_dset('/image_attributes/binning_table', tm_sc.binning_table)
        self.set_dset('/image_attributes/digital_offset', tm_sc.digital_offset)
        self.set_dset('/image_attributes/exposure_time', tm_sc.exposure_time)
        self.set_dset('/image_attributes/nr_coadditions', tm_sc.nr_coadditions)

    def fill_nomhk(self, nomhk_data):
        """Write nominal house-keeping telemetry packets (NomHK).

        Parameters
        ----------
        nomhk_data : numpy array
           Structured array with all NomHK telemetry parameters

        Notes
        -----
        Writes nomhk_data as TM_telemetry in group /engineering_data

        Parameters: temp_detector and temp_housing are extracted and converted
        to Kelvin and writen to the group /engineering_data
        """
        if len(nomhk_data) == 0:
            return

        self.set_dset('/engineering_data/NomHK_telemetry', nomhk_data)

        if np.all(nomhk_data['TS1_DEM_N_T'] == 0):
            self.set_dset('/engineering_data/temp_detector',
                          np.full(nomhk_data.size, 273))
        else:
            self.set_dset('/engineering_data/temp_detector',
                          frac_poly(nomhk_data['TS1_DEM_N_T']))

        if np.all(nomhk_data['TS2_HOUSING_N_T'] == 0):
            self.set_dset('/engineering_data/temp_housing',
                          np.full(nomhk_data.size, 293))
        else:
            self.set_dset('/engineering_data/temp_housing',
                          frac_poly(nomhk_data['TS2_HOUSING_N_T']))

        if np.all(nomhk_data['TS3_RADIATOR_N_T'] == 0):
            self.set_dset('/engineering_data/temp_radiator',
                          np.full(nomhk_data.size, 294))
        else:
            self.set_dset('/engineering_data/temp_radiator',
                          frac_poly(nomhk_data['TS3_RADIATOR_N_T']))

    def fill_demhk(self, demhk_data):
        """Write detector housekeeping telemetry packets (DemHK).

        Parameters
        ----------
        demhk_data : numpy array
           Structured array with all DemHK telemetry parameters

        Notes
        -----
        Writes demhk_data as DetTM_telemetry in group /engineering_data

        Parameters: temp_detector and temp_housing are extracted and converted
        to Kelvin and writen to the group /engineering_data
        """
        if len(demhk_data) == 0:
            return

        self.set_dset('/engineering_data/DemHK_telemetry', demhk_data)


# - class L1Bio -------------------------
class L1Bio(Lv1io):
    """This class can be used to create a SPEXone Level-1B product

    Parameters
    ----------
    product :  str
       Name of the SPEXone Level-1B product
    ref_date :  datetime.date
       Date of the first detector image
    dims :  dict
       Dimensions of the datasets, default values::

          bins_along_track: 400
          spatial_samples_per_image: 200
          intensity_bands_per_view: 50
          polarization_bands_per_view: 50

    Notes
    -----
    ToDo: make sure we store the reference date for image_time
    """
    processing_level = 'L1B'
    dset_stored = {
        '/BIN_ATTRIBUTES/image_time': 0,
        '/GEOLOCATION_DATA/altitude': 0,
        '/GEOLOCATION_DATA/latitude': 0,
        '/GEOLOCATION_DATA/longitude': 0,
        '/GEOLOCATION_DATA/sensor_azimuth': 0,
        '/GEOLOCATION_DATA/sensor_zenith': 0,
        '/GEOLOCATION_DATA/solar_azimuth': 0,
        '/GEOLOCATION_DATA/solar_zenith': 0,
        '/OBSERVATION_DATA/I': 0,
        '/OBSERVATION_DATA/I_noise': 0,
        '/OBSERVATION_DATA/AoLP': 0,
        '/OBSERVATION_DATA/AoLP_noise': 0,
        '/OBSERVATION_DATA/DoLP': 0,
        '/OBSERVATION_DATA/DoLP_noise': 0,
        '/OBSERVATION_DATA/Q_over_I': 0,
        '/OBSERVATION_DATA/Q_over_I_noise': 0,
        '/OBSERVATION_DATA/U_over_I': 0,
        '/OBSERVATION_DATA/U_over_I_noise': 0,
        '/SENSOR_VIEWS_BANDS/viewport_index': 0,
        '/SENSOR_VIEWS_BANDS/intensity_wavelengths': 0,
        '/SENSOR_VIEWS_BANDS/intensity_bandpasses': 0,
        '/SENSOR_VIEWS_BANDS/intensity_F0': 0,
        '/SENSOR_VIEWS_BANDS/polarization_wavelengths': 0,
        '/SENSOR_VIEWS_BANDS/polarization_bandpasses': 0,
        '/SENSOR_VIEWS_BANDS/polarization_F0': 0,
        '/SENSOR_VIEWS_BANDS/view_angles': 0
    }

    def close(self):
        """Close product and check if required datasets are filled with data.
        """
        if self.fid is None:
            return

        # check if atleast one dataset is updated
        if self.fid.dimensions['bins_along_track'].size == 0:
            self.fid.close()
            self.fid = None
            return

        # check of all required dataset their sizes
        self.check_stored()

        # update coverage time
        secnd = self.fid['/BIN_ATTRIBUTES/image_time'][0].data
        time0 = (self.epoch
                 + timedelta(seconds=int(secnd))
                 + timedelta(microseconds=int(secnd % 1)))

        secnd = self.fid['/BIN_ATTRIBUTES/image_time'][-1].data
        time1 = (self.epoch
                 + timedelta(seconds=int(secnd))
                 + timedelta(microseconds=int(secnd % 1)))

        self.fid.time_coverage_start = time0.isoformat(timespec='milliseconds')
        self.fid.time_coverage_end = time1.isoformat(timespec='milliseconds')

        self.fid.close()
        self.fid = None

    # -------------------------
    def check_stored(self):
        """Check variables with the same first dimension have equal sizes.
        """
        warn_str = ('SPEX Level-1B format check [WARNING]:'
                    ' size of variable "{:s}" is wrong, only {:d} elements')

        # check datasets in group /SENSOR_VIEWS_BANDS
        dim_sz = self.get_dim('number_of_views')
        res = []
        key_list = [x for x in self.dset_stored
                    if x.startswith('/SENSOR_VIEWS_BANDS')]
        for key in key_list:
            if key == '/SENSOR_VIEWS_BANDS/viewport_index':
                continue
            res.append(self.dset_stored[key])
        res = np.array(res)
        indx = (res != dim_sz).nonzero()[0]
        for ii in indx:
            print(warn_str.format(key_list[ii], res[ii]))

        # check datasets in all other groups
        dim_sz = self.get_dim('bins_along_track')
        res = []
        key_list = [x for x in self.dset_stored
                    if not x.startswith('/SENSOR_VIEWS_BANDS')]
        for key in key_list:
            res.append(self.dset_stored[key])
        res = np.array(res)
        indx = (res != dim_sz).nonzero()[0]
        for ii in indx:
            print(warn_str.format(key_list[ii], res[ii]))

        for ii, key in enumerate(self.dset_stored):
            print(ii, key, self.dset_stored[key])

    # ---------- PUBLIC FUNCTIONS ----------


# - class L1Cio -------------------------
class L1Cio(Lv1io):
    """This class can be used to create a SPEXone Level-1C product.

    Parameters
    ----------
    product :  str
       Name of the SPEXone Level-1B product
    ref_date :  datetime.date
       Date of the first detector image
    dims :  dict
       Dimensions of the datasets, default values::

          bins_along_track: 400
          spatial_samples_per_image: 200
          intensity_bands_per_view: 50
          polarization_bands_per_view: 50

    Notes
    -----
    ToDo: make sure we store the reference date for image_time
    """
    processing_level = 'L1C'
    dset_stored = {
        '/BIN_ATTRIBUTES/nadir_view_time': 0,
        '/BIN_ATTRIBUTES/view_time_offsets': 0,
        '/GEOLOCATION_DATA/latitude': 0,
        '/GEOLOCATION_DATA/longitude': 0,
        '/GEOLOCATION_DATA/altitude': 0,
        '/GEOLOCATION_DATA/altitude_variability': 0,
        '/GEOLOCATION_DATA/sensor_azimuth': 0,
        '/GEOLOCATION_DATA/sensor_zenith': 0,
        '/GEOLOCATION_DATA/solar_azimuth': 0,
        '/GEOLOCATION_DATA/solar_zenith': 0,
        '/OBSERVATION_DATA/obs_per_view': 0,
        '/OBSERVATION_DATA/AoLP': 0,
        '/OBSERVATION_DATA/AoLP_noise': 0,
        '/OBSERVATION_DATA/DoLP': 0,
        '/OBSERVATION_DATA/DoLP_noise': 0,
        '/OBSERVATION_DATA/I': 0,
        '/OBSERVATION_DATA/I_noise': 0,
        '/OBSERVATION_DATA/I_polsample': 0,
        '/OBSERVATION_DATA/I_polsample_noise': 0,
        '/OBSERVATION_DATA/QC': 0,
        '/OBSERVATION_DATA/QC_bitwise': 0,
        '/OBSERVATION_DATA/QC_polsample': 0,
        '/OBSERVATION_DATA/QC_polsample_bitwise': 0,
        '/OBSERVATION_DATA/Q_over_I': 0,
        '/OBSERVATION_DATA/Q_over_I_noise': 0,
        '/OBSERVATION_DATA/U_over_I': 0,
        '/OBSERVATION_DATA/U_over_I_noise': 0,
        '/SENSOR_VIEWS_BANDS/intensity_bandpasses': 0,
        '/SENSOR_VIEWS_BANDS/intensity_wavelengths': 0,
        '/SENSOR_VIEWS_BANDS/intensity_F0': 0,
        '/SENSOR_VIEWS_BANDS/polarization_bandpasses': 0,
        '/SENSOR_VIEWS_BANDS/polarization_wavelengths': 0,
        '/SENSOR_VIEWS_BANDS/polarization_F0': 0,
        '/SENSOR_VIEWS_BANDS/view_angles': 0
    }

    def close(self):
        """Close product and check if required datasets are filled with data.
        """
        if self.fid is None:
            return

        # check if atleast one dataset is updated
        if self.fid.dimensions['bins_along_track'].size == 0:
            self.fid.close()
            self.fid = None
            return

        # check of all required dataset their sizes
        self.check_stored()

        # update coverage time
        secnd = self.fid['/BIN_ATTRIBUTES/nadir_view_time'][0].data
        time0 = (self.epoch
                 + timedelta(seconds=int(secnd))
                 + timedelta(microseconds=int(secnd % 1)))

        secnd = self.fid['/BIN_ATTRIBUTES/nadir_view_time'][-1].data
        time1 = (self.epoch
                 + timedelta(seconds=int(secnd))
                 + timedelta(microseconds=int(secnd % 1)))

        self.fid.time_coverage_start = time0.isoformat(timespec='milliseconds')
        self.fid.time_coverage_end = time1.isoformat(timespec='milliseconds')

        self.fid.close()
        self.fid = None

    # -------------------------
    def check_stored(self):
        """Check variables with the same first dimension have equal sizes.
        """
        warn_str = ('SPEX Level-1C format check [WARNING]:'
                    ' size of variable "{:s}" is wrong, only {:d} elements')

        # check datasets in group /SENSOR_VIEWS_BANDS
        dim_sz = self.get_dim('number_of_views')
        res = []
        key_list = [x for x in self.dset_stored
                    if x.startswith('/SENSOR_VIEWS_BANDS')]
        for key in key_list:
            if key == '/SENSOR_VIEWS_BANDS/viewport_index':
                continue
            res.append(self.dset_stored[key])
        res = np.array(res)
        indx = (res != dim_sz).nonzero()[0]
        for ii in indx:
            print(warn_str.format(key_list[ii], res[ii]))

        # check datasets in all other groups
        dim_sz = self.get_dim('bins_along_track')
        res = []
        key_list = [x for x in self.dset_stored
                    if not x.startswith('/SENSOR_VIEWS_BANDS')]
        for key in key_list:
            res.append(self.dset_stored[key])
        res = np.array(res)
        indx = (res != dim_sz).nonzero()[0]
        for ii in indx:
            print(warn_str.format(key_list[ii], res[ii]))

        for ii, key in enumerate(self.dset_stored):
            print(ii, key, self.dset_stored[key])

    # ---------- PUBLIC FUNCTIONS ----------
