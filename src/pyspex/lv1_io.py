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
Contains the class `L1Aio` to write PACE/SPEXone data in Level-1A format.
"""
from __future__ import annotations

__all__ = ['L1Aio', 'get_l1a_name']

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np

from .lib.attrs_def import attrs_def
from .lib.l1a_def import init_l1a
from .lib.tlm_utils import convert_hk
from .version import pyspex_version

# - global parameters -------------------
MCP_TO_SEC = 1e-7
ONE_DAY = 24 * 60 * 60


# - local functions ---------------------
def _binning_table_(img_hk: np.ndarray) -> np.ndarray:
    """Return binning table identifier (zero for full-frame images).
    """
    if 'REG_FULL_FRAME' not in img_hk.dtype.names:
        print('[WARNING]: can not determine binning table identifier')
        return np.full(len(img_hk), -1, dtype='i1')

    full_frame = np.unique(img_hk['REG_FULL_FRAME'])
    if len(full_frame) > 1:
        print('[WARNING]: value of REG_FULL_FRAME not unique')
    full_frame = img_hk['REG_FULL_FRAME'][-1]

    cmv_outputmode = np.unique(img_hk['REG_CMV_OUTPUTMODE'])
    if len(cmv_outputmode) > 1:
        print('[WARNING]: value of REG_CMV_OUTPUTMODE not unique')
    cmv_outputmode = img_hk['REG_CMV_OUTPUTMODE'][-1]

    if full_frame == 1:
        if cmv_outputmode != 3:
            raise KeyError('Diagnostic mode with REG_CMV_OUTPMODE != 3')
        return np.zeros(len(img_hk), dtype='i1')

    if full_frame == 2:
        if cmv_outputmode != 1:
            raise KeyError('Science mode with REG_CMV_OUTPUTMODE != 1')
        bin_tbl_start = img_hk['REG_BINNING_TABLE_START']
        indx0 = (img_hk['REG_FULL_FRAME'] != 2).nonzero()[0]
        if indx0.size > 0:
            indx2 = (img_hk['REG_FULL_FRAME'] == 2).nonzero()[0]
            bin_tbl_start[indx0] = bin_tbl_start[indx2[0]]
        res = 1 + (bin_tbl_start - 0x80000000) // 0x400000
        return res & 0xFF

    raise KeyError('REG_FULL_FRAME not equal to 1 or 2')


def _digital_offset_(img_hk: np.ndarray) -> np.ndarray:
    """Returns digital offset including ADC offset [count].
    """
    buff = img_hk['DET_OFFSET'].astype('i4')
    if np.isscalar(buff):
        if buff >= 8192:
            buff -= 16384
    else:
        buff[buff >= 8192] -= 16384

    return buff + 70


def _exposure_time_(img_hk: np.ndarray) -> np.ndarray:
    """Returns exposure time in seconds [float].
    """
    # need first bit of address 121
    reg_pgagainfactor = img_hk['DET_BLACKCOL'] & 0x1
    reg_pgagain = img_hk['DET_PGAGAIN']
    exp_time = (1 + 0.2 * reg_pgagain) * 2 ** reg_pgagainfactor

    return MCP_TO_SEC * exp_time


def _nr_coadditions_(img_hk: np.ndarray) -> np.ndarray:
    """Returns number of coadditions.
    """
    return img_hk['REG_NCOADDFRAMES']


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
       Vnn file-version number (omitted when nn=1)
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


# - class L1Aio -------------------------
class L1Aio:
    """Class to create SPEXone Level-1A products.

    Parameters
    ----------
    product :  str
       Name of the SPEXone Level-1 product
    ref_date :  datetime.datetime
       Date of the first detector image
    dims :  dict
       Dimensions of the datasets, default values::

          number_of_images : None     # number of image frames
          samples_per_image : 184000  # depends on binning table
          hk_packets : None           # number of HK tlm-packets

    compression : bool, default=False
       Use compression on dataset /science_data/detector_images
    """
    product: Path
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

    def __init__(self, product: str, ref_date: datetime.datetime,
                 dims: dict, compression: bool = False):
        """Initialize access to a SPEXone Level-1 product.
        """
        self.product = Path(product)
        self.fid = None

        # initialize private class-attributes
        self.__epoch = ref_date

        # initialize Level-1 product
        if self.processing_level == 'L1A':
            self.fid = init_l1a(product, ref_date, dims, compression)
        else:
            raise KeyError('valid processing levels are: L1A')
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

    def close(self):
        """Close product and check if required datasets are filled with data.
        """
        if self.fid is None:
            return

        # check if at least one dataset is updated
        if self.fid.dimensions['number_of_images'].size == 0:
            self.fid.close()
            self.fid = None
            return

        # check of all required dataset their sizes
        self.check_stored(allow_empty=True)
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

    # - L1A specific functions ------------------------
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
        self.set_dset('/image_attributes/binning_table',
                      _binning_table_(img_hk))
        self.set_dset('/image_attributes/digital_offset',
                      _digital_offset_(img_hk))
        self.set_dset('/image_attributes/exposure_time',
                      _exposure_time_(img_hk))
        self.set_dset('/image_attributes/nr_coadditions',
                      _nr_coadditions_(img_hk))

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
                          convert_hk('TS1_DEM_N_T',
                                     nomhk_data['TS1_DEM_N_T']))

        if np.all(nomhk_data['TS2_HOUSING_N_T'] == 0):
            self.set_dset('/engineering_data/temp_housing',
                          np.full(nomhk_data.size, 293))
        else:
            self.set_dset('/engineering_data/temp_housing',
                          convert_hk('TS2_HOUSING_N_T',
                                     nomhk_data['TS2_HOUSING_N_T']))

        if np.all(nomhk_data['TS3_RADIATOR_N_T'] == 0):
            self.set_dset('/engineering_data/temp_radiator',
                          np.full(nomhk_data.size, 294))
        else:
            self.set_dset('/engineering_data/temp_radiator',
                          convert_hk('TS3_RADIATOR_N_T',
                                     nomhk_data['TS3_RADIATOR_N_T']))
