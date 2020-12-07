"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python class to create SPEXone Level-1 products

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

import numpy as np

from netCDF4 import Dataset

from .lib.attrs_def import attrs_def
from .lib.l1a_def import init_l1a
from .lib.l1b_def import init_l1b
from .lib.l1c_def import init_l1c

# - global parameters -------------------
MCP_TO_SEC = 1e-7


# - class LV1mps -------------------------
class LV1mps:
    """
    Class to convert raw register settings from the MPS

    Methods
    -------
    get(key)
       Return (raw) MPS parameter.
    number_channels
       Return number of LVDS channels used.
    lvds_clock
       Returns flag for LVDS clock: False: disabled & True: enabled.
    offset
       Returns digital offset including ADC offset [counts].
    pga_gain
       Returns PGA gain [Volt].
    exp_time
       Returns pixel exposure time [master clock periods].
    fot_time
       Returns frame overhead time [master clock periods].
    rot_time
       Returns image read-out time [master clock periods]
    frame_period
       Returns frame period [master clock periods].
    pll_control
       Returns raw PLL control parameters: pll_range, pll_out_fre, pll_div.
    exp_control
       Returns raw exposure time parameters: inte_sync, exp_dual, exp_ext.
    """
    def __init__(self, mps_data):
        """
        Initialize class L1A_mps

        Parameters
        mps_data :  ndarray
        """
        self.__mps = mps_data

    def get(self, key: str):
        """
        Return (raw) MPS parameter
        """
        return self.__mps[key] if key in self.__mps.dtype.names else None

    @property
    def number_channels(self) -> int:
        """
        Return number of LVDS channels used
        """
        return 2 ** (4 - (self.__mps['DET_OUTMODE'] & 0x3))

    @property
    def lvds_clock(self) -> bool:
        """
        Returns flag for LVDS clock: False: disabled & True: enabled
        """
        return ((self.__mps['DET_PLLENA'] & 0x3) == 0
                and (self.__mps['DET_PLLBYP'] & 0x3) != 0
                and (self.__mps['DET_CHENA'] & 0x40000) != 0)

    @property
    def offset(self) -> int:
        """
        Returns digital offset including ADC offset
        """
        buff = self.__mps['DET_OFFSET'].astype('i4')
        if np.isscalar(buff):
            if buff >= 8192:
                buff -= 16384
        else:
            buff[buff >= 8192] -= 16384

        return buff + 70

    @property
    def pga_gain(self) -> float:
        """
        Returns PGA gain [Volt]
        """
        # need first bit of address 121
        reg_pgagainfactor = self.__mps['DET_BLACKCOL'] & 0x1

        reg_pgagain = self.__mps['DET_PGAGAIN']

        return (1 + 0.2 * reg_pgagain) * 2 ** reg_pgagainfactor

    @property
    def exp_time(self) -> float:
        """
        Returns pixel exposure time [master clock periods]
        """
        return 129 * (0.43 * self.__mps['DET_FOTLEN']
                      + self.__mps['DET_EXPTIME'])

    @property
    def fot_time(self) -> int:
        """
        Returns frame overhead time [master clock periods]
        """
        return 129 * (self.__mps['DET_FOTLEN']
                      + 2 * (16 // self.number_channels))

    @property
    def rot_time(self) -> int:
        """
        Returns image read-out time [master clock periods]
        """
        return 129 * (16 // self.number_channels) * self.__mps['DET_NUMLINES']

    @property
    def frame_period(self) -> float:
        """
        Returns frame period [master clock periods]
        """
        return 2.38e-7 + (self.__mps['REG_NCOADDFRAMES']
                          * (self.exp_time + self.fot_time + self.rot_time))

    @property
    def pll_control(self) -> tuple:
        """
        Returns raw PLL control parameters: pll_range, pll_out_fre, pll_div

        Notes
        -----
        PLL_range:    bits [7], valid values: 0 or 1
        PLL_out_fre:  bits [4:7], valid values:  0, 1, 2 or 5
        PLL_div:      bits [0:3], valid values 9 (10-bit) or 11 (12-bit)

        Other PLL registers are: PLL_enable, PLL_in_fre, PLL_bypass, PLL_load
        """
        pll_div = self.__mps['DET_PLLRATE'] & 0xF             # bit [0:4]
        pll_out_fre = (self.__mps['DET_PLLRATE'] >> 4) & 0x7  # bit [4:7]
        pll_range = (self.__mps['DET_PLLRATE'] >> 7)          # bit [7]

        return (pll_range, pll_out_fre, pll_div)

    @property
    def exp_control(self) -> tuple:
        """
        Returns raw exposure time parameters: inte_sync, exp_dual, exp_ext
        """
        inte_sync = (self.__mps['INTE_SYNC'] >> 2) & 0x1
        exp_dual = (self.__mps['INTE_SYNC'] >> 1) & 0x1
        exp_ext = self.__mps['INTE_SYNC'] & 0x1

        return (inte_sync, exp_dual, exp_ext)


# - class LV1io -------------------------
class Lv1io:
    """
    Generic class to create SPEXone Level-1 products

    Attributes
    ----------
    product: pathlib.Path object
       Concrete path object to SPEXone Level-1 product
    inflight: boolean
       Flag to indicate data collected during in-flight of on-ground
    fid: netCDF5.Dataset object
       NetCDF4 Pointer to SPEXone Level-1 product
    dset_stored: dict
       Number of items stored for all required netCDF4 variables

    Methods
    -------
    close()
       Close all resources (currently a placeholder function).
    epoch
       Provide epoch for SPEXone.
    get_dim(ds_name)
       Returns size of dimension.
    get_attr(attr_name, ds_name=None)
       Read data of an attribute, global or attached to a group or variable.
    set_attr(attr_name, value, ds_name=None)
       Write data to an attribute, global or attached to a group or variable.
    get_dset(ds_name)
       Read data of a netCDF4 variable.
    set_dset(ds_name, value, ibgn=-1)
       Write/append data to a netCDF4 variable.
    sec_of_day(ccsds_sec, ccsds_usec)
       Convert CCSDS timestamp to seconds after midnight.
    fill_global_attrs(level, orbit=-1, bin_size=None)
       Define global attributes in the SPEXone Level-1 products.

    Notes
    -----
    The engineering data should be extended, suggestions:
    * temperatures of a.o. detector, FEE, optica, obm, telescope
    * instrument settings: exposure time, dead time, frame time, coadding, ...
    """
    processing_level = 'unknown'
    dset_stored = {}

    def __init__(self, product: str, append=False, **kwargs):
        """
        Initialize access to a SPEXone Level-1 product

        Parameters
        ----------
        product : str
           name of the SPEXone Level-1 product
        append : bool
           do no clobber, but add new data to existing product
        """
        self.__epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)

        # initialize private class-attributes
        self.product = Path(product)
        self.inflight = True
        if 'inflight' in kwargs:
            self.inflight = kwargs['inflight']
        self.fid = None

        # initialize Level-1 product
        if not append:
            if self.processing_level == 'L1A':
                self.fid = init_l1a(product, **kwargs)
            elif self.processing_level == 'L1B':
                self.fid = init_l1b(product, **kwargs)
            elif self.processing_level == 'L1C':
                self.fid = init_l1c(product, **kwargs)
            else:
                raise KeyError('valid processing levels are: L1A, L1B or L1C')
        else:
            # open Level-1 product in append mode
            self.fid = Dataset(self.product, "r+")

            # store current length of the first dimension
            for key in self.dset_stored:
                self.dset_stored[key] = self.fid[key].shape[0]

    def __repr__(self) -> str:
        class_name = type(self).__name__
        return '{}({!r})'.format(class_name, self.product)

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __enter__(self):
        """
        method called to initiate the context manager
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """
        method called when exiting the context manager
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self) -> None:
        """
        Close all resources (currently a placeholder function)
        """
        if self.fid is None:
            return

        self.fid.close()
        self.fid = None

    # ---------- PUBLIC FUNCTIONS ----------
    @property
    def epoch(self) -> datetime:
        """
        Provide epoch for SPEXone
        """
        return self.__epoch

    def get_dim(self, name: str):
        """
        Get size of a netCDF4 dimension
        """
        return self.fid.dimensions[name].size

    # ----- ATTRIBUTES --------------------
    def get_attr(self, name: str, ds_name=None):
        """
        Read data of an attribute, global or attached to a group or variable

        Parameters
        ----------
        name : string
           name of the attribute
        ds_name : string, optional
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
        """
        Write data to an attribute, global or attached to a group or variable

        Parameters
        ----------
        name : string
           name of the attribute
        value : scalar, array_like
           value or values to be written
        ds_name : string
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
                    raise KeyError(
                        'ds_name {} not present in product'.format(ds_name))
            else:
                if var_name not in self.fid.groups \
                   and var_name not in self.fid.variables:
                    raise KeyError(
                        'ds_name {} not present in product'.format(ds_name))

            if isinstance(value, str):
                self.fid[ds_name].setncattr(name, np.string_(value))
            else:
                self.fid[ds_name].setncattr(name, value)

    # ----- VARIABLES --------------------
    def get_dset(self, name: str):
        """
        Read data of a netCDF4 variable

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
                raise KeyError(
                    'dataset {} not present in Level-1 product'.format(name))
        else:
            if var_name not in self.fid.variables:
                raise KeyError(
                    'dataset {} not present in Level-1 product'.format(name))

        return self.fid[name][:]

    def set_dset(self, name: str, value, ibgn=-1) -> None:
        """
        Write/append data to a netCDF4 variable

        Parameters
        ----------
        name : string
           Name of Level-1 dataset
        value : scalar, array_like
           Value or values to be written
        ibgn : integer
           Index of the first (unlimited) dimension where to store the new data
           Default is to append the data
        """
        value = np.asarray(value)
        grp_name = str(PurePosixPath(name).parent)
        var_name = str(PurePosixPath(name).name)
        if grp_name != '.':
            if var_name not in self.fid[grp_name].variables:
                raise KeyError(
                    'dataset {} not present in Level-1 product'.format(name))
        else:
            if var_name not in self.fid.variables:
                raise KeyError(
                    'dataset {} not present in Level-1 product'.format(name))

        dims = self.fid[name].get_dims()
        if not dims:
            self.fid[name][...] = value
        elif dims[0].isunlimited():
            if ibgn < 0:
                ibgn = self.dset_stored[name]
            self.fid[name][ibgn:, ...] = value
        else:
            self.fid[name][...] = value
        self.dset_stored[name] += value.shape[0]

    # -------------------------
    def sec_of_day(self, ccsds_sec, ccsds_usec):
        """
        Convert CCSDS timestamp to seconds after midnight

        Parameters
        ----------
        ccsds_sec : numpy array (dtype='u4')
          seconds since 1970-01-01
        ccsds_usec : numpy array (dtype='u2')
          microseconds seconds

        Returns
        -------
        numpy.ndarray with sec_of_day
        """
        # determine midnight before start measurement
        if 'reference_day' in self.fid.ncattrs():
            reference_day = datetime.fromisoformat(self.fid.reference_day)
        else:
            tstamp0 = self.epoch + timedelta(seconds=int(ccsds_sec[0]))
            reference_day = datetime(year=tstamp0.year,
                                     month=tstamp0.month,
                                     day=tstamp0.day, tzinfo=timezone.utc)
            self.fid.reference_day = reference_day.isoformat()

        # store seconds since midnight
        sec_of_day = ccsds_sec - (reference_day - self.epoch).total_seconds()

        # return seconds since midnight
        return sec_of_day + ccsds_usec / 2**16

    # -------------------------
    def fill_global_attrs(self, orbit=-1, bin_size=None) -> None:
        """
        Define global attributes in the SPEXone Level-1 products

        Parameters
        ----------
        orbit_number: int
           Orbit revolution counter, default=-1
        bin_size: str, optional
           Size of the nadir footprint (cross-track), include unit: e.g. '5km'
        """
        dict_attrs = attrs_def(self.processing_level, self.inflight)
        dict_attrs['product_name'] = self.product.name
        dict_attrs['orbit_number'] = orbit
        if bin_size is not None:
            dict_attrs['bin_size_at_nadir'] = bin_size

        for key in dict_attrs:
            if dict_attrs[key] is not None:
                self.fid.setncattr(key, dict_attrs[key])


# - class L1Aio -------------------------
class L1Aio(Lv1io):
    """
    This class can be used to create a SPEXone Level-1A product

    Parameters
    ----------
    lv1_product: string
       Name of the Level-1A product
    append : boolean, optional
       Open file in append mode, parameter dims and inflight are ignored
       Default: False
    dims: dictionary, optional
       Provide size of various dimensions (L1A only).
       Default values:
            number_of_images : None     # number of image frames
            samples_per_image : 184000  # depends on binning table
            SC_records : None           # space-craft navigation records
            hk_packets : None           # number of HK tlm-packets
            wavelength : None
    inflight: boolean, optional
       In-flight data, only affects global attributes of L1A product.
       Default: False

    Attributes
    ----------
    product: pathlib.Path object
       Concrete path object to SPEXone Level-1 product
    inflight: boolean
       Flag to indicate data collected during in-flight of on-ground
    fid: netCDF5.Dataset object
       NetCDF4 Pointer to SPEXone Level-1 product
    dset_stored: dict
       Number of items stored for all required netCDF4 variables

    Methods
    -------
    close()
       Close product and check if required datasets are filled with data.
    epoch
       Provide epoch for SPEXone.
    get_dim(ds_name)
       Returns size of dimension.
    get_attr(attr_name, ds_name=None)
       Read data of an attribute, global or attached to a group or variable.
    set_attr(attr_name, value, ds_name=None)
       Write data to an attribute, global or attached to a group or variable.
    get_dset(ds_name)
       Read data of a netCDF4 variable.
    set_dset(ds_name, valu, ibgn=-1e)
       Write/append data to a netCDF4 variable.
    sec_of_day(ccsds_sec, ccsds_usec)
       Convert CCSDS timestamp to seconds after midnight.
    fill_global_attrs(level, orbit=-1, bin_size=None)
       Define global attributes in the SPEXone Level-1 products.
    check_stored(allow_empty=False)
       Check variables with the same first dimension have equal sizes.
    fill_time(ccsds_sec, ccsds_usec, group=None)
       Write time of Science telemetry packets (UTC/TAI) to L1A product.
    fill_mps(mps_data)
       Write Science telemetry packets (MPS) to L1A product.
    fill_nomhk(nomhk_data)
       Write nominal house-keeping telemetry packets (NomHK) to L1A product.
    fill_gse(reference=None)
       Write EGSE/OGSE data to L1A product.
    """
    processing_level = 'L1A'
    dset_stored = {
        '/science_data/detector_images': 0,
        '/science_data/detector_telemetry': 0,
        '/image_attributes/binning_table': 0,
        '/image_attributes/digital_offset': 0,
        '/image_attributes/nr_coadditions': 0,
        '/image_attributes/exposure_time': 0,
        '/image_attributes/image_CCSDS_sec': 0,
        '/image_attributes/image_CCSDS_usec': 0,
        '/image_attributes/image_time': 0,
        '/image_attributes/image_ID': 0,
        '/engineering_data/HK_telemetry': 0,
        '/engineering_data/temp_detector': 0,
        '/engineering_data/temp_housing': 0,
        '/engineering_data/HK_tlm_time': 0,
        '/navigation_data/adstate': 0,
        '/navigation_data/att_quat': 0,
        '/navigation_data/orb_pos': 0,
        '/navigation_data/orb_vel': 0,
        '/navigation_data/att_time': 0,
        '/navigation_data/orb_time': 0
    }

    def close(self):
        """
        Close product and check if required datasets are filled with data
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

        # update coverage time
        mps = LV1mps(self.get_dset('/science_data/detector_telemetry')[-1])

        # determine duration master clock cycle
        imro =  1e-1 * mps.get('FTI') * 2
        mcycl = 1e-1 * mps.get('FTI') * mps.get('REG_NCOADDFRAMES')

        img_sec = self.fid['/image_attributes/image_CCSDS_sec'][:].data
        img_usec = self.fid['/image_attributes/image_CCSDS_usec'][:].data

        time0 = (self.epoch
                 + timedelta(seconds=int(img_sec[0]))
                 + timedelta(microseconds=int(img_usec[0]))
                 - timedelta(milliseconds=mcycl + imro))

        time1 = (self.epoch
                 + timedelta(seconds=int(img_sec[-1]))
                 + timedelta(microseconds=int(img_usec[-1]))
                 - timedelta(milliseconds=imro))

        self.fid.time_coverage_start = time0.isoformat(timespec='milliseconds')
        self.fid.time_coverage_end = time1.isoformat(timespec='milliseconds')

        self.fid.close()
        self.fid = None

    # -------------------------
    def check_stored(self, allow_empty=False):
        """
        Check variables with the same first dimension have equal sizes
        """
        warn_str = ('SPEX Level-1 format check [WARNING]:'
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

        # check navigation datasets
        dim_sz = self.get_dim('SC_records')
        key_list = [x for x in self.dset_stored
                    if x.startswith('/navigation_data')]
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
    def fill_time(self, ccsds_sec, ccsds_usec, group=None) -> None:
        """
        Write time of Science telemetry packets (UTC/TAI) to L1A product

        Parameters
        ----------
        ccsds_sec : numpy array (dtype='u4')
          seconds since 1970-01-01
        ccsds_usec : numpy array (dtype='u2')
          microseconds seconds

        Note
        ----
        Writes parameters: image_CCSDS_sec, image_CCSDS_usec and image_time
        """
        if group is None:
            group = 'image_attributes'

        # calculate seconds of day
        sec_of_day = self.sec_of_day(ccsds_sec, ccsds_usec)

        if group in ('image_attributes', '/image_attributes'):
            self.set_dset('/image_attributes/image_CCSDS_sec', ccsds_sec)
            self.set_dset('/image_attributes/image_CCSDS_usec', ccsds_usec)
            self.set_dset('/image_attributes/image_time', sec_of_day)
        elif group in ('engineering_data', '/engineering_data'):
            self.set_dset('/engineering_data/HK_tlm_time', sec_of_day)

    def fill_mps(self, mps_data) -> None:
        """
        Write Science telemetry packets (MPS) to L1A product

        Parameters
        ----------
        mps_data : numpy array
           Structured array with all Science telemetry parameters

        Notes
        -----
        Writes mps_data as detector_telemetry in the group /science_data

        Parameters: binning_table, digital_offset, exposure_time
        and nr_coadding are extracted from the telemetry packets and writen
        in the group /image_attributes
        """
        if len(mps_data) == 0:
            return

        self.set_dset('/science_data/detector_telemetry', mps_data)

        mps = LV1mps(mps_data)
        self.set_dset('/image_attributes/binning_table',
                      mps.get('REG_BINNING_TABLE'))
        self.set_dset('/image_attributes/digital_offset',
                      mps.offset)
        self.set_dset('/image_attributes/exposure_time',
                      MCP_TO_SEC * mps.exp_time)
        self.set_dset('/image_attributes/nr_coadditions',
                      mps.get('REG_NCOADDFRAMES'))

    def fill_nomhk(self, nomhk_data):
        """
        Write nominal house-keeping telemetry packets (NomHK) to L1A product

        Parameters
        ----------
        nomhk_data : numpy array
           Structured array with all NomHK telemetry parameters

        Notes
        -----
        Writes nomhk_data as TM_telemetry in group /engineering_data

        Parameters: temp_detector and temp_housing are extracted and converted
        to degrees Celsius and writen to the group /engineering_data
        """
        if len(nomhk_data) == 0:
            return

        self.set_dset('/engineering_data/HK_telemetry', nomhk_data)

        if np.all(nomhk_data['TS1_DEM_N_T'] == 0):
            self.set_dset('/engineering_data/temp_detector',
                          np.full(nomhk_data.size, 273))
        else:
            self.set_dset('/engineering_data/temp_detector',
                          nomhk_data['TS1_DEM_N_T'])

        if np.all(nomhk_data['TS2_HOUSING_N_T'] == 0):
            self.set_dset('/engineering_data/temp_housing',
                          np.full(nomhk_data.size, 293))
        else:
            self.set_dset('/engineering_data/temp_housing',
                          nomhk_data['TS2_HOUSING_N_T'])

    def fill_gse(self, reference=None) -> None:
        """
        Write EGSE/OGSE data to L1A product

        Parameters
        ----------
        reference : dict, optional
           biweight value and spread of the signal measured during the
           measurement by a reference detector.
           Expected dictionary keys: 'value', 'error'
        """
        if reference is not None:
            dset = self.fid.createVariable('/gse_data/reference_signal',
                                           'f8', ())
            dset.long_name = "biweight median of reference-detector signal"
            dset.comment = "t_sat = min(2.28e-9 / S_reference, 30)"
            dset.units = 'A'
            dset[:] = reference['value']
            self.set_attr('Illumination_level',
                          reference['value'] * 5e9 / 1.602176634,
                          ds_name='gse_data')

            dset = self.fid.createVariable('/gse_data/reference_error',
                                           'f8', ())
            dset.long_name = "biweight spread of reference-detector signal"
            dset.units = 'A'
            dset[:] = reference['error']


# - class L1Bio -------------------------
class L1Bio(Lv1io):
    """
    This class can be used to create a SPEXone Level-1B product

    Parameters
    ----------
    lv1_product: string
       Name of the Level-1B product
    append : boolean, optional
       Open file in append mode, parameter dims and inflight are ignored
       Default: False
    number_of_images: int
       Number of images used as input to generate the L1B product.
       Default is None, then this dimension is UNLIMITED.
    spatial_samples: int
       Total number of spatial samples from all viewports, default is 200

    Attributes
    ----------
    product: pathlib.Path object
       Concrete path object to SPEXone Level-1 product
    inflight: boolean
       Flag to indicate data collected during in-flight of on-ground
    fid: netCDF5.Dataset object
       NetCDF4 Pointer to SPEXone Level-1 product
    dset_stored: dict
       Number of items stored for all required netCDF4 variables

    Methods
    -------
    close()
       Close product and check if required datasets are filled with data.
    epoch
       Provide epoch for SPEXone.
    get_dim(ds_name)
       Returns size of dimension.
    get_attr(attr_name, ds_name=None)
       Read data of an attribute, global or attached to a group or variable.
    set_attr(attr_name, value, ds_name=None)
       Write data to an attribute, global or attached to a group or variable.
    get_dset(ds_name)
       Read data of a netCDF4 variable.
    set_dset(ds_name, value, ibgn=-1)
       Write/append data to a netCDF4 variable.
    sec_of_day(ccsds_sec, ccsds_usec)
       Convert CCSDS timestamp to seconds after midnight.
    fill_global_attrs(level, orbit=-1, bin_size=None)
       Define global attributes in the SPEXone Level-1 products.
    check_stored()
       Check variables with the same first dimension have equal sizes.

    Notes
    -----
    ToDo: make sure we store the reference date for image_time
    """
    processing_level = 'L1B'
    dset_stored = {
        '/SENSOR_VIEWS_BANDS/viewport_index': 0,
        '/SENSOR_VIEWS_BANDS/view_angles': 0,
        '/SENSOR_VIEWS_BANDS/intensity_wavelengths': 0,
        '/SENSOR_VIEWS_BANDS/intensity_bandpasses': 0,
        '/SENSOR_VIEWS_BANDS/polarization_wavelengths': 0,
        '/SENSOR_VIEWS_BANDS/polarization_bandpasses': 0,
        '/SENSOR_VIEWS_BANDS/intensity_f0': 0,
        '/SENSOR_VIEWS_BANDS/polarization_f0': 0,
        '/BIN_ATTRIBUTES/image_time': 0,
        '/GEOLOCATION_DATA/latitude': 0,
        '/GEOLOCATION_DATA/longitude': 0,
        '/GEOLOCATION_DATA/altitude': 0,
        '/GEOLOCATION_DATA/altitude_variability': 0,
        '/GEOLOCATION_DATA/sensor_azimuth': 0,
        '/GEOLOCATION_DATA/sensor_zenith': 0,
        '/GEOLOCATION_DATA/solar_azimuth': 0,
        '/GEOLOCATION_DATA/solar_zenith': 0,
        '/OBSERVATION_DATA/I': 0,
        '/OBSERVATION_DATA/I_noise': 0,
        '/OBSERVATION_DATA/q': 0,
        '/OBSERVATION_DATA/q_noise': 0,
        '/OBSERVATION_DATA/u': 0,
        '/OBSERVATION_DATA/u_noise': 0,
        '/OBSERVATION_DATA/AoLP': 0,
        '/OBSERVATION_DATA/AoLP_noise': 0,
        '/OBSERVATION_DATA/DoLP': 0,
        '/OBSERVATION_DATA/DoLP_noise': 0
    }

    def close(self):
        """
        Close product and check if required datasets are filled with data
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
        """
        Check variables with the same first dimension have equal sizes
        """
        warn_str = ('SPEX Level-1 format check [WARNING]:'
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
    """
    This class can be used to create a SPEXone Level-1C product

    Parameters
    ----------
    lv1_product: string
       Name of the Level-1C product
    append : boolean, optional
       Open file in append mode, parameter dims and inflight are ignored
       Default: False
    number_of_images: int
       Number of images used as input to generate the L1B product.
       Default is None, then this dimension is UNLIMITED.

    Attributes
    ----------
    product: pathlib.Path object
       Concrete path object to SPEXone Level-1 product
    inflight: boolean
       Flag to indicate data collected during in-flight of on-ground
    fid: netCDF5.Dataset object
       NetCDF4 Pointer to SPEXone Level-1 product
    dset_stored: dict
       Number of items stored for all required netCDF4 variables

    Methods
    -------
    close()
       Close product and check if required datasets are filled with data.
    epoch
       Provide epoch for SPEXone.
    get_dim(ds_name)
       Returns size of dimension.
    get_attr(attr_name, ds_name=None)
       Read data of an attribute, global or attached to a group or variable.
    set_attr(attr_name, value, ds_name=None)
       Write data to an attribute, global or attached to a group or variable.
    get_dset(ds_name)
       Read data of a netCDF4 variable.
    set_dset(ds_name, value)
       Write/append data to a netCDF4 variable.
    sec_of_day(ccsds_sec, ccsds_usec)
       Convert CCSDS timestamp to seconds after midnight.
    fill_global_attrs(level, orbit=-1, bin_size=None)
       Define global attributes in the SPEXone Level-1 products.
    check_stored()
       Check variables with the same first dimension have equal sizes.

    Notes
    -----
    ToDo: make sure we store the reference date for image_time
    """
    processing_level = 'L1C'
    dset_stored = {
        '/SENSOR_VIEWS_BANDS/view_angles': 0,
        '/SENSOR_VIEWS_BANDS/intensity_wavelengths': 0,
        '/SENSOR_VIEWS_BANDS/intensity_bandpasses': 0,
        '/SENSOR_VIEWS_BANDS/polarization_wavelengths': 0,
        '/SENSOR_VIEWS_BANDS/polarization_bandpasses': 0,
        '/SENSOR_VIEWS_BANDS/intensity_f0': 0,
        '/SENSOR_VIEWS_BANDS/polarization_f0': 0,
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
        '/OBSERVATION_DATA/I': 0,
        '/OBSERVATION_DATA/I_noise': 0,
        '/OBSERVATION_DATA/I_polsample': 0,
        '/OBSERVATION_DATA/I_polsample_noise': 0,
        '/OBSERVATION_DATA/Q': 0,
        '/OBSERVATION_DATA/Q_noise': 0,
        '/OBSERVATION_DATA/U': 0,
        '/OBSERVATION_DATA/U_noise': 0,
        '/OBSERVATION_DATA/q': 0,
        '/OBSERVATION_DATA/q_noise': 0,
        '/OBSERVATION_DATA/u': 0,
        '/OBSERVATION_DATA/u_noise': 0,
        '/OBSERVATION_DATA/DoLP': 0,
        '/OBSERVATION_DATA/DoLP_noise': 0
    }

    def close(self):
        """
        Close product and check if required datasets are filled with data
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
        """
        Check variables with the same first dimension have equal sizes
        """
        warn_str = ('SPEX Level-1 format check [WARNING]:'
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
