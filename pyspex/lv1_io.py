"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python class to create a SPEXone Level-1 products

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path, PurePosixPath

from datetime import datetime, timedelta

import numpy as np

from netCDF4 import Dataset

from .lib.attrs_def import attrs_def
from .lib.l1a_def import init_l1a
from .lib.l1b_def import init_l1b
from .lib.l1c_def import init_l1c

# - global parameters -------------------


# - class LV1io -------------------------
class Lv1io:
    """
    Generic class to create SPEXone Level-1 products

    Parameters
    ----------
    lv1_product: string
       Name of the Level-1 product
    append : boolean, optional
       Open file in append mode, parameter dims and inflight are ignored
       Default: False

    Attributes
    ----------
    product: pathlib.Path object
       Concrete path object to SPEXone Level-1 product
    inflight: boolean
       Flag to indicate data collected during in-flight of on-ground
    fid: netCDF5.Dataset object
       NetCDF4 Pointer to SPEXone Level-1 product
    ref_date: datetime.date object
       Reference date for 'seconds of day' parameters
    dset_stored: dict
       Number of items stored for all required netCDF4 variables

    Methods
    -------
    get_dims(ds_name)
       Returns size of dimension
    get_attr(attr_name, ds_name=None)
       Read data of an attribute, global or attached to a group or variable
    set_attr(attr_name, value, ds_name=None)
       Write data to an attribute, global or attached to a group or variable
    get_dset(ds_name)
       Read data of a netCDF4 variable
    set_dset(ds_name, value)
       Write data to a netCDF4 variable
    fill_global_attrs(level, orbit=-1, bin_size=None)

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
        """
        self.__epoch = datetime(1970, 1, 1)

        # initialize private class-attributes
        self.product = Path(product)
        self.inflight = True
        if 'inflight' in kwargs:
            self.inflight = kwargs['inflight']
        self.fid = None
        self.ref_date = None

        # initialize Level-1 product
        if not append:
            if self.processing_level == 'L1A':
                init_l1a(product, **kwargs)
            if self.processing_level == 'L1B':
                init_l1b(product, **kwargs)
            if self.processing_level == 'L1C':
                init_l1c(product, **kwargs)
            else:
                raise KeyError('valid processing levels are: L1A, L1B or L1C')

        # open Level-1 product in append mode
        self.fid = Dataset(self.product, "r+")
        if append:
            if 'reference_day' in self.fid.ncattrs():
                self.ref_date = datetime.fromisoformat(self.fid.reference_day)

            # store current length of the first dimension
            for key in self.dset_stored:
                self.dset_stored[key] = self.fid[key].shape[0]

    def __repr__(self):
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

    def __exit__(self, exc_type, exc_value, traceback):
        """
        method called when exiting the context manager
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self):
        """
        close product and check if required datasets are filled with data
        """
        if self.fid is None:
            return

        if self.ref_date is not None \
           and 'reference_day' not in self.fid.ncattrs():
            self.fid.reference_day = self.ref_date.strftime('%Y-%m-%d')

        # check if atleast one dataset is updated
        if self.fid.dimensions['number_of_images'].size == 0:
            self.fid.close()
            self.fid = None
            return

        # update coverage time
        intg = (self.fid['/image_attributes/exposure_time'][-1].data
                * self.fid['/image_attributes/nr_coadditions'][-1].data)

        img_sec = self.fid['/image_attributes/image_CCSDS_sec'][:].data
        img_usec = self.fid['/image_attributes/image_CCSDS_usec'][:].data

        time0 = (self.epoch
                 + timedelta(seconds=int(img_sec[0]))
                 + timedelta(microseconds=int(img_usec[0]))
                 - timedelta(seconds=intg))

        time1 = (self.epoch
                 + timedelta(seconds=int(img_sec[-1]))
                 + timedelta(microseconds=int(img_usec[-1])))

        # print('Check if all required dataset are filled with data')
        self.fid.time_coverage_start = time0.isoformat(timespec='milliseconds')
        self.fid.time_coverage_end = time1.isoformat(timespec='milliseconds')

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
        Read data of a netCDF4 variable

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
    def sec_of_day(self, utc_sec: int, frac_sec: float) -> float:
        """
        Parameters
        ----------
        utc_sec : numpy array
          seconds since 1970-01-01 (integer)
        frac_sec : numpy array
          fractional seconds (double)
        leap_second : integer
          leap seconds since 1970

        Returns
        -------
        - second of day
        """
        # get seconds since epoch
        utc0 = self.epoch + timedelta(seconds=int(utc_sec))

        # get seconds between midnight and epoch
        if self.ref_date is None:
            self.ref_date = datetime(year=utc0.year,
                                     month=utc0.month,
                                     day=utc0.day)

        # return seconds since midnight
        return (utc0 - self.ref_date) / timedelta(seconds=1) + frac_sec

    # -------------------------
    def fill_global_attrs(self, orbit=-1, bin_size=None) -> None:
        """
        Define global attributes in the SPEXone Level-1 products
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
        '/engineering_data/temp_optics': 0,
        '/engineering_data/HK_tlm_time': 0,
        '/navigation_data/adstate': 0,
        '/navigation_data/att_quat': 0,
        '/navigation_data/orb_pos': 0,
        '/navigation_data/orb_vel': 0,
        '/navigation_data/att_time': 0,
        '/navigation_data/orb_time': 0
    }

    # ---------- PUBLIC FUNCTIONS ----------
    def fill_mps(self, mps_data) -> None:
        """
        Write MPS information to L1A product

        Parameters
        ----------
        mps_data : numpy array
          Structured array with all MPS parameters

        Notes
        -----
        - mps_data as /science_data/detector_telemetry

        User friendly parameters:
        - exposure time as /image_attributes/exposure_time
        - nr_coadditions as /image_attributes/nr_coadditions
        """
        def digital_offset(mps) -> int:
            """
            Return digital offset including ADC offset
            """
            buff = mps['DET_OFFSET'].astype('i4')
            buff[buff >= 8192] -= 16384

            return buff + 70

        def exposure_time(mps, mcp=1e-7) -> float:
            """
            Return detector exposure_time
            """
            exptime = 129 * (0.43 * mps['DET_FOTLEN'] + mps['DET_EXPTIME'])

            return mcp * exptime

        self.set_dset('/science_data/detector_telemetry', mps_data)

        self.set_dset('/image_attributes/binning_table',
                      mps_data['REG_BINNING_TABLE'])
        self.set_dset('/image_attributes/digital_offset',
                      digital_offset(mps_data))
        self.set_dset('/image_attributes/exposure_time',
                      exposure_time(mps_data))
        self.set_dset('/image_attributes/nr_coadditions',
                      mps_data['REG_NCOADDFRAMES'])

    def fill_time(self, utc_sec, frac_sec, ibgn=-1, *, leap_seconds=0) -> None:
        """
        Write TM time information to L1A product

        Parameters
        ----------
        utc_sec : numpy array
          seconds since 1970-01-01 (integer)
        frac_sec : numpy array
          fractional seconds (double)
        leap_second : integer
          leap seconds since 1970, use only when input are TAI seconds

        Stores
        ------
        - utc_sec + leap_seconds as /image_attributes/image_CCSDS_sec
        - frac_sec as /image_attributes/image_CCSDS_usec

        User friendly parameter 'image_time' as seconds since midnight:
        - utc_sec, frac_sec as /image_attributes/image_time
        """
        self.set_dset('/image_attributes/image_CCSDS_sec',
                      utc_sec + leap_seconds, ibgn)
        self.set_dset('/image_attributes/image_CCSDS_usec',
                      np.round(1e6 * frac_sec).astype(np.int32), ibgn)

        image_time = np.empty((utc_sec.size,), dtype=float)
        for ii in range(utc_sec.size):
            image_time[ii] = self.sec_of_day(utc_sec[ii], frac_sec[ii])

        self.set_dset('/image_attributes/image_time', image_time, ibgn)

    def fill_hk_time(self, utc_sec, frac_sec) -> None:
        """
        Write house-keeping timestamps

        Parameters
        ----------
        utc_sec : numpy array
          seconds since 1970-01-01 (integer)
        frac_sec : numpy array
          fractional seconds (double)
        leap_second : integer
          leap seconds since 1970

        Stores
        ------
        - utc_sec, frac_sec as /engineering_data/HK_tlm_time
        """
        hk_time = np.empty((utc_sec.size,), dtype=float)
        for ii in range(utc_sec.size):
            hk_time[ii] = self.sec_of_day(utc_sec[ii], frac_sec[ii])

        self.set_dset('/engineering_data/HK_tlm_time', hk_time)

    def fill_att_time(self, utc_sec, frac_sec) -> None:
        """
        Write attitude timestamps

        Parameters
        ----------
        utc_sec : numpy array
          seconds since 1970-01-01 (integer)
        frac_sec : numpy array
          fractional seconds (double)
        leap_second : integer
          leap seconds since 1970

        Stores
        ------
        - utc_sec, frac_sec as /navigation_data/att_time
        """
        att_time = np.empty((utc_sec.size,), dtype=float)
        for ii in range(utc_sec.size):
            att_time[ii] = self.sec_of_day(utc_sec[ii], frac_sec[ii])

        self.set_dset('/navigation_data/att_time', att_time)

    def fill_orb_time(self, utc_sec, frac_sec) -> None:
        """
        Write orbit vector timestamps

        Parameters
        ----------
        utc_sec : numpy array
          seconds since 1970-01-01 (integer)
        frac_sec : numpy array
          fractional seconds (double)
        leap_second : integer
          leap seconds since 1970

        Stores
        ------
        - utc_sec, frac_sec as /navigation_data/orb_time
        """
        orb_time = np.empty((utc_sec.size,), dtype=float)
        for ii in range(utc_sec.size):
            orb_time[ii] = self.sec_of_day(utc_sec[ii], frac_sec[ii])

        self.set_dset('/navigation_data/orb_time', orb_time)

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
            dset.comment = "t_sat = min([2.28e-9 / S_reference, 30])"
            dset.units = 'A'
            dset[:] = reference['value']

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
    orbit_number: int
       Orbit revolution counter, default=-1
    number_of_images: int
       Number of images used as input to generate the L1B product.
       Default is None, then this dimension is UNLIMITED.
    spatial_samples: int
       Total number of spatial samples from all viewports, default is 200
    """
    processing_level = 'L1B'
    dset_stored = {
        '/SENSOR_VIEW_BANDS/viewport_index': 0,
        '/SENSOR_VIEW_BANDS/view_angles': 0,
        '/SENSOR_VIEW_BANDS/intensity_wavelengths': 0,
        '/SENSOR_VIEW_BANDS/intensity_bandpasses': 0,
        '/SENSOR_VIEW_BANDS/polarization_wavelengths': 0,
        '/SENSOR_VIEW_BANDS/polarization_bandpasses': 0,
        '/SENSOR_VIEW_BANDS/intensity_f0': 0,
        '/SENSOR_VIEW_BANDS/polarization_f0': 0,
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
    orbit_number: int
       Orbit revolution counter, default=-1
    number_of_images: int
       Number of images used as input to generate the L1B product.
       Default is None, then this dimension is UNLIMITED.
    """
    processing_level = 'L1C'
    dset_stored = {
        '/SENSOR_VIEW_BANDS/view_angles': 0,
        '/SENSOR_VIEW_BANDS/intensity_wavelengths': 0,
        '/SENSOR_VIEW_BANDS/intensity_bandpasses': 0,
        '/SENSOR_VIEW_BANDS/polarization_wavelengths': 0,
        '/SENSOR_VIEW_BANDS/polarization_bandpasses': 0,
        '/SENSOR_VIEW_BANDS/intensity_f0': 0,
        '/SENSOR_VIEW_BANDS/polarization_f0': 0,
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

    # ---------- PUBLIC FUNCTIONS ----------
