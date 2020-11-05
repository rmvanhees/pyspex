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
    ref_date: datetime.date object
       Reference date for 'seconds of day' parameters
    dset_stored: dict
       Number of items stored for all required netCDF4 variables

    Methods
    -------
    close()
       Close all resources (currently a placeholder function).
    epoch()
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
    sec_of_day(self, utc_sec, frac_sec)
       Convert timestamp to second of day.
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
        self.ref_date = None

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

            if 'reference_day' in self.fid.ncattrs():
                self.ref_date = datetime.fromisoformat(
                    self.fid.reference_day, tzinfo=timezone.utc)

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
    def sec_of_day(self, utc_sec: int, frac_sec: float) -> float:
        """
        Convert timestamp to second of day

        Parameters
        ----------
        utc_sec : numpy array
          seconds since 1970-01-01 (integer)
        frac_sec : numpy array
          fractional seconds (double)

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
                                     day=utc0.day, tzinfo=timezone.utc)

        # return seconds since midnight
        return (utc0 - self.ref_date) / timedelta(seconds=1) + frac_sec

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
    ref_date: datetime.date object
       Reference date for 'seconds of day' parameters
    dset_stored: dict
       Number of items stored for all required netCDF4 variables

    Methods
    -------
    close()
       Close product and check if required datasets are filled with data.
    epoch()
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
    sec_of_day(self, utc_sec, frac_sec)
       Convert timestamp to second of day.
    fill_global_attrs(level, orbit=-1, bin_size=None)
       Define global attributes in the SPEXone Level-1 products.
    check_stored()
       Check variables with the same first dimension have equal sizes.
    fill_mps(mps_data)
       Write MPS information to L1A product.
    fill_time(sec_of_day, reference_day, leap_seconds=0)
       Write TM time information to L1A product.
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
        '/engineering_data/temp_optics': 0,
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

        if self.ref_date is not None \
           and 'reference_day' not in self.fid.ncattrs():
            self.fid.reference_day = self.ref_date.strftime('%Y-%m-%d')

        # check if atleast one dataset is updated
        if self.fid.dimensions['number_of_images'].size == 0:
            self.fid.close()
            self.fid = None
            return

        # check of all required dataset their sizes
        self.check_stored(allow_empty=True)

        # update coverage time
        # ToDo replace intg by master clock cycle or frame rate (FTI)
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

    def fill_time(self, sec_of_day, reference_day, *, leap_seconds=0) -> None:
        """
        Write TM time information to L1A product

        Parameters
        ----------
        sec_of_day : numpy array (float)
           seconds since midnight
        reference_day : datetime.datetime
           midnight
        leap_second : integer
          leap seconds since 1970, use only when input are TAI seconds
        """
        self.set_dset('/image_attributes/image_time', sec_of_day)

        utc_sec = sec_of_day.astype('u4') + leap_seconds \
            + (reference_day - self.epoch).total_seconds()
        frac_sec = np.round(1e6 * (sec_of_day % 1)).astype('i4')

        self.set_dset('/image_attributes/image_CCSDS_sec', utc_sec)
        self.set_dset('/image_attributes/image_CCSDS_usec', frac_sec)

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
    ref_date: datetime.date object
       Reference date for 'seconds of day' parameters
    dset_stored: dict
       Number of items stored for all required netCDF4 variables

    Methods
    -------
    close()
       Close product and check if required datasets are filled with data.
    epoch()
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
    sec_of_day(self, utc_sec, frac_sec)
       Convert timestamp to second of day.
    fill_global_attrs(level, orbit=-1, bin_size=None)
       Define global attributes in the SPEXone Level-1 products.
    check_stored()
       Check variables with the same first dimension have equal sizes.
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

        if self.ref_date is not None \
           and 'reference_day' not in self.fid.ncattrs():
            self.fid.reference_day = self.ref_date.strftime('%Y-%m-%d')

        # check if atleast one dataset is updated
        if self.fid.dimensions['bins_along_track'].size == 0:
            self.fid.close()
            self.fid = None
            return

        # check of all required dataset their sizes
        self.check_stored()

        # update coverage time
        # ToDo epoch should be reference_day
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
    ref_date: datetime.date object
       Reference date for 'seconds of day' parameters
    dset_stored: dict
       Number of items stored for all required netCDF4 variables

    Methods
    -------
    close()
       Close product and check if required datasets are filled with data.
    epoch()
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
    sec_of_day(self, utc_sec, frac_sec)
       Convert timestamp to second of day.
    fill_global_attrs(level, orbit=-1, bin_size=None)
       Define global attributes in the SPEXone Level-1 products.
    check_stored()
       Check variables with the same first dimension have equal sizes.
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

        if self.ref_date is not None \
           and 'reference_day' not in self.fid.ncattrs():
            self.fid.reference_day = self.ref_date.strftime('%Y-%m-%d')

        # check if atleast one dataset is updated
        if self.fid.dimensions['bins_along_track'].size == 0:
            self.fid.close()
            self.fid = None
            return

        # check of all required dataset their sizes
        self.check_stored()

        # update coverage time
        # ToDo epoch should be reference_day
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
