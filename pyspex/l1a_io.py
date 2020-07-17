"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python class to create a SPEXone L1A calibration product

Copyright (c) 2019 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path, PurePosixPath

from datetime import datetime, timedelta

import numpy as np

from netCDF4 import Dataset

from pyspex.lib.l1a_def import init_l1a

# - global parameters -------------------


# - local functions ---------------------


# - class L1Aio -------------------------
class L1Aio:
    """
    This class can be used to define a SPEXone L1A calibration product
    or to read from an existing SPEXone L1A calibration product.

    Parameters
    ----------
    l1a_product: string
       Name of the L1A product
    dims: dictionary, optional
       Provide size of various dimensions.
       Default values:
            number_of_images : None     # number of image frames
            samples_per_image : 184000  # depends on binning table
            SC_records : None           # space-craft navigation records (1 Hz?)
            hk_packets : None           # number of HK tlm-packets (1 Hz?)
            wavelength : None
    inflight: boolean, optional
       In-flight data, only affects global attributes of L1A product.
       Default: False
    append : boolean, optional
       Open file in append mode, parameter dims and inflight are ignored
       Defaut: False

    Attributes
    ----------
    l1a_path: object
       pathlib object
    fid: object
       netCDF4.Dataset object
    written_dset: list
       list of object written to SPEXone L1A product

    Methods
    -------
    set_attr(name, value, ds_name=None)
       Write data to an attribute, global or attached to a dataset
    get_attr(name, ds_name=None)
       Read data of an attribute, global or attached to a dataset
    set_dset(name, value)
       Write data to a dataset
    get_dset(name)
       Read data of a dataset
    fill_mps(mps)
       Write MPS information to L1A product
    fill_time(utc_sec, frac_sec)
       Write time information of telemetry data to L1A product
    fill_images(images)
       Write image data to L1A product

    Notes
    -----
    The engineering data should be extended, suggestions:
    * temperatures of a.o. detector, FEE, optica, obm, telescope
    * instrument settings: exposure time, dead time, frame time, coadding, ...
    """
    def __init__(self, l1a_product: str,
                 dims=None, inflight=False, append=False):
        """
        Initialize access to a SPEXone L1a product
        """
        self.__epoch = datetime(1970, 1, 1)

        # initialize private class-attributes
        self.l1a_path = Path(l1a_product)
        self.fid = None
        self.written_dset = []

        # initialize L1A product
        if not append:
            if dims is None:
                dims = {}
            init_l1a(self.l1a_path, dims, inflight=inflight)

        # open L1A product in append mode
        self.fid = Dataset(self.l1a_path, "r+")

    def __repr__(self):
        class_name = type(self).__name__
        return '{}({!r})'.format(class_name, self.l1a_path)

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

        # check if atleast one dataset is updated
        if not self.written_dset \
           or self.fid.dimensions['number_of_images'].size == 0:
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
        self.fid.time_coverage_stop = time1.isoformat(timespec='milliseconds')

        self.fid.close()
        self.fid = None

    @property
    def epoch(self) -> datetime:
        """
        Provide epoch for SPEXone
        """
        return self.__epoch

    # ---------- PUBLIC FUNCTIONS ----------
    def set_attr(self, name: str, value, ds_name=None) -> None:
        """
        Write or update an attribute of a SPEXone L1A product

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
                        'ds_name {} not present in L1A product'.format(ds_name))
            else:
                if var_name not in self.fid.groups \
                   and var_name not in self.fid.variables:
                    raise KeyError(
                        'ds_name {} not present in L1A product'.format(ds_name))

            if isinstance(value, str):
                self.fid[ds_name].setncattr(name, np.string_(value))
            else:
                self.fid[ds_name].setncattr(name, value)

    # -------------------------
    def get_attr(self, name: str, ds_name=None):
        """
        Read the value of a SPEXone L1A product attribute

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

    # -------------------------
    def get_dim(self, name: str):
        """
        Get size of dimension
        """
        return self.fid.dimensions[name].size

    # -------------------------
    def set_dset(self, name: str, value, ibgn=0) -> None:
        """
        Write values to a SPEXone L1A dataset

        Parameters
        ----------
        name : string
           name of L1A dataset
        value : scalar, array_like
           value or values to be written
        """
        grp_name = str(PurePosixPath(name).parent)
        var_name = str(PurePosixPath(name).name)
        if grp_name != '.':
            if var_name not in self.fid[grp_name].variables:
                raise KeyError(
                    'dataset {} not present in L1A product'.format(name))
        else:
            if var_name not in self.fid.variables:
                raise KeyError(
                    'dataset {} not present in L1A product'.format(name))

        dims = self.fid[name].get_dims()
        if not dims:
            self.fid[name][...] = value
        elif dims[0].isunlimited():
            self.fid[name][ibgn:, ...] = value
        else:
            self.fid[name][...] = value

        if name not in self.written_dset:
            self.written_dset.append(name)

    # -------------------------
    def get_dset(self, name: str):
        """
        Read values from a SPEXone L1A dataset

        Parameters
        ----------
        name : string
           name of L1A dataset

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
                    'dataset {} not present in L1A product'.format(name))
        else:
            if var_name not in self.fid.variables:
                raise KeyError(
                    'dataset {} not present in L1A product'.format(name))

        return self.fid[name][:]

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

    def fill_time(self, utc_sec, frac_sec, ibgn=0, *, leap_seconds=0) -> None:
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

        # get seconds since midnight
        utc0 = self.epoch + timedelta(seconds=int(utc_sec[0]))
        ref_date = datetime(year=utc0.year, month=utc0.month, day=utc0.day)

        image_time = np.empty((utc_sec.size,), dtype=float)
        for ii in range(utc_sec.size):
            utc = self.epoch + timedelta(seconds=int(utc_sec[ii]))
            image_time[ii] = (
                (utc - ref_date) / timedelta(seconds=1) + frac_sec[ii])

        ds_name = '/image_attributes/image_time'
        self.set_dset(ds_name, image_time, ibgn)
        self.set_attr('reference', ref_date.isoformat(), ds_name=ds_name)

    def fill_hk_time(self, utc_sec, frac_sec, leap_seconds=27) -> None:
        """
        Write house-keeping time information to L1A product

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
        # get seconds since midnight
        utc0 = self.epoch + timedelta(seconds=int(utc_sec[0]))
        ref_date = datetime(year=utc0.year, month=utc0.month, day=utc0.day)

        hk_time = np.empty((utc_sec.size,), dtype=float)
        for ii in range(utc_sec.size):
            utc = self.epoch + timedelta(seconds=int(utc_sec[ii]))
            hk_time[ii] = (
                (utc - ref_date) / timedelta(seconds=1) + frac_sec[ii])

        ds_name = '/engineering_data/HK_tlm_time'
        self.set_dset(ds_name, hk_time)
        self.set_attr('reference', ref_date.isoformat(), ds_name=ds_name)

    def fill_images(self, images, ibgn=0) -> None:
        """
        Write image data to L1A product

        Parameters
        ----------
        images : numpy array
          detector images

        Notes
        -----
        - images as /science_data/detector_images
        """
        self.set_dset('/science_data/detector_images', images, ibgn)

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
