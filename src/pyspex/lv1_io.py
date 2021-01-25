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

from .tm_science import TMscience
from .lib.attrs_def import attrs_def
from .lib.l1a_def import init_l1a
from .lib.l1b_def import init_l1b
from .lib.l1c_def import init_l1c

# - global parameters -------------------
MCP_TO_SEC = 1e-7


def frac_poly(xx_in, coefs=None):
    """
    Temperature [K] calibration derived by Paul Tol (2020-10-21)

    Parameters
    ----------
    xx    :  ndarray
    coefs :  tuple
      coefficients of fractional polynomial: r0, r1, r2, r3, r4

    Returns
    -------
    ndarray, dtype float
    """
    xx = xx_in.astype(float)

    if coefs is None:
        coefs = (273.15 + 21.19, 6.97828e+7,
                 -3.53275e-25, 7.79625e-31, -4.6505E-32)

    return (coefs[0]
            + coefs[1] / xx
            + coefs[2] * xx ** 4
            + (coefs[3] + coefs[4] * np.log(xx)) * xx ** 5)


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
    fill_global_attrs(level, orbit=-1, bin_size=None)
       Define global attributes in the SPEXone Level-1 products.

    Notes
    -----
    The engineering data should be extended, suggestions:
    * Temperatures of a.o. detector, FEE, optica, obm, telescope
    * Instrument settings: exposure time, dead time, frame time, coadding, ...
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

        self.dset_stored[name] += 1 if value.shape == () else value.shape[0]

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
    fill_global_attrs(level, orbit=-1, bin_size=None)
       Define global attributes in the SPEXone Level-1 products.
    check_stored(allow_empty=False)
       Check variables with the same first dimension have equal sizes.
    sec_of_day(ccsds_sec, ccsds_subsec)
       Convert CCSDS timestamp to seconds after midnight.
    fill_time(ccsds_sec, ccsds_subsec, group=None)
       Write time of Science telemetry packets (UTC/TAI) to L1A product.
    fill_science(img_data, img_hk, img_id)
       Write Science data and housekeeping telemetry (Science) to L1A product.
    fill_nomhk(nomhk_data)
       Write nominal housekeeping telemetry packets (NomHK) to L1A product.
    fill_demhk(demhk_data)
       Write detector housekeeping telemetry packets (DemHK) to L1A product.
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
        '/image_attributes/image_CCSDS_subsec': 0,
        '/image_attributes/image_time': 0,
        '/image_attributes/image_ID': 0,
        '/engineering_data/NomHK_telemetry': 0,
        '/engineering_data/DemHK_telemetry': 0,
        '/engineering_data/temp_detector': 0,
        '/engineering_data/temp_housing': 0,
        '/engineering_data/temp_radiator': 0,
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

        # define object to access Science telemetry parameters
        mps = TMscience(self.get_dset('/science_data/detector_telemetry')[-1])

        # determine duration master clock cycle
        imro = 1e-1 * mps.get('FTI') * 2
        mcycl = 1e-1 * mps.get('FTI') * mps.get('REG_NCOADDFRAMES')

        img_sec = self.fid['/image_attributes/image_CCSDS_sec'][:].data
        img_subsec = self.fid['/image_attributes/image_CCSDS_subsec'][:].data

        time0 = (self.epoch
                 + timedelta(seconds=int(img_sec[0]))
                 + timedelta(microseconds=int(1e6 * img_subsec[0] / 65536))
                 - timedelta(milliseconds=mcycl + imro))

        time1 = (self.epoch
                 + timedelta(seconds=int(img_sec[-1]))
                 + timedelta(microseconds=int(1e6 * img_subsec[-1] / 65536))
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
    def sec_of_day(self, ccsds_sec, ccsds_subsec) -> tuple:
        """
        Convert CCSDS timestamp to seconds after midnight

        Parameters
        ----------
        ccsds_sec : numpy array (dtype='u4')
          Seconds since 1970-1-1
        ccsds_subsec : numpy array (dtype='u2')
          Sub-seconds as (1 / 2**16) seconds

        Returns
        -------
        numpy.ndarray with sec_of_day
        """
        # determine midnight before start measurement
        tstamp0 = self.epoch + timedelta(seconds=int(ccsds_sec[0]))
        reference_day = datetime(year=tstamp0.year,
                                 month=tstamp0.month,
                                 day=tstamp0.day, tzinfo=timezone.utc)

        # store seconds since midnight
        sec_of_day = ccsds_sec - (reference_day - self.epoch).total_seconds()

        # return seconds since midnight
        return (reference_day, sec_of_day + ccsds_subsec / 65536)

    def fill_time(self, ccsds_sec, ccsds_subsec, group=None) -> None:
        """
        Write time of Science telemetry packets (UTC/TAI) to L1A product

        Parameters
        ----------
        ccsds_sec : numpy array (dtype='u4')
          Seconds since 1970-1-1
        ccsds_subsec : numpy array (dtype='u2')
          Sub-seconds as (1 / 2**16) seconds

        Note
        ----
        Writes parameters: image_time, image_CCSDS_sec and image_CCSDS_subsec
        """
        if group is None:
            group = 'image_attributes'

        # calculate seconds of day
        reference_day, sec_of_day = self.sec_of_day(ccsds_sec, ccsds_subsec)

        if group in ('image_attributes', '/image_attributes'):
            self.set_dset('/image_attributes/image_CCSDS_sec', ccsds_sec)
            self.set_dset('/image_attributes/image_CCSDS_subsec', ccsds_subsec)
            self.set_dset('/image_attributes/image_time', sec_of_day)
            self.set_attr('units',
                          'seconds since {}'.format(reference_day.isoformat()),
                          ds_name='/image_attributes/image_time')
        elif group in ('engineering_data', '/engineering_data'):
            self.set_dset('/engineering_data/HK_tlm_time', sec_of_day)
            self.set_attr('units',
                          'seconds since {}'.format(reference_day.isoformat()),
                          ds_name='/engineering_data/HK_tlm_time')
        else:
            self.set_dset('/navigation_data/att_time', sec_of_day)
            self.set_attr('units',
                          'seconds since {}'.format(reference_day.isoformat()),
                          ds_name='/navigation_data/att_time')
            self.set_dset('/navigation_data/orb_time', sec_of_day)
            self.set_attr('units',
                          'seconds since {}'.format(reference_day.isoformat()),
                          ds_name='/navigation_data/orb_time')

    def fill_science(self, img_data, img_hk, img_id) -> None:
        """
        Write Science data and housekeeping telemetry (Science) to L1A product

        Parameters
        ----------
        img_data : numpy array (uint16)
           Detector image data
        img_hk : numpy array ()
           Structured array with all Science telemetry parameters
        img_id : numpy array (uint16)
           Detector image counter

        Notes
        -----
        Adds detector_telemetry data to the group /science_data

        Parameters: binning_table, digital_offset, exposure_time
        and nr_coadding are extracted from the telemetry packets and writen
        in the group /image_attributes
        """
        if len(img_hk) == 0:
            return

        self.set_dset('/science_data/detector_images', img_data)
        self.set_dset('/science_data/detector_telemetry', img_hk)

        mps = TMscience(img_hk)
        self.set_dset('/image_attributes/binning_table', mps.binning_table_id)
        self.set_dset('/image_attributes/digital_offset', mps.offset)
        self.set_dset('/image_attributes/exposure_time',
                      MCP_TO_SEC * mps.exp_time)
        self.set_dset('/image_attributes/nr_coadditions',
                      mps.get('REG_NCOADDFRAMES'))
        self.set_dset('/image_attributes/image_ID', img_id)

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
        """
        Write detector housekeeping telemetry packets (DemHK) to L1A product

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
