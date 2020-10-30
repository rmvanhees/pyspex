"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation to create the SPEXone CKD product, or update/read CKD
parameters

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime
from pathlib import Path
# import argparse

import h5py
import numpy as np

from pyspex import version

# - global parameters ------------------------------
VERSION = '000100'


# - local functions --------------------------------
def str_netcdf():
    """
    return string to identify a netCDF dimension
    """
    return 'This is a netCDF dimension but not a netCDF variable.'


# - class CKDio -------------------------
class CKDio:
    """
    Defines class to read SPEXone CKD parameters

    Attributes
    ----------
    filename :  str
    offset : int
    verbose : bool

    Methods
    -------
    add_offset(values, binning=-1)
       Add Offset CKD to CKD product.
    add_darkflux(values, binning=-1)
       Add Dark-flux CKD to CKD product.
    add_non_linearity(values, binning=-1)
       Add non-linearity CKD to CKD product.
    add_prnu(values, binning=-1)
       Add PRNU CKD to CKD product.
    get_offset(binning=-1)
       Read Offset CKD.
    get_darkflux(binning=-1)
       Read Dark-flux CKD.
    get_non_linearity(values, binning=-1)
       Read non-Linearity CKD.
    get_prnu(binning=-1)
       Read PRNU CKD.

    Notes
    -----

    Examples
    --------
    """
    def __init__(self, ckd_file, verbose=False):
        """
        Initialize class attributes

        Parameters
        ----------
        ckd_file :  str
           Name of CKD file
        verbose :  bool
           Be verbose
        """
        self.filename = ckd_file
        self.offset = 0
        self.verbose = verbose

    def __initialize_product(self, valid_date=None, overwrite=False):
        """
        Initialize SPEXone product with CKD parameters

        Define name of SPEXone CKD product as:
           SPX1_L1_CKD_<date_valid>_<version>_<date_created>.h5
        - date_valid : ISO UTC date-time as yymmddTHHMMSS
        - version : 6 digits as MMmmrr (for Major, minor and release versions)
        - date_created : ISO UTC date-time as yymmddTHHMMSS
        """
        if valid_date is None:
            valid_date = '20191001T000000'

        creation_date = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        self.filename = 'SPX1_L1_CKD_{:13s}_{:6s}_{:13s}.h5'.format(
            valid_date, VERSION, creation_date)

        if Path(self.filename).is_file():
            if not overwrite:
                raise FileExistsError
            Path(self.filename).unlink()

        with h5py.File(self.filename, 'w') as fid:
            fid.attrs['title'] = 'SPEXone Level-1 CKD product'
            fid.attrs['comment'] = ''
            fid.attrs['name'] = self.filename
            fid.attrs['type'] = 'CKD product'
            fid.attrs['mission'] = 'PACE'
            fid.attrs['instrument'] = 'SPEXone'
            fid.attrs['creator'] = \
                'SRON Netherlands Institute for Space Research'
            fid.attrs['creator_date'] = creation_date
            fid.attrs['creator_version'] = VERSION
            fid.attrs['sw_version'] = version.get()
            fid.attrs['valitity_start'] = valid_date
            fid.attrs['validity_stop'] = '20500101T000000'

            grp = fid.create_group('/FULL_FRAME')
            # create dimensions
            dset = grp.create_dataset('row', (2048,), dtype='u2')
            dset.make_scale(str_netcdf())
            dset.attrs['long_name'] = 'row'
            dset[:] = np.arange(0, 2048, dtype='u2')

            dset = grp.create_dataset('column', (2048,), dtype='u2')
            dset.make_scale(str_netcdf())
            dset.attrs['long_name'] = 'column'
            dset[:] = np.arange(0, 2048, dtype='u2')

            _ = fid.create_group('/BINNING_00')
            # create dimensions of default binning scheme

    def add_offset(self, values, binning=-1):
        """
        Add Offset CKD to CKD product
        """
        if self.filename is None or not Path(self.filename).is_file():
            self.__initialize_product()

        with h5py.File(self.filename, 'r+') as fid:
            if binning == -1:
                grp = fid['/FULL_FRAME']
            else:
                grp = fid['/BINNING_{:03d}'.format(binning)]
            if 'offset' in fid:
                raise ValueError('dataset offset already exists')

            frame_shape = (grp['row'].size, grp['column'].size)
            dset = grp.create_dataset('offset', frame_shape,
                                      compression=1, shuffle=True,
                                      fillvalue=np.nan, dtype='f4')
            dset.dims[0].attach_scale(grp['row'])
            dset.dims[1].attach_scale(grp['column'])
            dset.attrs['long_name'] = "Offset"
            dset[:] = values
            dset.attrs['date_created'] = \
                datetime.utcnow().isoformat(timespec='milliseconds')

    def add_darkflux(self, values, binning=-1):
        """
        Add Dark-flux CKD to CKD product
        """
        if self.filename is None or not Path(self.filename).is_file():
            self.__initialize_product()

        with h5py.File(self.filename, 'r+') as fid:
            if binning == -1:
                grp = fid['/FULL_FRAME']
            else:
                grp = fid['/BINNING_{:03d}'.format(binning)]
            if 'darkflux' in fid:
                raise ValueError('dataset darkflux already exists')

            frame_shape = (grp['row'].size, grp['column'].size)
            dset = grp.create_dataset('darkflux', frame_shape,
                                      compression=1, shuffle=True,
                                      fillvalue=np.nan, dtype='f4')
            dset.dims[0].attach_scale(grp['row'])
            dset.dims[1].attach_scale(grp['column'])
            dset.attrs['long_name'] = "Dark-flux"
            dset[:] = values
            dset.attrs['date_created'] = \
                datetime.utcnow().isoformat(timespec='milliseconds')

    def add_non_linearity(self, values, binning=-1):
        """
        Add non-linearity CKD to CKD product
        """
        if self.filename is None or not Path(self.filename).is_file():
            self.__initialize_product()

        with h5py.File(self.filename, 'r+') as fid:
            if binning == -1:
                grp = fid['/FULL_FRAME']
            else:
                grp = fid['/BINNING_{:03d}'.format(binning)]
            if 'nonlinearity' in fid:
                raise ValueError('dataset nonlinearity already exists')
            raise ValueError('not yet implemented')

    def add_prnu(self, values, binning=-1):
        """
        Add PRNU CKD to CKD product
        """
        if self.filename is None or not Path(self.filename).is_file():
            self.__initialize_product()

        with h5py.File(self.filename, 'r+') as fid:
            if binning == -1:
                grp = fid['/FULL_FRAME']
            else:
                grp = fid['/BINNING_{:03d}'.format(binning)]
            if 'PRNU' in fid:
                raise ValueError('dataset PRNU already exists')

            frame_shape = (grp['row'].size, grp['column'].size)
            dset = grp.create_dataset('PRNU', frame_shape,
                                      compression=1, shuffle=True,
                                      fillvalue=np.nan, dtype='f4')
            dset.dims[0].attach_scale(grp['row'])
            dset.dims[1].attach_scale(grp['column'])
            dset.attrs['long_name'] = "Pixel Response Non-Uniformity"
            dset[:] = values
            dset.attrs['date_created'] = \
                datetime.utcnow().isoformat(timespec='milliseconds')

    def get_offset(self, binning=-1):
        """
        Read Offset CKD
        """
        with h5py.File(self.filename, 'r') as fid:
            if binning == -1:
                grp = fid['/FULL_FRAME']
            else:
                grp = fid['/BINNING_{:03d}'.format(binning)]
            if 'offset' not in grp:
                raise ValueError('CKD offset does not exist')

            values = grp['offset'][:]

        return values

    def get_darkflux(self, binning=-1):
        """
        Read Dark-flux CKD
        """
        with h5py.File(self.filename, 'r') as fid:
            if binning == -1:
                grp = fid['/FULL_FRAME']
            else:
                grp = fid['/BINNING_{:03d}'.format(binning)]
            if 'darkflux' not in grp:
                raise ValueError('CKD darkflux does not exist')

            values = grp['darkflux'][:]

        return values

    def get_non_linearity(self, values, binning=-1):
        """
        Read non-Linearity CKD
        """
        with h5py.File(self.filename, 'r') as fid:
            if binning == -1:
                grp = fid['/FULL_FRAME']
            else:
                grp = fid['/BINNING_{:03d}'.format(binning)]
            if 'nonlinearity' not in grp:
                raise ValueError('CKD nonlinearity does not exist')

            values = grp['nonlinearity'][:]

        return values

    def get_prnu(self, binning=-1):
        """
        Read PRNU CKD
        """
        with h5py.File(self.filename, 'r') as fid:
            if binning == -1:
                grp = fid['/FULL_FRAME']
            else:
                grp = fid['/BINNING_{:03d}'.format(binning)]
            if 'PRNU' not in grp:
                raise ValueError('CKD PRNU does not exist')

            values = grp['PRNU'][:]

        return values


# - main function ----------------------------------
def main():
    """
    main function
    """
    ckd = CKDio(None, verbose=True)
    ckd.add_offset(np.zeros((2048, 2048), dtype=float))
    ckd.add_darkflux(np.zeros((2048, 2048), dtype=float))
    ckd.add_prnu(np.ones((2048, 2048), dtype=float))

    print(np.mean(ckd.get_offset()))
    print(np.mean(ckd.get_darkflux()))
    print(np.mean(ckd.get_prnu()))


# --------------------------------------------------
if __name__ == '__main__':
    main()
