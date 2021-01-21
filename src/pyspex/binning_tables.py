"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python class to handle SPEXone binning tables

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from pyspex import version


# - global parameters ------------------------------
FILL_VALUE = 0xFFFFFFFF  # 0X7FFFFFFF


# - local functions --------------------------------
class BinningTables:
    """
    Defines class to store and obtain SPEXone binning CKD

    Attributes
    ----------

    Methods
    -------

    Notes
    -----

    Examples
    --------
    """
    def __init__(self, mode='read', ckd_dir=None):
        """
        Initialize class attributes
        """
        if ckd_dir is None:
            self.ckd_dir = Path('/nfs/SPEXone/share/ckd')
            if not self.ckd_dir.is_dir():
                self.ckd_dir = Path('/data/richardh/SPEXone/share/ckd')
        else:
            self.ckd_dir = Path(ckd_dir)
        if not self.ckd_dir.is_dir():
            raise FileNotFoundError('directory with SPEXone CKD does not exist')
        self.ckd_file = None

        if mode == 'read':
            self.search_ckd()
        elif mode == 'write':
            self.create_ckd()
        else:
            self.ckd_file = mode

    def search_ckd(self):
        """
        Search CKD file with binning tables
        """
        ckd_files = list(Path(self.ckd_dir).glob('SPX1_OCAL_L1A_TBL_*.nc'))
        if not ckd_files:
            raise FileNotFoundError('No CKD with binning tables found')

        # only read the latest version of the binning-table CKD
        self.ckd_file = sorted(ckd_files)[-1]

    def create_ckd(self, release=1):
        """
        Initialize netCDF file with binning tables
        """
        # initialize netCDF file with binning tables
        self.ckd_file = 'SPX1_OCAL_L1A_TBL_{:s}_{:04d}.nc'.format(
            datetime.utcnow().strftime("%Y%m%dT%H%M%S"), release)

        with Dataset(self.ckd_dir / self.ckd_file, 'w') as fid:
            fid.setncattr('title', 'SPEXone Level-1 binning-tables')
            fid.setncattr('processing_version', version.get())
            fid.setncattr('date_created',
                          datetime.utcnow().isoformat(timespec='seconds'))
            fid.setncattr('Convensions', 'CF-1.6')
            fid.setncattr('project', 'PACE Project')
            fid.setncattr('instrument', 'SPEXone')
            fid.setncattr('institution',
                          'SRON Netherlands Institute for Space Research')

            fid.createDimension('row', 1024)
            fid.createDimension('column', 1024)

    def add_table(self, table_id: int, lineskip_arr, binning_table):
        """
        Add new binning table definition to CKD

        Parameters
        ----------
        table_id :  int
        lineskip_arr :  ndarray
        binning_table :  ndarray
          Binning table as written to DEM
        """
        index, count = np.unique(binning_table[lineskip_arr == 1, :],
                                 return_counts=True)

        with Dataset(self.ckd_dir / self.ckd_file, 'r+') as fid:
            gid = fid.createGroup('/Table_{:03d}'.format(table_id))
            gid.enabled_lines = np.uint16(lineskip_arr.sum())
            gid.flex_binned_pixels = np.uint32(index.max()+1)

            dset = gid.createVariable('binning_table', 'u4', ('row', 'column'),
                                      fill_value=FILL_VALUE,
                                      chunksizes=(128, 128),
                                      zlib=True, complevel=1, shuffle=True)
            dset.long_name = 'binning table'
            dset.valid_min = np.uint32(0)
            dset.valid_max = np.uint32(index.max())
            dset[:] = binning_table

            dset = gid.createVariable('lineskip_arr', 'u1', ('row',),
                                      zlib=True, complevel=1, shuffle=True)
            dset.long_name = 'lineskip array'
            dset.valid_min = np.uint8(0)
            dset.valid_max = np.uint8(1)
            dset[:] = lineskip_arr

            gid.createDimension('bins', count.size)
            dset = gid.createVariable('count_table', 'u2', ('bins',),
                                      zlib=True, complevel=1, shuffle=True)
            dset.long_name = 'number of aggregated pixel readings'
            dset.valid_min = np.uint16(0)
            dset.valid_max = np.uint16(count.max())
            dset[:] = count.astype('u2')

    def unbin(self, table_id: int, img_binned):
        """
        Reverse binning of detector data
        """
        with Dataset(self.ckd_dir / self.ckd_file, 'r') as fid:
            gid = fid['/Table_{:03d}'.format(table_id)]
            binning_table = gid.variables['binning_table'][:]
            lineskip_arr = gid.variables['lineskip_arr'][:]
            count_table = gid.variables['count_table'][:]

        revert = np.full(binning_table.shape, np.nan)
        table = binning_table[lineskip_arr == 1, :].reshape(-1)
        revert[lineskip_arr == 1, :] = \
            (img_binned / count_table)[table].reshape(-1, 1024)

        return revert
        
