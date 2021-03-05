"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python class to handle SPEXone binning tables

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime, timezone
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
    ckd_dir : pathlib.Path
       Directory with binning table CKD
    ckd_file : str
       Filename of the binning table CKD

    Methods
    -------
    search_ckd()
       Search current CKD file with binning tables.
    create_ckd(release=1)
       Initialize CKD file with binning tables (netCDF4 format).
    add_table(table_id, lineskip_arr, binning_table)
       Add a binning table definition to existing CKD file.
    unbin(table_id: int, img_binned)
       Return unbinned detector data.

    Notes
    -----
    The binning tables as defined on-ground are supposed to be available
    during the whole mission at the same memory location. Because these
    original binning tables are necessary for re-processing and may
    facilitate instrument performance monitoring. New binning tables should
    be added to the CKD, this can be performed without changing the name of
    the CKD file. Actually, the name of the CKD file, which contains it
    creation date, should only be changed when the format of the file is
    changed. Therefore, the software should always use the CKD file with
    latest creation time in its name.

    In case any of the binning tables are overwritten or moved in memory, we
    could use the following strategy, where the 3 number digit in the CKD
    filename indicated from which MPS version it is valid. Therefore, this
    number is currently 001. Note, however, that this strategy is not part
    of the current S/W implementation.

    Examples
    --------
    ...
    """
    def __init__(self, mode='r', ckd_dir=None) -> None:
        """
        Initialize class attributes

        Parameters
        ----------
        mode : char
           Read/write mode, choices are 'r': read, 'w': write, 'a': append
        ckd_dir : str
           Name of directory for the binning table CKD

        Raises
        ------
        FileNotFoundError
           directory with SPEXone CKD does not exist
        KeyError
           read/write mode should be 'r', 'w' or 'a'
        """
        if ckd_dir is None:
            self.ckd_dir = Path('/nfs/SPEXone/share/ckd')
            if not self.ckd_dir.is_dir():
                self.ckd_dir = Path('/data/richardh/SPEXone/share/ckd')
        else:
            self.ckd_dir = Path(ckd_dir)
        if not self.ckd_dir.is_dir():
            raise FileNotFoundError('directory with SPEXone CKD does not exist')

        if mode in ('r', 'a'):
            self.search_ckd()
        elif mode == 'w':
            self.create_ckd()
        else:
            raise KeyError("read/write mode should be 'r', 'w' or 'a'")

    def search_ckd(self) -> None:
        """
        Search CKD file with binning tables

        Raises
        ------
        FileNotFoundError
           No CKD with binning tables found
        """
        ckd_files = list(Path(self.ckd_dir).glob('SPX1_OCAL_L1A_TBL_*.nc'))
        if not ckd_files:
            raise FileNotFoundError('No CKD with binning tables found')

        # only read the latest version of the binning-table CKD
        self.ckd_file = sorted(ckd_files)[-1]

    def create_ckd(self, release=1) -> None:
        """
        Initialize CKD file with binning tables (netCDF4 format)

        Parameters
        ----------
        release :  int, default=1
           Minimum version of the MPS for which the binning tables are valid
        """
        # initialize netCDF file with binning tables
        self.ckd_file = 'SPX1_OCAL_L1A_TBL_{:s}_{:03d}.nc'.format(
            datetime.utcnow().strftime("%Y%m%dT%H%M%S"), release)

        with Dataset(self.ckd_dir / self.ckd_file, 'w') as fid:
            fid.title = 'SPEXone Level-1 binning-tables'
            fid.Convensions = 'CF-1.6'
            fid.project = 'PACE Project'
            fid.instrument = 'SPEXone'
            fid.institution = 'SRON Netherlands Institute for Space Research'
            fid.processing_version = version.get()
            fid.date_created = datetime.now(timezone.utc).isoformat(
                timespec='seconds')

            fid.createDimension('row', 1024)
            fid.createDimension('column', 1024)

    def add_table(self, table_id: int, lineskip_arr, binning_table) -> None:
        """
        Add a binning table definition to existing CKD file

        Parameters
        ----------
        table_id :  int
          Table identifier (integer between 1 and 255)
        lineskip_arr :  ndarray
          Lineskip array definition
        binning_table :  ndarray
          Binning table definition
        """
        index, count = np.unique(binning_table[lineskip_arr == 1, :],
                                 return_counts=True)

        with Dataset(self.ckd_dir / self.ckd_file, 'r+') as fid:
            gid = fid.createGroup('/Table_{:03d}'.format(table_id))
            gid.enabled_lines = np.uint16(lineskip_arr.sum())
            gid.flex_binned_pixels = np.uint32(index.max()+1)
            gid.date_created = datetime.now(timezone.utc).isoformat(
                timespec='seconds')

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
        Return unbinned detector data

        Parameters
        ----------
        table_id :  int
           Table identifier (integer between 1 and 255)
        img_binned : ndarray
           Binned image data

        Returns
        -------
           Unbinned image data (no interpolation)
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
