#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
This module contains the class `BinningTables` to deal with SPEXone
binning-tables or to generate a file with binning-table definitions.
"""
__all__ = ['BinningTables']

from datetime import datetime, timezone
from os import environ
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from pyspex import version


# - global parameters ------------------------------
FILL_VALUE = 0xFFFFFFFF  # 0X7FFFFFFF


# - local functions --------------------------------
class BinningTables:
    """Class to handle SPEXone binning-table definitions.

    Parameters
    ----------
    ckd_dir : str
        Specify the name of directory with SPEXone binning-table files.

    Raises
    ------
    FileNotFoundError
        Directory with SPEXone SPEXone binning-table files does not exist.

    Notes
    -----
    Syntax of the file name with SPEXone binning-tables:

          SPX1_CKD_BIN_TBL_<yyyymmddTHHMMSS>_<NNN>.nc

    where yyyymmddTHHMMSS defines the validity start (UTC) and NNN the
    release number of the file format.

    The binning tables as defined on-ground are supposed to be available
    during the whole mission at the same on-board memory location.
    Because these original binning tables are necessary for re-processing
    and may facilitate instrument performance monitoring.
    Therefore, it is prefered that a new binning table is added to the
    current set, without changing the validity start string.
    However, new binning-table file should be released in case any of the
    binning tables are overwritten.

    Examples
    --------
    # create new file with binning-table definitions:

    >>> bin_tbl = BinningTables()
    >>> bin_tbl.create_if_needed(validity_start)
    >>> bin_tbl.add_table(0, lineskip_arr, binning_table)
    >>> bin_tbl.add_table(1, lineskip_arr, binning_table)

    # add a new binning-table to an existing file:

    >>> bin_tbl BinningTables()
    >>> bin_tbl.create_if_needed(validity_start)
    >>> bin_tbl.add_table(2, lineskip_arr, binning_table)

    # use binning-table '130' to unbin SPEXone detector data

    >>> bin_tbl BinningTables()
    >>> bin_tbl.search(coverage_start)
    >>> img = bin_tbl.unbin(130, img_binned)

    """
    def __init__(self, ckd_dir=None) -> None:
        """Initialize class attributes.
        """
        if ckd_dir is None:
            self.ckd_dir = Path('/nfs/SPEXone/share/ckd')
            if not self.ckd_dir.is_dir():
                self.ckd_dir = Path(environ.get('CKD_DIR', '.'))
        else:
            self.ckd_dir = Path(ckd_dir)
        if not self.ckd_dir.is_dir():
            raise FileNotFoundError('directory with SPEXone CKD does not exist')

        self.ckd_file = None

    def create_if_needed(self, validity_start: str, release=1) -> None:
        """Initialize CKD file for binning tables if not exist.

        Parameters
        ----------
        validity_start: str
           Validity start of the CKD data, as ``yyyymmddTHHMMSS``
        release :  int, default=1
           Release number, start at 1
        """
        self.ckd_file = f'SPX1_CKD_BIN_TBL_{validity_start}_{release:03d}.nc'

        if (self.ckd_dir / self.ckd_file).is_file():
            return

        # initialize netCDF file with binning tables
        with Dataset(self.ckd_dir / self.ckd_file, 'w') as fid:
            fid.title = 'SPEXone Level-1 binning-tables'
            fid.Convensions = 'CF-1.6'
            fid.project = 'PACE Project'
            fid.instrument = 'SPEXone'
            fid.institution = 'SRON Netherlands Institute for Space Research'
            fid.processing_version = version.get()
            fid.validity_start = validity_start + '+00:00'
            fid.release_number = np.uint16(release)
            fid.date_created = datetime.now(timezone.utc).isoformat(
                timespec='seconds')

            fid.createDimension('row', 1024)
            fid.createDimension('column', 1024)

    def search(self, coverage_start=None) -> None:
        """Search CKD file with binning tables.

        Parameters
        ----------
        coverage_start : str, default=None
           time_coverage_start or start of the measurement (UTC)

        Raises
        ------
        FileNotFoundError
           No CKD with binning tables found
        """
        ckd_files = list(Path(self.ckd_dir).glob('SPX1_CKD_BIN_TBL_*.nc'))
        if not ckd_files:
            raise FileNotFoundError('No CKD with binning tables found')
        ckd_files = [x.name for x in ckd_files]

        # use the latest version of the binning-table CKD
        if coverage_start is None:
            self.ckd_file = sorted(ckd_files)[-1]
            return

        # use binning-table CKD based on coverage_start
        coverage_date = datetime.fromisoformat(coverage_start)
        for ckd_fl in sorted(ckd_files, reverse=True):
            validity_date = datetime.strptime(ckd_fl.split('_')[4] + '+00:00',
                                              '%Y%m%dT%H%M%S%z')
            if validity_date < coverage_date:
                self.ckd_file = ckd_fl
                break
        else:
            raise FileNotFoundError('No valid CKD with binning tables found')

    def add_table(self, table_id: int, lineskip_arr, binning_table) -> None:
        """Add a binning table definition to existing file.

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
            gid = fid.createGroup(f'/Table_{table_id:03d}')
            gid.tabel_id = table_id
            gid.REG_BINNING_TABLE_START = hex(0x80000000
                                              + 0x400000 * (table_id - 1))
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
        """Return unbinned detector data.

        Parameters
        ----------
        table_id :  int
           Table identifier (integer between 1 and 255)
        img_binned : np.ndarray
           Binned image data (1D array)

        Returns
        -------
        numpy.ndarray
           Unbinned image data (no interpolation).
        """
        with Dataset(self.ckd_dir / self.ckd_file, 'r') as fid:
            if f'Table_{table_id:03d}' not in fid.groups:
                raise KeyError(f'Table_{table_id:03d} not defined')
            gid = fid[f'Table_{table_id:03d}']
            binning_table = gid.variables['binning_table'][:]
            lineskip_arr = gid.variables['lineskip_arr'][:]
            count_table = gid.variables['count_table'][:]

        revert = np.full(binning_table.shape, np.nan)
        table = binning_table[lineskip_arr == 1, :].reshape(-1)
        revert[lineskip_arr == 1, :] = \
            (img_binned / count_table)[table].reshape(-1, 1024)

        return revert
