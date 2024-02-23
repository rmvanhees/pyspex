#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2024 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Tools to read or write definitions of SPEXone binning-tables."""

from __future__ import annotations

__all__ = ["BinningTables"]

from datetime import UTC, datetime
from importlib.resources import files

import numpy as np

# pylint: disable=no-name-in-module
from netCDF4 import Dataset

from .lib import pyspex_version

# - global parameters ------------------------------
FILL_VALUE = 0xFFFFFFFF  # 0X7FFFFFFF


# - local functions --------------------------------
class BinningTables:
    """Class to handle SPEXone binning-table definitions.

    Raises
    ------
    FileNotFoundError
        Directory with SPEXone binning-table files does not exist.

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
    Therefore, it is preferred that a new binning table is added to the
    current set, without changing the validity start string.
    However, new binning-table file should be released in case any of the
    binning tables are overwritten.

    Examples
    --------
    # create new file with binning-table definitions::

    > bin_tbl = BinningTables()
    > bin_tbl.create_if_needed(validity_start)
    > bin_tbl.add_table(0, lineskip_arr, binning_table)
    > bin_tbl.add_table(1, lineskip_arr, binning_table)

    # use binning-table '130' to unbin SPEXone detector data::

    > bin_tbl = BinningTables(130)
    > img_1d = bin_tbl.unbin(img_binned)
    > img_2d = bin_tbl.to_image(img_1d)

    """

    def __init__(self: BinningTables, table_id: int | None = None) -> None:
        """Initialize class attributes."""
        self.bin_tbl = files("pyspex.data").joinpath("binning_tables.nc")
        if table_id is None:
            return

        if not self.bin_tbl.is_file():
            raise FileNotFoundError(f"{self.bin_tbl} not found")

        with Dataset(self.bin_tbl, "r") as fid:
            if f"Table_{table_id:03d}" not in fid.groups:
                raise KeyError(f"Table_{table_id:03d} not defined")
            gid = fid[f"Table_{table_id:03d}"]
            self.binning_table = gid.variables["binning_table"][:]
            self.lineskip_arr = gid.variables["lineskip_arr"][:]
            self.count_table = gid.variables["count_table"][:]

    def unbin(self: BinningTables, img_binned: np.ndarray) -> np.ndarray:
        """Return detector data corrected for flexible and fixed binning (still 1-D).

        Parameters
        ----------
        img_binned : np.ndarray
           Binned image data (1D array)

        Returns
        -------
        np.ndarray
           unbinned image data

        """
        mask = self.count_table > 5
        img_binned[mask] = np.nan
        img_binned[~mask] /= (4 * self.count_table[~mask])
        return img_binned

    def to_image(self: BinningTables, img_1d: np.ndarray) -> np.ndarray:
        """Return unbinned detector data.

        Parameters
        ----------
        img_1d : np.ndarray
           unbinned image data (1-D array)

        Returns
        -------
        np.ndarray
           2x2 image (2-D array).

        """
        revert = np.full(self.binning_table.shape, np.nan)
        table = self.binning_table[self.lineskip_arr == 1, :].reshape(-1)
        revert[self.lineskip_arr == 1, :] = img_1d[table].reshape(-1, 1024)
        return revert

    def create_if_needed(
        self: BinningTables, validity_start: str, release: int = 1
    ) -> None:
        """Initialize CKD file for binning tables if not exist.

        Parameters
        ----------
        validity_start: str
           Validity start of the CKD data, as ``yyyymmddTHHMMSS``
        release :  int, default=1
           Release number, start at 1

        """
        # initialize netCDF file with binning tables
        with Dataset("binning_table.nc", "w") as fid:
            fid.title = "SPEXone Level-1 binning-tables"
            fid.Conventions = "CF-1.6"
            fid.project = "PACE Project"
            fid.instrument = "SPEXone"
            fid.institution = "SRON Netherlands Institute for Space Research"
            fid.processing_version = pyspex_version()
            fid.validity_start = validity_start + "+00:00"
            fid.release_number = np.uint16(release)
            fid.date_created = datetime.now(UTC).isoformat(timespec="seconds")

            fid.createDimension("row", 1024)
            fid.createDimension("column", 1024)

    def add_table(
        self: BinningTables,
        table_id: int,
        lineskip_arr: np.ndarray,
        binning_table: np.ndarray,
    ) -> None:
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
        index, count = np.unique(
            binning_table[lineskip_arr == 1, :], return_counts=True
        )

        with Dataset(self.bin_tbl, "r+") as fid:
            gid = fid.createGroup(f"/Table_{table_id:03d}")
            gid.tabel_id = table_id
            gid.REG_BINNING_TABLE_START = hex(0x80000000 + 0x400000 * (table_id - 1))
            gid.enabled_lines = np.uint16(lineskip_arr.sum())
            gid.flex_binned_pixels = np.uint32(index.max() + 1)
            gid.date_created = datetime.now(UTC).isoformat(timespec="seconds")

            dset = gid.createVariable(
                "binning_table",
                "u4",
                ("row", "column"),
                fill_value=FILL_VALUE,
                chunksizes=(128, 128),
                zlib=True,
                complevel=1,
                shuffle=True,
            )
            dset.long_name = "binning table"
            dset.valid_min = np.uint32(0)
            dset.valid_max = np.uint32(index.max())
            dset[:] = binning_table

            dset = gid.createVariable(
                "lineskip_arr", "u1", ("row",), zlib=True, complevel=1, shuffle=True
            )
            dset.long_name = "lineskip array"
            dset.valid_min = np.uint8(0)
            dset.valid_max = np.uint8(1)
            dset[:] = lineskip_arr

            gid.createDimension("bins", count.size)
            dset = gid.createVariable(
                "count_table", "u2", ("bins",), zlib=True, complevel=1, shuffle=True
            )
            dset.long_name = "number of aggregated pixel readings"
            dset.valid_min = np.uint16(0)
            dset.valid_max = np.uint16(count.max())
            dset[:] = count.astype("u2")
