#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Tools to read or write definitions of SPEXone binning-tables."""

from __future__ import annotations

__all__ = ["BinningTables"]

import datetime as dt
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Self

import h5py
import numpy as np

from .lib import pyspex_version

if TYPE_CHECKING:
    from numpy.typing import NDArray


# - global parameters ------------------------------
DATA_DIR_CSV = Path("/nfs/SPEXone/share/binning")
# DATA_DIR_CSV = Path("/data/richardh/SPEXone/share/binning")

# name of the output file with binning-table definitions
NAME_BIN_TBL = Path("./binning_tables.nc")


# - class BinningCKD -------------------------------
class BinningCKD:
    """Class to store the SPEXone binning-table definitions.

    Parameters
    ----------
    overwrite :  bool, default=False
       Any existing file may be overwritten

    Raises
    ------
    FileNotFoundError
        Directory with SPEXone binning-table CSV files does not exist.

    Notes
    -----
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

    > with BinningCKD(overwrite=True) with bin_ckd:
    >    bin_ckd.add_tbl_pre_launch()
    >    bin_ckd.add_tbl_post_launch()

    """

    def __init__(self: BinningCKD, overwrite: bool = False) -> None:
        """Initialize new file to store SPEXone binning-table."""
        self.fid = None
        if NAME_BIN_TBL.is_file() and not overwrite:
            print("Please use parameter 'overwrite' to replace existing CKD.")
            return

        fid = h5py.File(NAME_BIN_TBL, "w")
        fid.attrs["title"] = "SPEXone Level-1 binning-tables"
        fid.attrs["creator_name"] = "SRON/Earth"
        fid.attrs["creator_email"] = "SPEXone-MPC@sron.nl"
        fid.attrs["creator_url"] = "https://www.sron.nl/missions-earth/pace-spexone"
        fid.attrs["institution"] = "SRON"
        fid.attrs["publisher_name"] = "SRON/Earth"
        fid.attrs["publisher_email"] = "SPEXone-MPC@sron.nl"
        fid.attrs["publisher_url"] = "https://www.sron.nl/missions-earth/pace-spexone"
        fid.attrs["project"] = "PACE Project"
        fid.attrs["conventions"] = "CF-1.10 ACDD-1.3"
        fid.attrs["keyword_vocabulary"] = (
            "NASA Global Change Master Directory (GCMD) Science Keywords"
        )
        fid.attrs["stdname_vocabulary"] = (
            "NetCDF Climate and Forecast (CF) Metadata Convention"
        )
        fid.attrs["title"] = "PACE SPEXone Level-1A Data"
        fid.attrs["platform"] = "PACE"
        fid.attrs["instrument"] = "SPEXone"
        fid.processing_version = pyspex_version()
        fid.attrs["date_created"] = (
            dt.datetime.now(dt.UTC).replace(tzinfo=None).isoformat(timespec="seconds")
        )
        dset = fid.create_dataset("column", dtype="u2", shape=(1024,))
        dset.make_scale()
        dset = fid.create_dataset("row", dtype="u2", shape=(1024,))
        dset.make_scale()
        self.fid = fid

    def __enter__(self: BinningCKD) -> Self:
        """Initiate the context manager."""
        return self

    def __exit__(self: BinningCKD, *args: object) -> bool:
        """Exit the context manager."""
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self: BinningCKD) -> None:
        """Close resources, after sanity check of SQLite and HDF5 databases."""
        if self.fid:
            # print(f"CLOSE HDF5 file: {self.fid.filename}")
            self.fid.close()

    @staticmethod
    def bin_tbl_defs() -> dict[int, tuple[str, str]]:
        """Return listing of binning-table definitions.

        Returns
        -------
        dict[int, tuple[str, str]]
            binning-table ID, names of files with lineskip and flexible binning

        """
        return {
            20: ("lineskiparrays_EXT_9p8_FM_0.csv", "binning_EXT_9p8_FM_0_add.csv"),
            24: ("lineskiparrays_EXT_9p8_FM_4.csv", "binning_EXT_9p8_FM_4_add.csv"),
            29: ("lineskiparrays_EXT_9p8_FM_8.csv", "binning_EXT_9p8_FM_8_add.csv"),
            50: ("lineskiparrays_170_offs2_2.csv", "binning_170_offs2_2_add_1.csv"),
            51: ("lineskiparrays_170_offs2_2.csv", "binning_170_offs2_2_add_2.csv"),
            52: ("lineskiparrays_170_offs2_2.csv", "binning_170_offs2_2_add_3.csv"),
            53: ("lineskiparrays_170_offs2_2.csv", "binning_170_offs2_2_add_4.csv"),
            54: ("lineskiparrays_170_offs2_2.csv", "binning_170_offs2_2_add_5.csv"),
            55: ("lineskiparrays_170_offs2_2.csv", "binning_170_offs2_2_add_6.csv"),
            56: ("lineskiparrays_160_views.csv", "binning_160_views_add_1.csv"),
            57: ("lineskiparrays_160_views.csv", "binning_160_views_add_2.csv"),
            58: ("lineskiparrays_160_views.csv", "binning_160_views_add_3.csv"),
            59: ("lineskiparrays_160_views.csv", "binning_160_views_add_4.csv"),
            60: ("lineskiparrays_160_views.csv", "binning_160_views_add_5.csv"),
        }

    def __write__(self: BinningCKD, data_dir: Path, group: str | None = None) -> None:
        """Write binning-table definition to a HDF5 file.

        Parameters
        ----------
        data_dir :  Path
            Directory with the binning-table definitions
        group :  str, optional
            root in the HDF5 file where the data will be stored

        """
        for tbl_id, (line_fl, bin_fl) in self.bin_tbl_defs().items():
            lineskip_data = np.loadtxt(data_dir / line_fl, delimiter=",").astype("u1")
            binning_data = np.loadtxt(data_dir / bin_fl, delimiter=",", dtype="u4")
            iadd = (data_dir / bin_fl).stem.split("_")[-1]
            iadd = 1 if iadd == "add" else int(iadd)
            lineskip_data = lineskip_data[iadd, :]
            index, count = np.unique(
                binning_data[lineskip_data == 1, :], return_counts=True
            )
            gid = self.fid.create_group(
                str(Path("/" if group is None else group) / f"Table_{tbl_id:03d}")
            )
            gid.attrs["tabel_id"] = tbl_id
            gid.attrs["REG_BINNING_TABLE_START"] = hex(
                0x80000000 + 0x400000 * (tbl_id - 1)
            )
            gid.attrs["enabled_lines"] = np.uint16(lineskip_data.sum())
            gid.attrs["flex_binned_pixels"] = np.uint32(index.max() + 1)

            dset = gid.create_dataset("bins", dtype="u2", shape=(count.size,))
            dset.make_scale()

            dset = gid.create_dataset(
                "count_table",
                count.shape,
                dtype="u2",
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            dset.dims[0].attach_scale(gid["bins"])
            dset.attrs["long_name"] = "number of aggregated pixel readings"
            dset.attrs["valid_min"] = np.uint16(0)
            dset.attrs["valid_max"] = np.uint16(count.max())
            dset[:] = count.astype("u2")

            dset = gid.create_dataset(
                "lineskip_arr",
                lineskip_data.shape,
                dtype="u1",
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            dset.dims[0].attach_scale(self.fid["row"])
            dset.attrs["long_name"] = "lineskip array"
            dset.attrs["valid_min"] = np.uint8(0)
            dset.attrs["valid_max"] = np.uint8(1)
            dset[:] = lineskip_data

            dset = gid.create_dataset(
                "binning_table",
                binning_data.shape,
                dtype="u4",
                fillvalue=np.iinfo("u4").max,
                chunks=(128, 128),
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            dset.dims[0].attach_scale(self.fid["row"])
            dset.dims[1].attach_scale(self.fid["column"])
            dset.attrs["long_name"] = "binning table"
            dset.attrs["valid_min"] = np.uint32(0)
            dset.attrs["valid_max"] = np.uint32(index.max())
            dset[:] = binning_data

    def add_tbl_pre_launch(self: BinningCKD) -> None:
        """Add tables with flexible binning used before launch."""
        data_dir = DATA_DIR_CSV / "20210208T152000"
        if not data_dir.is_dir():
            raise FileNotFoundError(f"can not find folder: {data_dir}")
        gid = self.fid.create_group("20210208T152000")
        gid.attrs["validity_date"] = str(np.datetime64("2021-02-08T15:20:00"))
        self.__write__(data_dir, "20210208T152000")

    def add_tbl_post_launch(self: BinningCKD) -> None:
        """Add tables with flexible binning used after launch."""
        data_dir = DATA_DIR_CSV / "20210304T124000"
        if not data_dir.is_dir():
            raise FileNotFoundError(f"can not find folder: {data_dir}")
        self.fid.attrs["validity_date"] = str(np.datetime64("2021-03-04T12:40:00"))
        self.__write__(data_dir)


# - class BinningTables ----------------------------
class BinningTables:
    """Class to apply the SPEXone binning-tables.

    Parameters
    ----------
    table_id :  int, optional
       Binning table ID number between one and 255
    coverage_start :  np.datetime64, optional
       Default is to return the table definitions as used after 2021-03-04

    Examples
    --------
    # use binning-table '130' to unbin SPEXone detector data::

    > with BinningTables(130) as bin_tbl:
    >    img_2d = bin_tbl.to_image(img_binned)

    """

    def __init__(
        self: BinningTables,
        table_id: int | None = None,
        *,
        coverage_start: np.datetime64 | None = None,
    ) -> None:
        """Initialize class attributes."""
        binning_db = files("pyspex.Data").joinpath("binning_tables.nc")
        if table_id is None:
            return

        if not binning_db.is_file():
            raise FileNotFoundError(f"{binning_db} not found")

        with h5py.File(binning_db) as fid:
            if coverage_start is None or coverage_start > np.datetime64(
                "2021-03-04T12:40:00"
            ):
                ds_name = f"Table_{table_id:03d}"
            else:
                ds_name = f"/20210208T152000/Table_{table_id:03d}"
            if ds_name not in fid:
                raise KeyError(f"{ds_name} not defined")
            self.binning_table = fid[f"{ds_name}/binning_table"][:]
            self.lineskip_arr = fid[f"{ds_name}/lineskip_arr"][:]
            self.count_table = fid[f"{ds_name}/count_table"][:]

    def __enter__(self: BinningCKD) -> Self:
        """Initiate the context manager."""
        return self

    def __exit__(self: BinningCKD, *args: object) -> bool:
        """Exit the context manager."""
        return False  # any exception is raised by the with statement.

    def _unbin_(self: BinningTables, img_1d: NDArray) -> NDArray[float]:
        """Return detector data corrected for number of flexible binnings (still 1-D).

        Parameters
        ----------
        img_1d :  NDArray
           Binned image data (N * 1-D array) where N is number of images

        Returns
        -------
        NDArray[float]
           unbinned image data (N * 1-D array)

        Examples
        --------
        Let's read 2 executions of measurements with MPS=47 from a SPEXone L1A product:

          > spx = SPXtlm()
          > spx.from_l1a(l1a_product, tlm_type='sci', mps_id=47)
          > print(len(spx.science.images), spx.science.images[0].shape)
          2, (14, 203500)
          > bin_tbl = BinningTables(spx.science.binning_table())
          > img_1d = bin_tbl.unbin(spx.science.images)
          > print(img_1d.shape)
          (28, 203500)

        """
        img_1d = img_1d.astype(float)
        img_1d /= self.count_table
        mask = self.count_table > 5

        if img_1d.ndim == 2:
            img_1d[:, mask] = np.nan
            return img_1d

        img_1d[mask] = np.nan
        return img_1d

    def to_image(
        self: BinningTables, img_binned: NDArray | tuple[NDArray], unbin: bool = True
    ) -> NDArray:
        """Return 2x2 binned 2-D detector image data.

        Parameters
        ----------
        img_binned : NDArray | tuple[NDArray]
           image data (N * 1-D array)
        unbin :  bool, default=True
           return image data as floats corrected for flexible binning

        Returns
        -------
        NDArray
           2x2 binned image with shape=(N, 1024, 1024)

        """
        if isinstance(img_binned, tuple):
            img_1d = np.concatenate(img_binned, axis=0)
        else:
            img_1d = np.asarray(img_binned)

        if unbin:
            img_1d = self._unbin_(img_1d)

        fill_value = (
            np.nan
            if np.issubdtype(img_1d.dtype, np.floating)
            else np.iinfo(img_1d.dtype).max
        )
        table = self.binning_table[self.lineskip_arr == 1, :].reshape(-1)
        if img_1d.ndim == 2:
            new_shape = (img_1d.shape[0], *self.binning_table.shape)
            revert = np.full(new_shape, fill_value)
            revert[:, self.lineskip_arr == 1, :] = img_1d[:, table].reshape(
                img_1d.shape[0], -1, 1024
            )
        else:
            revert = np.full(self.binning_table.shape, fill_value)
            revert[self.lineskip_arr == 1, :] = img_1d[table].reshape(-1, 1024)

        return revert

    def to_binned(
        self: BinningTables,
        image: NDArray | tuple[NDArray],
    ) -> NDArray[np.uint16]:
        """Return a 1-D array binned according to the current binning-table.

        Parameters
        ----------
        image :  NDArray
           2-D detector image with shape (N, 1024, 1024) or (N, 2024, 2024)

        Returns
        -------
        NDarray[uint16]
           binned 1-D image data

        """
        if isinstance(image, tuple):
            img_2d = np.concatenate(image, axis=0)
        else:
            img_2d = np.asarray(image)
        n_img = img_2d.shape[0]

        # perform 2x2 binning
        if img_2d.shape[1] == img_2d.shape[2] == 2048:
            img_2d = np.sum(img_2d.reshape(1024, 2, 1024, 2), axis=(1, 3))

        # pylint: disable=no-member
        binning_table = self.binning_table.reshape(-1)

        # perform flexible binning
        buf_1d = np.full((n_img, self.count_table.size), np.nan)
        uvals, counts = np.unique(self.binning_table, return_counts=True)

        # start_time = time.time()
        mask = counts == 1
        mask2 = np.isin(binning_table, uvals[mask])
        buf_1d[:, uvals[mask]] = img_2d.reshape(n_img, -1)[:, mask2]
        # print(f"processing count==1 took: {time.time() - start_time:.3f} sec.")

        # start_time = time.time()
        mask = counts == 2
        mask2 = np.isin(binning_table, uvals[mask])
        buf_1d[:, uvals[mask]] = np.sum(
            img_2d.reshape(n_img, -1)[:, mask2].reshape(n_img, -1, 2), axis=2
        )
        # print(f"processing count==2 took: {time.time() - start_time:.3f} sec.")

        # start_time = time.time()
        mask = counts == 3
        # pylint: disable=no-member
        mask2 = np.isin(binning_table, uvals[mask])
        buf_1d[:, uvals[mask]] = np.sum(
            img_2d.reshape(n_img, -1)[:, mask2].reshape(n_img, -1, 3), axis=2
        )
        # print(f"processing count==3 took: {time.time() - start_time:.3f} sec.")

        # return array with data-type uint16
        buf_1d[np.isnan(buf_1d)] = np.iinfo("u2").max
        return buf_1d.astype("u2")
