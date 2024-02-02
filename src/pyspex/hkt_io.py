#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2024 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Contains the class `HKTio` to read PACE HKT products."""

from __future__ import annotations

__all__ = ["HKTio", "copy_hkt_nav", "check_coverage_nav"]

import datetime as dt
import logging
from enum import IntFlag, auto
from pathlib import Path

# from typing import TYPE_CHECKING
import h5py
import numpy as np
import numpy.typing as npt
import xarray as xr

# pylint: disable=no-name-in-module
from netCDF4 import Dataset

from .lib.ccsds_hdr import CCSDShdr
from .lib.leap_sec import get_leap_seconds

# if TYPE_CHECKING:

# - global parameters -----------------------
module_logger = logging.getLogger("pyspex.hkt_io")

EPOCH = dt.datetime(1958, 1, 1, tzinfo=dt.UTC)

# valid data coverage range
VALID_COVERAGE_MIN = dt.datetime(2021, 1, 1, tzinfo=dt.UTC)
VALID_COVERAGE_MAX = dt.datetime(2035, 1, 1, tzinfo=dt.UTC)

# expect the navigation data to extend at least 10 seconds
# w.r.t. time_coverage_start and time_coverage_end.


class CoverageFlag(IntFlag):
    """Define flags for coverage_quality (navigation_data)."""

    GOOD = 0
    MISSING_SAMPLES = auto()
    TOO_SHORT_EXTENDS = auto()
    NO_EXTEND_AT_START = auto()
    NO_EXTEND_AT_END = auto()


# - high-level r/w functions ------------
def copy_hkt_nav(hkt_list: tuple[Path, ...], l1a_file: Path) -> None:
    """Read/copy navigation data from one or more HKT products.

    Parameters
    ----------
    hkt_list : tuple[Path, ...]
       listing of PACE-HKT products collocated with SPEXone measurements
    l1a_file :  Path
       name of the SPEXone level-1A product

    """
    nav = None
    for hkt_file in sorted(hkt_list):
        if nav is None:
            nav = xr.open_dataset(hkt_file, group="navigation_data")
        else:
            res = ()
            buff = xr.open_dataset(hkt_file, group="navigation_data")
            for key, xarr in buff.data_vars.items():
                res += (xr.concat((nav[key], buff[key]), dim=xarr.dims[0]),)
            nav = xr.merge(res, combine_attrs="drop_conflicts")

    # fix coordinates
    if "att_time" in nav.data_vars:
        nav = nav.rename({"att_records": "att_time"}).set_coords(["att_time"])
    if "orb_time" in nav.data_vars:
        nav = nav.rename({"orb_records": "orb_time"}).set_coords(["orb_time"])
    if "tilt_time" in nav.data_vars:
        nav = nav.rename({"tilt_records": "tilt_time"}).set_coords(["tilt_time"])
    # clean-up Dataset attributes
    for key in list(nav.attrs):
        del nav.attrs[key]

    # add PACE navigation data to existing level-1A product.
    nav.to_netcdf(l1a_file, group="navigation_data", mode="a")
    nav.close()


def check_coverage_nav(l1a_file: Path) -> bool:
    """Check time coverage of navigation data.

    Parameters
    ----------
    l1a_file :  Path
       name of the SPEXone level-1A product

    """
    coverage_quality = CoverageFlag.GOOD

    # obtain time_coverage_range from the Level-1A product
    xds_l1a = xr.open_dataset(
        l1a_file,
        group="image_attributes",
        drop_variables=("icu_time_sec", "icu_time_subsec"),
    )
    module_logger.debug(
        "SPEXone measurement time-coverage: %s - %s",
        xds_l1a["image_time"].values[0],
        xds_l1a["image_time"].values[-1],
    )

    xds_nav = xr.open_dataset(l1a_file, group="navigation_data")
    module_logger.debug(
        "SPEXone navigation time-coverage: %s - %s",
        xds_nav["att_time"].values[0],
        xds_nav["att_time"].values[-1],
    )
    time_coverage_start = str(xds_nav["att_time"].values[0])[:23]
    time_coverage_end = str(xds_nav["att_time"].values[-1])[:23]

    # check at the start of the data
    if xds_l1a["image_time"].values[0] < xds_nav["att_time"].values[0]:
        coverage_quality |= CoverageFlag.NO_EXTEND_AT_START
        module_logger.error(
            "time coverage of navigation data starts after 'time_coverage_start'"
        )

    diff_coverage = xds_l1a["image_time"].values[0] - xds_nav["att_time"].values[0]
    if diff_coverage < np.timedelta64(10, "s"):
        coverage_quality |= CoverageFlag.TOO_SHORT_EXTENDS
        module_logger.warning(
            "time coverage of navigation data starts after 'time_coverage_start - %s'",
            np.timedelta64(10, "s"),
        )

    # check at the end of the data
    if xds_l1a["image_time"].values[-1] > xds_nav["att_time"].values[-1]:
        coverage_quality |= CoverageFlag.NO_EXTEND_AT_END
        module_logger.error(
            "time coverage of navigation data ends before 'time_coverage_end'"
        )

    diff_coverage = xds_nav["att_time"].values[-1] - xds_l1a["image_time"].values[-1]
    if diff_coverage < np.timedelta64(10, "s"):
        coverage_quality |= CoverageFlag.TOO_SHORT_EXTENDS
        module_logger.warning(
            "time coverage of navigation data ends before 'time_coverage_end + %s'",
            np.timedelta64(10, "s"),
        )

    # ToDo: check for completeness
    # close interface
    xds_l1a.close()
    xds_nav.close()

    # add coverage flag and attributes to Level-1A product
    with Dataset(l1a_file, "a") as fid:
        gid = fid["/navigation_data"]
        gid.time_coverage_start = time_coverage_start
        gid.time_coverage_end = time_coverage_end
        dset = gid.createVariable("coverage_quality", "u1", fill_value=255)
        dset[:] = coverage_quality
        dset.long_name = "coverage quality of navigation data"
        dset.standard_name = "status_flag"
        dset.valid_range = np.array([0, 15], dtype="u2")
        dset.flag_values = np.array([0, 1, 2, 4, 8], dtype="u2")
        dset.flag_meanings = (
            "good missing-samples too_short_extends no_extend_at_start no_extend_at_end"
        )

    # generate warning if time-coverage of navigation data is too short
    if coverage_quality & CoverageFlag.TOO_SHORT_EXTENDS:
        return False

    return True


# - class HKTio -------------------------
class HKTio:
    """Class to read housekeeping and navigation data from PACE-HKT products.

    Parameters
    ----------
    filename : Path | str
        name of the PACE HKT product

    Notes
    -----
    This class has the following methods::

     - reference_date -> datetime
     - set_reference_date()
     - coverage() -> tuple[datetime, datetime]
     - housekeeping(instrument: str) -> tuple[np.ndarray, ...]
     - navigation() -> dict

    """

    def __init__(self: HKTio, filename: Path | str) -> None:
        """Initialize access to a PACE HKT product."""
        self._reference_date: dt.datetime | None = None
        self.filename: Path = filename if isinstance(filename, Path) else Path(filename)
        if not self.filename.is_file():
            raise FileNotFoundError(f"file {filename} not found")
        self.set_reference_date()

    # ---------- PUBLIC FUNCTIONS ----------
    @property
    def reference_date(self: HKTio) -> dt.datetime:
        """Return reference date of all time_of_day variables."""
        return self._reference_date

    def set_reference_date(self: HKTio) -> None:
        """Set reference date of current PACE HKT product."""
        ref_date = None
        with h5py.File(self.filename) as fid:
            grp = fid["navigation_data"]
            if "att_time" in grp and "units" in grp["att_time"].attrs:
                # pylint: disable=no-member
                words = grp["att_time"].attrs["units"].decode().split(" ")
                if len(words) > 2:
                    # Note timezone 'Z' is only accepted by Python 3.11+
                    ref_date = dt.datetime.fromisoformat(words[2] + "T00:00:00+00:00")

        if ref_date is None:
            coverage = self.coverage()
            ref_date = dt.datetime.combine(
                coverage[0].date(), dt.time(0), tzinfo=dt.UTC
            )

        self._reference_date = ref_date

    def read_hk_dset(self: HKTio, instrument: str) -> np.ndarray | None:
        """Return housekeeping data of a given instrument."""
        with h5py.File(self.filename) as fid:
            gid = fid["housekeeping_data"]
            if instrument == "spx":
                ds_name = (
                    "SPEX_HKT_packets"
                    if "SPEX_HKT_packets" in gid
                    else "SPEXone_HKT_packets"
                )
            elif instrument == "sc":
                ds_name = "SC_HKT_packets"
            elif instrument == "oci":
                ds_name = "OCI_HKT_packets"
            elif instrument == "harp":
                ds_name = (
                    "HARP_HKT_packets"
                    if "HARP_HKT_packets" in gid
                    else "HARP2_HKT_packets"
                )
            else:
                raise KeyError("data of unknown instrument requested")

            res = gid[ds_name][:] if ds_name in gid else None

        return res

    def coverage(self: HKTio) -> list[dt.datetime, dt.datetime]:
        """Return data coverage."""
        one_day = dt.timedelta(days=1)
        with h5py.File(self.filename) as fid:
            # pylint: disable=no-member
            # Note timezone 'Z' is only accepted by Python 3.11+
            val = fid.attrs["time_coverage_start"].decode()
            coverage_start = dt.datetime.fromisoformat(val.replace("Z", "+00:00"))
            val = fid.attrs["time_coverage_end"].decode()
            coverage_end = dt.datetime.fromisoformat(val.replace("Z", "+00:00"))

        if abs(coverage_end - coverage_start) < one_day:
            return [coverage_start, coverage_end]

        module_logger.warning("attributes time_coverage_* are not present or invalid")
        # derive time_coverage_start/end from spacecraft telemetry
        res = self.read_hk_dset("sc")
        dt_list = ()
        for packet in res:
            try:
                ccsds_hdr = CCSDShdr()
                ccsds_hdr.read("raw", packet)
            except ValueError as exc:
                module_logger.warning('CCSDS header read error with "%s"', exc)
                break

            val = ccsds_hdr.tstamp(EPOCH)
            if (val > VALID_COVERAGE_MIN) & (val < VALID_COVERAGE_MAX):
                dt_list += (val,)

        dt_arr: npt.NDArray[dt.datetime] = np.array(dt_list)
        ii = dt_arr.size // 2
        leap_sec = get_leap_seconds(dt_arr[ii].timestamp(), epochyear=1970)
        dt_arr -= dt.timedelta(seconds=leap_sec)
        mn_val = min(dt_arr)
        mx_val = max(dt_arr)
        if mx_val - mn_val > one_day:
            indx_close_to_mn = (dt_arr - mn_val) <= one_day
            indx_close_to_mx = (mx_val - dt_arr) <= one_day
            module_logger.warning(
                "coverage_range: %s[%d] - %s[%d]",
                mn_val,
                np.sum(indx_close_to_mn),
                mx_val,
                np.sum(indx_close_to_mx),
            )
            if np.sum(indx_close_to_mn) > np.sum(indx_close_to_mx):
                mx_val = max(dt_arr[indx_close_to_mn])
            else:
                mn_val = min(dt_arr[indx_close_to_mx])

        return [mn_val, mx_val]

    def housekeeping(self: HKTio, instrument: str = "spx") -> tuple[np.ndarray, ...]:
        """Get housekeeping telemetry data.

        Parameters
        ----------
        instrument : {'spx', 'oci', 'harp', 'sc'}, default='spx'
           name of PACE instrument: 'harp': HARP2, 'oci': OCI,
           'sc': spacecraft, 'spx': SPEXone.

        Notes
        -----
        Current implementation only works for SPEXone.

        """
        res = self.read_hk_dset(instrument)
        if res is None:
            return ()

        ccsds_hk = ()
        for packet in res:
            try:
                ccsds_hdr = CCSDShdr()
                ccsds_hdr.read("raw", packet)
            except ValueError as exc:
                module_logger.warning('CCSDS header read error with "%s"', exc)
                break

            try:
                dtype_apid = ccsds_hdr.data_dtype
            except ValueError:
                print(
                    f"APID: 0x{ccsds_hdr.apid:x};"
                    f" Packet Length: {ccsds_hdr.packet_size:d}"
                )
                dtype_apid = None

            if dtype_apid is not None:  # all valid APIDs
                buff = np.frombuffer(packet, count=1, offset=0, dtype=dtype_apid)
                ccsds_hk += (buff,)
            else:
                module_logger.warning(
                    "package with APID 0x%x and length %d is not implemented",
                    ccsds_hdr.apid,
                    ccsds_hdr.packet_size,
                )

        return ccsds_hk
