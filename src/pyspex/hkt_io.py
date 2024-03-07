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

__all__ = ["CoverageFlag", "HKTio"]

import logging
from enum import IntFlag, auto
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import xarray as xr

from .lib.ccsds_hdr import CCSDShdr

if TYPE_CHECKING:
    import datetime as dt

# - global parameters -----------------------
module_logger = logging.getLogger("pyspex.hkt_io")

# reduce extend of navigation data to
TEN_SECONDS = np.timedelta64(10, "s")
# require an extent of navigation data of
SEVEN_SECONDS = np.timedelta64(7, "s")


# - class CoverageFlag ----------------------
class CoverageFlag(IntFlag):
    """Define flags for coverage_quality (navigation_data)."""

    MISSING_SAMPLES = auto()
    TOO_SHORT_EXTENDS = auto()
    NO_EXTEND_AT_START = auto()
    NO_EXTEND_AT_END = auto()

    @classmethod
    def check(
        cls: CoverageFlag,
        nav_data: xr.Dataset,
        coverage: list[dt.datetime, dt.datetime],
    ) -> int:
        """Check coverage time of navigation data."""
        coverage_quality = cls(0)
        l1a_time_range = (
            np.datetime64(coverage[0].replace(tzinfo=None)),
            np.datetime64(coverage[1].replace(tzinfo=None)),
        )
        nav_time_range = (
            nav_data["att_time"].values[0],
            nav_data["att_time"].values[-1],
        )

        # check at the start of the data
        if l1a_time_range[0] < nav_time_range[0]:
            coverage_quality |= CoverageFlag.NO_EXTEND_AT_START
            module_logger.error(
                "time coverage of navigation data starts after L1A science data"
            )

        diff_coverage = l1a_time_range[0] - nav_time_range[0]
        if diff_coverage <= SEVEN_SECONDS:
            coverage_quality |= CoverageFlag.TOO_SHORT_EXTENDS
            module_logger.warning(
                "extends of time coverage of navigation data too short",
            )

        # check at the end of the data
        if l1a_time_range[1] > nav_time_range[1]:
            coverage_quality |= CoverageFlag.NO_EXTEND_AT_END
            module_logger.error(
                "time coverage of navigation data ends before L1A science data"
            )

        diff_coverage = nav_time_range[1] - l1a_time_range[1]
        if diff_coverage <= SEVEN_SECONDS:
            coverage_quality |= CoverageFlag.TOO_SHORT_EXTENDS
            module_logger.warning(
                "extends of time coverage of navigation data too short",
            )
        return coverage_quality


# - class HKTio -------------------------
class HKTio:
    """Class to read housekeeping and navigation data from PACE-HKT products.

    Parameters
    ----------
    flnames :  str | Path | list[Path]
        sorted list of PACE_HKT filenames (netCDF4 format)

    Notes
    -----
    This class has the following methods::

     - housekeeping(instrument: str) -> tuple[np.ndarray]
     - navigation() -> dict

    """

    def __init__(self: HKTio, flnames: str | Path | list[Path]) -> None:
        """Initialize access to PACE HKT products."""
        self.nav_data: xr.Dataset | None = None
        if isinstance(flnames, str | Path):
            self.flnames = [flnames] if isinstance(flnames, Path) else [Path(flnames)]
        else:
            self.flnames = flnames

        for hkt_fl in self.flnames:
            if not hkt_fl.is_file():
                raise FileNotFoundError(f"file {hkt_fl} not found on system")

    @staticmethod
    def __read_hkt__(hkt_file: Path, instrument: str) -> np.ndarray | None:
        """Return housekeeping data of a given instrument."""
        with h5py.File(hkt_file) as fid:
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

    # ---------- PUBLIC FUNCTIONS ----------
    def housekeeping(self: HKTio, instrument: str = "spx") -> tuple[np.ndarray]:
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
        res = None
        for hkt_file in self.flnames:
            buff = self.__read_hkt__(hkt_file, instrument)
            if buff is not None:
                res = buff if res is None else np.concatenate((res, buff), axis=0)
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

    def navigation(self: HKTio) -> None:
        """..."""
        if self.nav_data is not None:
            self.nav_data.close()
            self.nav_data = None

        nav = None
        for hkt_file in self.flnames:
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
        nav.attrs["time_coverage_start"] = str(nav["att_time"].values[0])[:23] + "Z"
        nav.attrs["time_coverage_end"] = str(nav["att_time"].values[-1])[:23] + "Z"
        self.nav_data = nav

    def add_nav(
        self: HKTio, l1a_file: Path, coverage: list[dt.datetime, dt.datetime]
    ) -> None:
        """..."""
        time_range = (
            np.datetime64(coverage[0].replace(tzinfo=None)) - TEN_SECONDS,
            np.datetime64(coverage[1].replace(tzinfo=None)) + TEN_SECONDS,
        )
        att_indx = (
            (self.nav_data.att_time.values >= time_range[0])
            & (self.nav_data.att_time.values <= time_range[1])
        ).nonzero()[0]
        orb_indx = (
            (self.nav_data.orb_time.values >= time_range[0])
            & (self.nav_data.orb_time.values <= time_range[1])
        ).nonzero()[0]
        tilt_indx = (
            (self.nav_data.tilt_time.values >= time_range[0])
            & (self.nav_data.tilt_time.values <= time_range[1])
        ).nonzero()[0]
        nav = self.nav_data.isel(
            att_time=att_indx, orb_time=orb_indx, tilt_time=tilt_indx
        )
        self.nav_data.close()

        # add coverage-quality flag
        xarr = xr.DataArray(
            np.array([CoverageFlag.check(nav, coverage)]),
            name="coverage_quality",
            attrs={
                "long_name": "coverage quality of navigation data",
                "standard_name": "status_flag",
                "valid_range": np.array([0, 15], dtype="u2"),
                "flag_values": np.array([0, 1, 2, 4, 8], dtype="u2"),
                "flag_meanings": (
                    "good missing-samples too_short_extends no_extend_at_start"
                    " no_extend_at_end"
                ),
            },
        )
        nav["coverage_quality"] = xarr
        nav.to_netcdf(l1a_file, group="navigation_data", mode="a")
        nav.close()
