#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2025 SRON
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

from .lib.ccsds_hdr import CCSDShdr

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
        coverage_nav: tuple[np.datetime64, np.datetime64],
        coverage_spx: tuple[np.datetime64, np.datetime64],
    ) -> int:
        """Check coverage time of navigation data.

        Parameters
        ----------
        coverage_nav: tuple[np.datetime64, np.datetime64]
           time-coverage of the PACE-HKT data
        coverage_spx: tuple[np.datetime64, np.datetime64]
           time-coverage of the SPEXone measurement data

        """
        coverage_quality = cls(0)

        # check at the start of the data
        if coverage_spx[0] < coverage_nav[0]:
            coverage_quality |= CoverageFlag.NO_EXTEND_AT_START
            module_logger.error(
                "time coverage of navigation data starts after L1A science data"
            )

        diff_coverage = coverage_spx[0] - coverage_nav[0]
        if diff_coverage <= SEVEN_SECONDS:
            coverage_quality |= CoverageFlag.TOO_SHORT_EXTENDS
            module_logger.warning(
                "extends of time coverage of navigation data too short",
            )

        # check at the end of the data
        if coverage_spx[1] > coverage_nav[1]:
            coverage_quality |= CoverageFlag.NO_EXTEND_AT_END
            module_logger.error(
                "time coverage of navigation data ends before L1A science data"
            )

        diff_coverage = coverage_nav[1] - coverage_spx[1]
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
        list of PACE_HKT filenames (netCDF4 format)

    Notes
    -----
    This class has the following methods::

     - housekeeping(instrument: str) -> tuple[np.ndarray]
     - navigation() -> dict

    """

    def __init__(self: HKTio, flnames: str | Path | list[Path]) -> None:
        """Initialize access to PACE HKT products."""
        self.time_coverage = ()
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
        """Read housekeeping telemetry data from the HKT products.

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

    def navigation(self: HKTio) -> dict[str, NDArray]:
        """Read navigation data from the HKT products."""
        nav_data = {}
        for hkt_file in self.flnames:
            with h5py.File(hkt_file) as fid:
                # pylint: disable=no-member
                gid = fid["/navigation_data"]
                for key in gid:
                    data = gid[key][:]
                    if key.endswith("_time"):
                        data = np.datetime64(
                            gid["att_time"].attrs["units"].decode().split(" ")[2]
                        ) + (1e6 * data).astype("timedelta64[us]")

                    nav_data[key] = (
                        data
                        if key not in nav_data
                        else np.append(nav_data[key], data, axis=0)
                    )

        # order data in time
        if not np.all(nav_data["att_time"][:-1] <= nav_data["att_time"][1:]):
            indx = np.argsort(nav_data["att_time"])
            for key in nav_data:
                if key.startswith("att_"):
                    nav_data[key] = nav_data[key][indx, ...]

        if not np.all(nav_data["orb_time"][:-1] <= nav_data["orb_time"][1:]):
            indx = np.argsort(nav_data["orb_time"])
            for key in nav_data:
                if key.startswith("orb_"):
                    nav_data[key] = nav_data[key][indx, ...]

        if not np.all(nav_data["tilt_time"][:-1] <= nav_data["tilt_time"][1:]):
            indx = np.argsort(nav_data["tilt_time"])
            for key in nav_data:
                if key.startswith("tilt"):
                    nav_data[key] = nav_data[key][indx, ...]

        # set time_coverage
        self.time_coverage = (nav_data["att_time"][0], nav_data["att_time"][-1])

        return nav_data

    def nav_coverage_adjust(
        self: HKTio,
        nav_data: dict[str, NDArray],
        coverage_spx: tuple(np.datetime64, np.datetime64),
    ) -> dict[str, NDArray] | None:
        """Return navigation data within SPEXone time-coverage extended by 10 sec."""
        coverage = (coverage_spx[0] - TEN_SECONDS, coverage_spx[1] + TEN_SECONDS)

        att_mask = (nav_data["att_time"] >= coverage[0]) & (
            nav_data["att_time"] <= coverage[1]
        )
        orb_mask = (nav_data["orb_time"] >= coverage[0]) & (
            nav_data["orb_time"] <= coverage[1]
        )
        tilt_mask = (nav_data["tilt_time"] >= coverage[0]) & (
            nav_data["tilt_time"] <= coverage[1]
        )

        nav_dict = {}
        for key, value in nav_data.items():
            if key.startswith("att_"):
                nav_dict[key] = value[att_mask, ...]
            elif key.startswith("orb_"):
                nav_dict[key] = value[orb_mask, ...]
            else:
                nav_dict[key] = value[tilt_mask, ...]

        # update time_coverage
        self.time_coverage = (nav_data["att_time"][0], nav_data["att_time"][-1])

        return nav_dict

    def nav_coverage_flag(
        self: HKTio,
        coverage_spx: tuple(np.datetime64, np.datetime64),
    ) -> int:
        """Check completeness of the navigation data."""
        return CoverageFlag.check(self.time_coverage, coverage_spx)
