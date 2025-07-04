# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Module with dedicated functions and classes to write a L1A product."""

from __future__ import annotations

__all__ = [
    "CoverageFlag",
    "SpexL1A",
    "check_input_files",
    "create_l1a",
    "nav_adjust",
    "read_hkt_nav",
]

import datetime as dt
import logging
import sys
from dataclasses import asdict
from enum import IntFlag, auto
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Self

import h5py
import numpy as np
from h5yaml.conf_from_yaml import conf_from_yaml
from h5yaml.yaml_h5py import H5Yaml

from pyspex.lib import pyspex_version
from pyspex.tlm import get_l1a_filename

if TYPE_CHECKING:
    from dataclasses import dataclass

    from numpy.typing import NDArray

    from pyspex.lib.hk_tlm import HKtlm
    from pyspex.lib.sci_tlm import SCItlm
    from pyspex.tlm import SPXtlm


# - global parameters ------------------------------
module_logger = logging.getLogger("pyspex.l1a")

# reduce extend of navigation data to
TEN_SECONDS = np.timedelta64(10, "s")
# require an extent of navigation data of
SEVEN_SECONDS = np.timedelta64(7, "s")


# - local functions --------------------------------
def check_input_files(config: dataclass) -> dataclass:
    """Check SPEXone level-0 files on existence and format.

    Parameters
    ----------
    config :  dataclass
       A dataclass that contains the settings of the L0-L1A processor

    Returns
    -------
    dataclass
       fields 'l0_format' {'raw', 'st3', 'dsb'} and 'l0_list' are updated.

    Raises
    ------
    FileNotFoundError
       If files are not found on the system.
    TypeError
       If determined file type differs from value supplied by user.

    """
    file_list = config.l0_list
    if file_list[0].suffix == ".H":
        if not file_list[0].is_file():
            raise FileNotFoundError(file_list[0])
        data_dir = file_list[0].parent
        file_stem = file_list[0].stem
        file_list = (
            sorted(data_dir.glob(file_stem + ".[0-9]"))
            + sorted(data_dir.glob(file_stem + ".?[0-9]"))
            + sorted(data_dir.glob(file_stem + "_hk.[0-9]"))
        )
        if not file_list:
            raise FileNotFoundError(file_stem + ".[0-9]")

        config.l0_format = "raw"
        config.l0_list = file_list
    elif file_list[0].suffix == ".ST3":
        if not file_list[0].is_file():
            raise FileNotFoundError(file_list[0])
        config.l0_format = "st3"
        config.l0_list = [file_list[0]]
    elif file_list[0].suffix == ".spx":
        file_list_out = []
        for flname in file_list:
            if not flname.is_file():
                raise FileNotFoundError(flname)

            if flname.suffix == ".spx":
                file_list_out.append(flname)

        if not file_list_out:
            raise FileNotFoundError(file_list)
        config.l0_format = "dsb"
        config.l0_list = file_list_out
    else:
        raise TypeError("Input files not recognized as SPEXone level-0 data")

    return config


def read_hkt_nav(hkt_list: list[Path]) -> dict[str, NDArray[float]]:
    """Read navigation data from a list of PACE_HKT files.

    Parameters
    ----------
    hkt_list :  list[Path]
        sorted list of PACE_HKT filenames (netCDF4 format)

    """
    res = {
        "/navigation_data/att_time": [],
        "/navigation_data/att_quat": [],
        "/navigation_data/att_rate": [],
        "/navigation_data/orb_time": [],
        "/navigation_data/orb_pos": [],
        "/navigation_data/orb_vel": [],
        "/navigation_data/orb_lat": [],
        "/navigation_data/orb_lon": [],
        "/navigation_data/orb_alt": [],
        "/navigation_data/tilt_time": [],
        "/navigation_data/tilt": [],
        "/navigation_data/tilt_flag": [],
    }
    if not hkt_list:
        for key in res:
            res[key] = np.array([])
        return res

    for hkt_file in hkt_list:
        with h5py.File(hkt_file) as fid:
            for key, value in res.items():
                if key.endswith("_time"):
                    midnight = fid[key].attrs["units"].split(b" ")[2].decode()
                    value.append(
                        np.datetime64(midnight)
                        + (1e6 * fid[key][:]).astype("timedelta64[us]")
                    )
                else:
                    value.append(fid[key][:])

    for key in res:
        res[key] = np.concatenate(res[key])

    return res


def nav_adjust(
    nav_dict: dict[str, NDArray[float]], coverage_in: list[dt.datetime, dt.datetime]
) -> dict[str, NDArray[float]] | None:
    """Return navigation data with in time-interval coverage."""
    coverage = [
        coverage_in[0].replace(tzinfo=None) - TEN_SECONDS,
        coverage_in[1].replace(tzinfo=None) + TEN_SECONDS,
    ]

    att_mask = (nav_dict["/navigation_data/att_time"] >= coverage[0]) & (
        nav_dict["/navigation_data/att_time"] <= coverage[1]
    )
    orb_mask = (nav_dict["/navigation_data/orb_time"] >= coverage[0]) & (
        nav_dict["/navigation_data/orb_time"] <= coverage[1]
    )
    tilt_mask = (nav_dict["/navigation_data/tilt_time"] >= coverage[0]) & (
        nav_dict["/navigation_data/tilt_time"] <= coverage[1]
    )

    res = {}
    for key, value in nav_dict.items():
        if key.startswith("/navigation_data/att_"):
            res[key] = value[att_mask] if value.ndim == 1 else value[att_mask, :]
        elif key.startswith("/navigation_data/orb_"):
            res[key] = value[orb_mask] if value.ndim == 1 else value[orb_mask, :]
        else:
            res[key] = value[tilt_mask] if value.ndim == 1 else value[tilt_mask, :]

    return res


def create_l1a(
    config: dataclass,
    tlm: SPXtlm,
    nav_dict: dict | None,
    mode: str | None = None,
) -> None:
    """All calls necessary to generate a SPEXone L1A product."""
    dims_nav = {}
    if config.hkt_list:
        nav_tmp = nav_adjust(nav_dict, tlm.coverage)
        dims_nav = {
            "/navigation_data/att_time": nav_tmp["/navigation_data/att_time"].size,
            "/navigation_data/orb_time": nav_tmp["/navigation_data/orb_time"].size,
            "/navigation_data/tilt_time": nav_tmp["/navigation_data/tilt_time"].size,
        }

    with SpexL1A(
        get_l1a_filename(config, tlm.coverage, mode),
        tlm.coverage,
        dims={
            "hk_packets": tlm.nomhk.size,
            "number_of_images": tlm.science.size,
            "samples_per_image": np.max([img.size for img in tlm.science.images]),
        }
        | dims_nav,
    ) as l1a:
        l1a.write_config(config)
        l1a.write_img_vars(tlm.science)  # before write_hk_vars
        l1a.write_hk_vars(tlm.nomhk)
        if config.hkt_list:
            l1a.write_nav_vars(nav_tmp)


# - class CoverageFlag ----------------------------
class CoverageFlag(IntFlag):
    """Define flags for coverage_quality (navigation_data)."""

    MISSING_SAMPLES = auto()
    TOO_SHORT_EXTENDS = auto()
    NO_EXTEND_AT_START = auto()
    NO_EXTEND_AT_END = auto()

    @classmethod
    def check(
        cls: CoverageFlag,
        att_time: NDArray[int],
        l1a_time_range: list[dt.datetime, dt.datetime],
    ) -> int:
        """Check coverage time of navigation data."""
        coverage_quality = cls(0)
        nav_time_range = (att_time[0], att_time[-1])

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


# - class SpexL1A ---------------------------------
# pylint: disable=no-member
class SpexL1A(H5Yaml):
    """Class to generate a SPEXone level-1A product.

    Parameters
    ----------
    l1a_name :  Path | str
       Name of the SPEXone level-1A product
    time_coverage :  list[dt.datetime, dt.datetime]
       Time coverage of the measurement data
    mode :  str, default="w"
       Support standard modes like 'r+' or 'w' (later is the default)
    dims :  dict[str, int], optional
       Change one or more unlimited dimensions to fixed-size dimensions
       Default of samples_per_image = row * column as defined in the YAML definition

    """

    def __init__(
        self: SpexL1A,
        l1a_name: Path | str,
        time_coverage: list[dt.datetime, dt.datetime],
        mode: str = "w",
        *,
        dims: dict[str, int] | None = None,
    ) -> None:
        """Initialize SpexL1A object and create empty SPEXone level-1A product."""
        self.logger = logging.getLogger("pyspex.SpexL1A")
        self.filename = l1a_name if isinstance(l1a_name, Path) else Path(l1a_name)

        super().__init__(files("pyspex.Data") / "h5_level_1a.yaml")

        # convert unlimited dimensions to fixed-size dimensions
        if "samples_per_image" not in dims:
            dims["samples_per_image"] = (
                self.h5_def["dimensions"]["column"]["_size"]
                * self.h5_def["dimensions"]["row"]["_size"]
            )
        for key, value in dims.items():
            if self.h5_def["dimensions"][key]["_size"] == 0:
                self.h5_def["dimensions"][key]["_size"] = value

        # create empty file and (re-)open file in append mode
        if mode == "w":
            self.create(self.filename)
        self.fid = h5py.File(self.filename, "r+")
        self.fid.attrs["time_coverage_start"] = (
            time_coverage[0].replace(tzinfo=None).isoformat(timespec="milliseconds")
        )
        self.fid.attrs["time_coverage_end"] = (
            time_coverage[1].replace(tzinfo=None).isoformat(timespec="milliseconds")
        )

    def __enter__(self: SpexL1A) -> Self:
        """Initiate the context manager."""
        return self

    def __exit__(self: SpexL1A, *args: object) -> bool:
        """Exit the context manager."""
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self: SpexL1A) -> None:
        """Close resources, after sanity check of L1A product."""
        self.add_attrs()
        self.fid.close()

    def add_attrs(self: SpexL1A) -> None:
        """Add global attributes to HDF5 product."""
        try:
            attrs_def = conf_from_yaml(files("pyspex.Data") / "h5_global_attrs.yaml")
        except RuntimeError as exc:
            raise RuntimeError from exc

        # variable globale attributes
        self.fid.attrs["product_name"] = self.filename.name
        self.fid.attrs["date_created"] = (
            dt.datetime.now(dt.UTC)
            .replace(tzinfo=None)
            .isoformat(timespec="milliseconds")
        )
        # constant global attributes
        for key, value in attrs_def["attrs_global"].items():
            if key not in self.fid.attrs:
                self.fid.attrs[key] = value
        for key, value in attrs_def["attrs_nasa"].items():
            self.fid.attrs[key] = value
        self.fid.attrs["history"] = " ".join(sys.argv)

        gid = self.fid["/processing_control"]
        gid.attrs["software_name"] = Path(sys.argv[0]).name
        gid.attrs["software_version"] = pyspex_version()
        for key, value in attrs_def["attrs_cntrl"].items():
            gid.attrs[key] = value

    def write_config(self: SpexL1A, config: dataclass) -> None:
        """Write command-line settings to /processing_control/input_parameters."""
        self.fid.attrs["processing_version"] = config.processing_version
        gid = self.fid["/processing_control/input_parameters"]
        for key, value in asdict(config).items():
            if key in ("l0_list", "hkt_list"):
                gid.attrs[key] = "" if not value else ",".join([x.name for x in value])
            else:
                gid.attrs[key] = f"{value}"

    def write_hk_vars(self: SpexL1A, nomhk: HKtlm) -> None:
        """Write nominal housekeeping data.

        Parameters
        ----------
        nomhk :  HKtlm
           SPEXone nominal housekeeping telemetry records

        """
        group = "/engineering_data"

        hk_time = np.array(
            [x.replace(tzinfo=None) for x in nomhk.tstamp], dtype="datetime64[us]"
        )

        # obtain last midnight and update the units accordingly
        dset = self.fid[f"{group}/HK_tlm_time"]
        if dset.attrs["units"].find("%Y-%m-%d %H:%M:%S") > 0:
            midnight = hk_time[0].astype("datetime64[D]")
            dset.attrs["units"] = dset.attrs["units"].replace(
                "%Y-%m-%d %H:%M:%S", f"{midnight:s} 00:00:00"
            )
        else:
            midnight = np.datetime64(dset.attrs["units"].split(" ")[2])
        dset[:] = (hk_time - midnight).astype(float) / 1e6

        self.fid[f"{group}/NomHK_telemetry"][:] = nomhk.tlm
        self.fid[f"{group}/temp_detector"][:] = nomhk.convert("TS1_DEM_N_T")
        self.fid[f"{group}/temp_housing"][:] = nomhk.convert("TS2_HOUSING_N_T")
        self.fid[f"{group}/temp_radiator"][:] = nomhk.convert("TS3_RADIATOR_N_T")
        self.logger.debug("wrote data to group: %s.", group)

    def write_img_vars(self: SpexL1A, science: SCItlm) -> None:
        """Write Science image and Science housekeeping data.

        Parameters
        ----------
        science :  SCItlm
           SPEXone Science telemetry records

        """
        group = "/image_attributes"

        image_time = np.array(
            [x.replace(tzinfo=None) for x in science.tstamp["dt"]],
            dtype="datetime64[us]",
        )

        # obtain last midnight and update the units accordingly
        dset = self.fid[f"{group}/image_time"]
        if dset.attrs["units"].find("%Y-%m-%d %H:%M:%S") > 0:
            midnight = image_time[0].astype("datetime64[D]")
            dset.attrs["units"] = dset.attrs["units"].replace(
                "%Y-%m-%d %H:%M:%S", f"{midnight:s} 00:00:00"
            )
        else:
            midnight = np.datetime64(dset.attrs["units"].split(" ")[2])
        dset[:] = (image_time - midnight).astype(float) / 1e6

        self.fid[f"{group}/timedelta_centre"][:] = (
            (science.tlm["REG_NCOADDFRAMES"] - 1) * science.frame_period()
            + science.exposure_time()
        ) / 2000
        self.fid[f"{group}/icu_time_sec"][:] = science.tstamp["tai_sec"]
        self.fid[f"{group}/icu_time_subsec"][:] = science.tstamp["sub_sec"]
        self.fid[f"{group}/image_id"][:] = np.bitwise_and(
            science.hdr["sequence"], 0x3FFF
        )
        self.fid[f"{group}/binning_table"][:] = science.binning_table()
        self.fid[f"{group}/digital_offset"][:] = science.digital_offset()
        self.fid[f"{group}/nr_coadditions"][:] = science.tlm["REG_NCOADDFRAMES"]
        self.fid[f"{group}/exposure_time"][:] = science.exposure_time() / 1000
        self.logger.debug("wrote data to group: %s.", group)

        group = "/science_data"
        self.fid[f"{group}/science_hk"][:] = science.tlm
        if len(np.unique([img.size for img in science.images])) == 1:
            images = science.images
        else:
            nj_max = 0
            for img in science.images:
                nj_max = max(nj_max, img.size)
            images = np.full((science.size, nj_max), np.iinfo("u2").max)
            for ii, img in enumerate(science.images):
                images[ii, : img.size] = img
        self.fid[f"{group}/detector_images"][:] = images
        self.logger.debug("wrote data to group: %s.", group)

    def write_nav_vars(
        self: SpexL1A,
        nav_dict: dict,
    ) -> None:
        """Write navigation-data.

        Parameters
        ----------
        nav_dict :  dict[str, NDArray]
           Dictionary holing all navigation parameters

        """
        group = "/navigation_data"

        for key, value in nav_dict.items():
            dset = self.fid[key]
            if key.endswith("_time"):
                if dset.attrs["units"].find("%Y-%m-%d %H:%M:%S") > 0:
                    midnight = value[0].astype("datetime64[D]")
                    dset.attrs["units"] = dset.attrs["units"].replace(
                        "%Y-%m-%d %H:%M:%S", f"{midnight:s} 00:00:00"
                    )
                else:
                    midnight = np.datetime64(dset.attrs["units"].split(" ")[2])
                dset[:] = (value - midnight).astype(float) / 1e6
            else:
                dset[:] = value

        # check coverage of attitude data
        gid = self.fid[group]
        gid["coverage_quality"][()] = CoverageFlag.check(
            nav_dict["/navigation_data/att_time"],
            (
                np.datetime64(self.fid.attrs["time_coverage_start"]),
                np.datetime64(self.fid.attrs["time_coverage_end"]),
            ),
        )

        # write or update attutude coverage-time
        midnight = np.datetime64(gid["att_time"].attrs["units"].split(" ")[2])
        gid.attrs["time_coverage_start"] = str(
            midnight + (1e3 * gid["att_time"][0]).astype("timedelta64[ms]")
        )
        gid.attrs["time_coverage_end"] = str(
            midnight + (1e3 * gid["att_time"][-1]).astype("timedelta64[ms]")
        )
        self.logger.debug("wrote data to group: %s.", group)
