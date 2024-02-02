#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2024 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""`SPXtlm`, class to read/access PACE/SPEXone telemetry data."""
from __future__ import annotations

__all__ = ["SPXtlm"]

import datetime as dt
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

# pylint: disable=no-name-in-module
from netCDF4 import Dataset

from .hkt_io import HKTio, check_coverage_nav, copy_hkt_nav
from .l1a_io import L1Aio
from .lib import pyspex_version
from .lib.hk_tlm import HKtlm
from .lib.leap_sec import get_leap_seconds
from .lib.sci_tlm import DET_CONSTS, SCItlm
from .lv0_lib import dump_hkt, dump_science, read_lv0_data

if TYPE_CHECKING:
    from dataclasses import dataclass

# - global parameters -----------------------
module_logger = logging.getLogger("pyspex.tlm")

FULLFRAME_BYTES = 2 * DET_CONSTS["dimFullFrame"]
DATE_MIN = dt.datetime(2020, 1, 1, 1, tzinfo=dt.UTC)
TSTAMP_MIN = int(DATE_MIN.timestamp())


# - helper functions ------------------------
def get_l1a_filename(
    config: dataclass,
    coverage: tuple[dt.datetime, dt.datetime],
    mode: str | None = None,
) -> Path:
    """Return filename of level-1A product.

    Parameters
    ----------
    config :  dataclass
        Settings for the L0->l1A processing
    coverage :  tuple[dt.datetime, dt.datetime]
        time_coverage_start and time_coverage_end of the data
    mode :  {'binned', 'full'} | None, default=None
        Select Science packages with full-frame images or binned images

    Returns
    -------
    Path
        Filename of level-1A product.

    Notes
    -----
    === Inflight ===
    L1A file name format, following the NASA ... naming convention:
        PACE_SPEXONE[_TTT].YYYYMMDDTHHMMSS.L1A[.Vnn].nc
    where
        TTT is an optional data type (e.g., for the calibration data files)
        YYYYMMDDTHHMMSS is time stamp of the first image in the file
        Vnn file-version number (omitted when nn=1)

    for example (file-version=1):
        [Science Product] PACE_SPEXONE.20230115T123456.L1A.nc
        [Calibration Product] PACE_SPEXONE_CAL.20230115T123456.L1A.nc
        [Dark science Product] PACE_SPEXONE_DARK.20230115T123456.L1A.nc

    === OCAL ===
    L1A file name format:
        SPX1_OCAL_<msm_id>[_YYYYMMDDTHHMMSS]_L1A_vvvvvvv.nc
    where
        msm_id is the measurement identifier
        YYYYMMDDTHHMMSS is time stamp of the first image in the file
        vvvvvvv is the git-hash string of the pyspex repository

    """
    if config.outfile:
        return config.outdir / config.outfile

    # +++++ in-flight product-name convention +++++
    if config.l0_format != "raw":
        if config.eclipse is None:
            subtype = "_OCAL"
        elif not config.eclipse:
            subtype = ""
        else:
            subtype = "_CAL" if mode == "full" else "_DARK"

        prod_ver = "" if config.file_version == 1 else f".V{config.file_version:02d}"

        return config.outdir / (
            f'PACE_SPEXONE{subtype}'
            f'.{coverage[0].strftime("%Y%m%dT%H%M%S"):15s}'
            f'.L1A{prod_ver}.nc'
        )

    # +++++ OCAL product-name convention +++++
    # determine measurement identifier
    msm_id = config.l0_list[0].stem
    try:
        new_date = dt.datetime.strptime(msm_id[-22:], "%y-%j-%H:%M:%S.%f").strftime(
            "%Y%m%dT%H%M%S.%f"
        )
    except ValueError:
        pass
    else:
        msm_id = msm_id[:-22] + new_date

    return config.outdir / f"SPX1_OCAL_{msm_id}_L1A_{pyspex_version(githash=True)}.nc"


def add_hkt_navigation(l1a_file: Path, hkt_list: tuple[Path, ...]) -> int:
    """Add PACE navigation information from PACE_HKT products.

    Parameters
    ----------
    l1a_file :  Path
       name of an existing L1A product.
    hkt_list :  list[Path, ...]
       listing of files from which the navigation data has to be read

    """
    # read PACE navigation data from HKT files
    # and add PACE navigation data to existing level-1A product
    copy_hkt_nav(hkt_list, l1a_file)
    # check time coverage of navigation data.
    return check_coverage_nav(l1a_file)


def add_proc_conf(l1a_file: Path, yaml_conf: Path) -> None:
    """Add dataset 'processor_configuration' to an existing L1A product.

    Parameters
    ----------
    l1a_file :  Path
       name of an existing L1A product.
    yaml_conf :  Path
       name of the YAML file with the processor settings

    """
    with Dataset(l1a_file, "r+") as fid:
        dset = fid.createVariable("processor_configuration", str)
        dset.comment = (
            "Configuration parameters used during"
            " the processor run that produced this file."
        )
        dset.markup_language = "YAML"
        dset[0] = "".join(
            [
                s
                for s in yaml_conf.open(encoding="ascii").readlines()
                if not (s == "\n" or s.startswith("#"))
            ]
        )


# - class SPXtlm ----------------------------
class SPXtlm:
    """Access/convert parameters of SPEXone Science telemetry data.

    Notes
    -----
    This class has the following methods::

     - set_coverage(coverage: tuple[datetime, datetime] | None,
                    update: boot = False) -> None
     - reference_date() -> datetime
     - time_coverage_start() -> datetime
     - time_coverage_end() -> datetime
     - from_hkt(flnames: str | Path | list[Path], *,
                instrument: str | None = None, dump: bool = False) -> None
     - from_lv0(flnames: str | Path | list[Path], file_format: str = 'dsb',
                *, tlm_type: str | None = None,
                debug: bool = False, dump: bool = False) -> None
     - from_l1a(flname: str | Path,
                *, tlm_type: str | None = None,
                mps_id: int | None = None) -> None
     - gen_l1a(config: dataclass, mode: str) -> None


    Examples
    --------
    Read data from SPEXone level-0 products

    >>> from pyspex.tlm import SPXtlm
    >>> spx = SPXtlm()
    >>> spx.from_l0(list_l0_products, tlm_type='hk')
    # returns list of TcAccept, TcReject, TcExecute, TcFail and EventRp
    >>> spx.nomhk.events
    # return detector images
    >>> spx.from_l0(list_l0_products, tlm_type='sci')
    >>> spx.science.images

    Read data from a SPEXone level-1a product

    >>> from pyspex.tlm import SPXtlm
    >>> spx = SPXtlm()
    >>> spx.from_l1a(l1a_product, tlm_type='sci', mps_is=47)
    >>> spx.science.images

    Read data from a PACE-HKT product

    >>> from pyspex.tlm import SPXtlm
    >>> spx = SPXtlm()
    >>> spx.from_hkt(hkt_product)
    # returns list of TcAccept, TcReject, TcExecute, TcFail and EventRp
    >>> spx.nomhk.events

    """

    def __init__(self: SPXtlm) -> None:
        """Initialize SPXtlm object."""
        self.logger = logging.getLogger(__name__)
        self.file_list: list[Path] | None = None
        self._coverage: list[dt.datetime, dt.datetime] | None = None
        self.nomhk = HKtlm()
        self.science = SCItlm()

    @property
    def coverage(self: SPXtlm) -> list[dt.datetime, dt.datetime] | None:
        """Return data time_coverage."""
        return self._coverage

    def sel(self: SPXtlm, mask: np.NDArray[bool]) -> SPXtlm:
        """Return subset of SPXtlm object using a mask array."""
        spx = SPXtlm()
        spx.logger = self.logger
        spx.file_list = self.file_list.copy()
        spx.science = self.science.sel(mask)

        # set Science time_coverage_range
        indices = mask.nonzero()[0]
        if len(indices) == 1:
            frame_period = dt.timedelta(milliseconds=self.science.frame_period(0))
            spx.set_coverage(
                [
                    spx.science.tstamp[0]["dt"],
                    spx.science.tstamp[0]["dt"] + frame_period,
                ]
            )
        else:
            frame_period = dt.timedelta(milliseconds=self.science.frame_period(-1))
            spx.set_coverage(
                [
                    spx.science.tstamp[0]["dt"],
                    spx.science.tstamp[-1]["dt"] + frame_period,
                ]
            )

        # select nomhk data within Science time_coverage_range
        hk_tstamps = np.array(self.nomhk.tstamp, dtype="datetime64")
        dt_min = np.datetime64(spx.coverage[0]) - np.timedelta64(1, "s")
        dt_max = np.datetime64(spx.coverage[1]) + np.timedelta64(1, "s")
        hk_mask = (hk_tstamps >= dt_min) & (hk_tstamps <= dt_max)
        spx.nomhk = self.nomhk.sel(hk_mask)
        return spx

    def set_coverage(
        self: SPXtlm,
        coverage_new: list[dt.datetime, dt.datetime] | None,
        update: bool = False,
    ) -> None:
        """Set, reset or update the class attribute `coverage`."""
        if not update or self._coverage is None:
            self._coverage = coverage_new
            return

        one_hour = dt.timedelta(hours=1)
        if self._coverage[0] - one_hour < coverage_new[0] < self._coverage[0]:
            self._coverage[0] = coverage_new[0]
        if self._coverage[1] < coverage_new[1] < self._coverage[1] + one_hour:
            self._coverage[1] = coverage_new[1]

    @property
    def reference_date(self: SPXtlm) -> dt.datetime:
        """Return date of reference day (tzone aware)."""
        if self._coverage is None:
            raise ValueError("no valid timestamps found")

        return dt.datetime.combine(self._coverage[0].date(), dt.time(0), dt.UTC)

    @property
    def time_coverage_start(self: SPXtlm) -> dt.datetime | None:
        """Return time_coverage_start."""
        return None if self._coverage is None else self._coverage[0]

    @property
    def time_coverage_end(self: SPXtlm) -> dt.datetime | None:
        """Return time_coverage_end."""
        return None if self._coverage is None else self._coverage[1]

    def from_hkt(
        self: SPXtlm,
        flnames: str | Path | list[Path],
        *,
        instrument: str | None = None,
        dump: bool = False,
    ) -> None:
        """Read telemetry data from PACE HKT product(s).

        Parameters
        ----------
        flnames :  str | Path | list[Path]
           sorted list of PACE_HKT filenames (netCDF4 format)
        instrument :  {'spx', 'sc', 'oci', 'harp'}, default='spx'
           abbreviations for the PACE instruments
        dump :  bool, default=False
           dump header information of the telemetry packages @1Hz for
           debugging purposes

        """
        self.set_coverage(None)
        if isinstance(flnames, str | Path):
            flnames = [flnames] if isinstance(flnames, Path) else [Path(flnames)]
        if instrument is None:
            instrument = "spx"
        elif instrument not in ["spx", "sc", "oci", "harp"]:
            raise KeyError("instrument not in ['spx', 'sc', 'oci', 'harp']")

        self.file_list = flnames
        ccsds_hk: tuple[np.ndarray, ...] | tuple[()] = ()
        for name in flnames:
            hkt = HKTio(name)
            ccsds_hk += hkt.housekeeping(instrument)

        # check number of telemetry data-packages
        if not ccsds_hk:
            return
        # perform dump of telemetry data-packages
        if dump:
            dump_hkt(flnames[0].stem + "_hkt.dump", ccsds_hk)
            return
        # check if TAI timestamp is valid
        ii = len(ccsds_hk) // 2
        if tai_sec := ccsds_hk[ii]["hdr"]["tai_sec"][0] == 0:
            return

        # set epoch
        epoch = dt.datetime(1958, 1, 1, tzinfo=dt.UTC)
        epoch -= dt.timedelta(seconds=get_leap_seconds(float(tai_sec)))
        self.nomhk.extract_l0_hk(ccsds_hk, epoch)

        # reject nomHK records before or after a big time-jump
        _mm = np.diff(self.nomhk.tstamp) > np.timedelta64(1, "D")
        if np.any(_mm):
            indx = _mm.nonzero()[0]
            _mm = np.full(self.nomhk.size, True, dtype=bool)
            _mm[indx[0] + 1 :] = False
            if np.sum(_mm) < self.nomhk.size // 2:
                _mm = ~_mm
            self.logger.warning(
                "rejected nomHK: %d -> %d", self.nomhk.size, np.sum(_mm)
            )
            self.nomhk.sel(_mm)

        # set time-coverage
        self.set_coverage(
            [self.nomhk.tstamp[0], self.nomhk.tstamp[-1] + dt.timedelta(seconds=1)]
        )

    def from_lv0(
        self: SPXtlm,
        flnames: str | Path | list[Path],
        file_format: str = "dsb",
        *,
        tlm_type: str | None = None,
        debug: bool = False,
        dump: bool = False,
    ) -> None:
        """Read telemetry data from SPEXone level-0 product(s).

        Parameters
        ----------
        flnames :  str | Path | list[Path]
           sorted list of CCSDS filenames
        file_format : {'raw', 'st3', 'dsb'}, default='dsb'
           type of CCSDS data
        tlm_type :  {'hk', 'sci', 'all'}, default='all'
           select type of telemetry packages
           Note that we always read the complete level-0 products.
        debug : bool, default=False
           run in debug mode, read only packages heades
        dump :  bool, default=False
           dump header information of the telemetry packages @1Hz for
           debugging purposes

        """
        self.set_coverage(None)
        if isinstance(flnames, str | Path):
            flnames = [flnames] if isinstance(flnames, Path) else [Path(flnames)]
        if file_format not in ["raw", "st3", "dsb"]:
            raise KeyError("file_format not in ['raw', 'st3', 'dsb']")
        if tlm_type is None:
            tlm_type = "all"
        elif tlm_type not in ["hk", "sci", "all"]:
            raise KeyError("tlm_type not in ['hk', 'sci', 'all']")
        self.file_list = flnames

        # read telemetry data
        ccsds_sci, ccsds_hk = read_lv0_data(flnames, file_format, debug=debug)

        # perform ASCII dump
        if dump and ccsds_hk:
            dump_hkt(flnames[0].stem + "_hkt.dump", ccsds_hk)
        if dump and ccsds_sci:
            dump_science(flnames[0].stem + "_sci.dump", ccsds_sci)

        # exit when debugging or only an ASCII data-dump is requested
        if debug or dump:
            return

        # exit when Science data is requested and no Science data is available
        #   or when housekeeping is requested and no housekeeping data is available
        if (not ccsds_sci and tlm_type == "sci") or (not ccsds_hk and tlm_type == "hk"):
            self.logger.info("Asked for tlm_type=%s, but none found", tlm_type)
            return

        # set epoch
        if file_format == "dsb":
            # check if TAI timestamp is valid
            ii = len(ccsds_hk) // 2
            if tai_sec := ccsds_hk[ii]["hdr"]["tai_sec"][0] == 0:
                return

            epoch = dt.datetime(1958, 1, 1, tzinfo=dt.UTC)
            epoch -= dt.timedelta(seconds=get_leap_seconds(float(tai_sec)))
        else:
            epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)

        if tlm_type != "hk":
            # collect Science telemetry data
            if self.science.extract_l0_sci(ccsds_sci, epoch) == 0:
                self.logger.info("no valid Science package found")
            else:
                # reject Science records before or after a big time-jump
                _mm = np.diff(self.science.tstamp["dt"]) > np.timedelta64(1, "D")
                if np.any(_mm):
                    indx = _mm.nonzero()[0]
                    _mm = np.full(self.science.size, True, dtype=bool)
                    _mm[indx[0] + 1 :] = False
                    if np.sum(_mm) < self.science.size // 2:
                        _mm = ~_mm
                    self.logger.warning(
                        "rejected science: %d -> %d", self.science.size, np.sum(_mm)
                    )
                    self.science.sel(_mm)

                # set time-coverage
                intg = dt.timedelta(milliseconds=self.science.frame_period(-1))
                self.set_coverage(
                    [self.science.tstamp["dt"][0], self.science.tstamp["dt"][-1] + intg]
                )

        # collected NomHK telemetry data
        if tlm_type != "sci" and ccsds_hk:
            self.nomhk.extract_l0_hk(ccsds_hk, epoch)

            # reject nomHK records before or after a big time-jump
            _mm = np.diff(self.nomhk.tstamp) > np.timedelta64(1, "D")
            if np.any(_mm):
                indx = _mm.nonzero()[0]
                _mm = np.full(self.nomhk.size, True, dtype=bool)
                _mm[indx[0] + 1 :] = False
                if self.coverage is None:
                    if np.sum(_mm) < self.nomhk.size // 2:
                        _mm = ~_mm
                else:
                    dt_start = abs(self.coverage[0] - self.nomhk.tstamp[0])
                    dt_end = abs(self.coverage[-1] - self.nomhk.tstamp[-1])
                    if dt_start > dt_end:
                        _mm = ~_mm
                self.logger.warning(
                    "rejected nomHK: %d -> %d", self.nomhk.size, np.sum(_mm)
                )
                self.nomhk.sel(_mm)

            # set time-coverage (only, when self._coverage is None)
            self.set_coverage(
                [self.nomhk.tstamp[0], self.nomhk.tstamp[-1] + dt.timedelta(seconds=1)]
            )

    def from_l1a(
        self: SPXtlm,
        flname: str | Path,
        *,
        tlm_type: str | None = None,
        mps_id: int | None = None,
    ) -> None:
        """Read telemetry data from SPEXone level-1A product.

        Parameters
        ----------
        flname :  Path
           name of one SPEXone level-1A product
        tlm_type :  {'hk', 'sci', 'all'}, default='all'
           select type of telemetry packages
        mps_id :  int, optional
           select on MPS ID

        """
        self.set_coverage(None)
        if tlm_type is None:
            tlm_type = "all"
        elif tlm_type not in ["hk", "sci", "all"]:
            raise KeyError("tlm_type not in ['hk', 'sci', 'all']")

        self.file_list = [flname]
        with h5py.File(flname) as fid:
            # collect Science telemetry data
            if tlm_type != "hk":
                self.science.extract_l1a_sci(fid, mps_id)
                if self.science.size == 0:
                    return

                _mm = self.science.tstamp["tai_sec"] > TSTAMP_MIN
                if np.any(_mm):
                    tstamp = self.science.tstamp["dt"][_mm]
                    ii = int(np.nonzero(_mm)[0][-1])
                    intg = dt.timedelta(milliseconds=self.science.frame_period(ii))
                    self.set_coverage([tstamp[0], tstamp[-1] + intg])

            # collected NomHk telemetry data
            if tlm_type != "sci":
                self.nomhk.extract_l1a_hk(fid, mps_id)
                if self.nomhk.size == 0:
                    return
                self.set_coverage(
                    [
                        self.nomhk.tstamp[0],
                        self.nomhk.tstamp[-1] + dt.timedelta(seconds=1),
                    ]
                )

    def __select_msm_mode(self: SPXtlm, mode: str) -> dict | None:
        """Obtain image and housekeeping dimensions given data-mode.

        Parameters
        ----------
        mode :  {'all', 'binned', 'full'}
           Select image data in binned mode or full-frame mode, or no selection

        Returns
        -------
        dict {'sci_mask': np.ndarray | [],
              'hk_mask': np.ndarray,
              'dims': {'number_of_images': int,
                       'samples_per_image': int,
                       'hk_packets': int}
        }

        """
        if mode == "all":
            return {
                "hk_mask": np.full(self.nomhk.size, True),
                "sci_mask": np.full(self.science.size, True),
                "dims": {
                    "number_of_images": self.science.size,
                    "samples_per_image": (
                        DET_CONSTS["dimRow"]
                        if self.science.size == 0
                        else np.max([x.size for x in self.science.images])
                    ),
                    "hk_packets": self.nomhk.size,
                },
            }

        if mode == "full":
            if (
                self.science.tlm is None
                or np.sum(self.science.tlm["IMRLEN"] == FULLFRAME_BYTES) == 0
            ):
                return None

            sci_mask = self.science.tlm["IMRLEN"] == FULLFRAME_BYTES
            mps_list = np.unique(self.science.tlm["MPS_ID"][sci_mask])
            self.logger.debug("unique Diagnostic MPS: %s", mps_list)
            hk_mask = np.in1d(self.nomhk.tlm["MPS_ID"], mps_list)

            return {
                "sci_mask": sci_mask,
                "hk_mask": hk_mask,
                "dims": {
                    "number_of_images": np.sum(sci_mask),
                    "samples_per_image": DET_CONSTS["dimFullFrame"],
                    "hk_packets": np.sum(hk_mask),
                },
            }

        if mode == "binned":
            if (
                self.science.tlm is None
                or np.sum(self.science.tlm["IMRLEN"] < FULLFRAME_BYTES) == 0
            ):
                return None

            sci_mask = self.science.tlm["IMRLEN"] < FULLFRAME_BYTES
            mps_list = np.unique(self.science.tlm["MPS_ID"][sci_mask])
            self.logger.debug("unique Science MPS: %s", mps_list)
            hk_mask = np.in1d(self.nomhk.tlm["MPS_ID"], mps_list)
            return {
                "sci_mask": sci_mask,
                "hk_mask": hk_mask,
                "dims": {
                    "number_of_images": np.sum(sci_mask),
                    "samples_per_image": np.max(
                        [self.science.images[ii].size for ii in sci_mask.nonzero()[0]]
                    ),
                    "hk_packets": np.sum(hk_mask),
                },
            }
        raise KeyError("mode should be 'binned', 'full' or 'all'")

    def gen_l1a(self: SPXtlm, config: dataclass, mode: str) -> None:
        """Generate a SPEXone level-1A product.

        Parameters
        ----------
        config :  dataclass
           Settings for the L0->L1A processing
        mode :  {'all', 'binned', 'full'}
           Select Science packages with full-frame images or binned images

        """
        if (selection := self.__select_msm_mode(mode)) is None:
            return

        # set time-coverage range
        coverage = None
        if np.sum(selection["sci_mask"]) > 0:
            _mm = selection["sci_mask"] & self.science.tstamp["tai_sec"] > TSTAMP_MIN
            if np.any(_mm):
                tstamp = self.science.tstamp["dt"][_mm]
                ii = int(np.nonzero(_mm)[0][-1])
                intg = dt.timedelta(milliseconds=self.science.frame_period(ii))
                coverage = (tstamp[0], tstamp[-1] + intg)

        if coverage is None:
            tstamp = [
                self.nomhk.tstamp[ii] for ii in np.nonzero(selection["hk_mask"])[0]
            ]
            tstamp = [x for x in tstamp if x > DATE_MIN]
            coverage = (tstamp[0], tstamp[-1] + dt.timedelta(seconds=1))

        l1a_path = get_l1a_filename(config, coverage, mode)
        with L1Aio(
            l1a_path,
            self.reference_date,
            selection["dims"],
            compression=config.compression,
        ) as l1a:
            l1a.fill_global_attrs(inflight=config.l0_format != "raw")
            if config.processing_version:
                l1a.set_attr("processing_version", config.processing_version)
            l1a.set_attr("icu_sw_version", f'0x{self.nomhk.tlm["ICUSWVER"][0]:x}')
            l1a.set_attr(
                "time_coverage_start", coverage[0].isoformat(timespec="milliseconds")
            )
            l1a.set_attr(
                "time_coverage_end", coverage[1].isoformat(timespec="milliseconds")
            )
            l1a.set_attr("input_files", [x.name for x in config.l0_list])
            self.logger.debug("(1) initialized level-1A product")

            self._fill_engineering(l1a, selection["hk_mask"])
            self.logger.debug("(2) added engineering data")
            self._fill_science(l1a, selection["sci_mask"])
            self.logger.debug("(3) added science data")
            self._fill_image_attrs(l1a, config.l0_format, selection["sci_mask"])
            self.logger.debug("(4) added image attributes")

        # add processor_configuration
        if config.yaml_fl:
            add_proc_conf(l1a_path, config.yaml_fl)

        # add PACE navigation information from HKT products
        if config.hkt_list:
            status_ok = add_hkt_navigation(l1a_path, config.hkt_list)
            self.logger.debug("(5) added PACE navigation data")

            if not status_ok:
                raise UserWarning("time-coverage of navigation data is too short")

        self.logger.info("successfully generated: %s", l1a_path.name)

    def _fill_engineering(self: SPXtlm, l1a: L1Aio, hk_mask: np.ndarray) -> None:
        """Fill datasets in group '/engineering_data'."""
        if np.sum(hk_mask) == 0:
            return

        l1a.set_dset("/engineering_data/NomHK_telemetry", self.nomhk.tlm[hk_mask])
        ref_date = self.reference_date
        l1a.set_dset(
            "/engineering_data/HK_tlm_time",
            [
                (x - ref_date).total_seconds()
                for ii, x in enumerate(self.nomhk.tstamp)
                if hk_mask[ii]
            ],
        )
        l1a.set_dset(
            "/engineering_data/temp_detector",
            self.nomhk.convert("TS1_DEM_N_T")[hk_mask],
        )
        l1a.set_dset(
            "/engineering_data/temp_housing",
            self.nomhk.convert("TS2_HOUSING_N_T")[hk_mask],
        )
        l1a.set_dset(
            "/engineering_data/temp_radiator",
            self.nomhk.convert("TS3_RADIATOR_N_T")[hk_mask],
        )

    def _fill_science(self: SPXtlm, l1a: L1Aio, sci_mask: np.ndarray) -> None:
        """Fill datasets in group '/science_data'."""
        if np.sum(sci_mask) == 0:
            return

        img_list = [img for ii, img in enumerate(self.science.images) if sci_mask[ii]]
        img_sz = [img.size for img in img_list]
        if len(np.unique(img_sz)) != 1:
            images = np.zeros((len(img_sz), np.max(img_sz)), dtype="u2")
            for ii, img in enumerate(img_list):
                images[ii, : len(img)] = img
        else:
            images = np.vstack(img_list)
        l1a.set_dset("/science_data/detector_images", images)
        l1a.set_dset("/science_data/detector_telemetry", self.science.tlm[sci_mask])

    def _fill_image_attrs(
        self: SPXtlm, l1a: L1Aio, lv0_format: str, sci_mask: np.ndarray
    ) -> None:
        """Fill datasets in group '/image_attributes'."""
        if np.sum(sci_mask) == 0:
            return

        l1a.set_dset(
            "/image_attributes/icu_time_sec", self.science.tstamp["tai_sec"][sci_mask]
        )
        # modify attribute units for non-DSB products
        if lv0_format != "dsb":
            # timestamp of 2020-01-01T00:00:00+00:00
            l1a.set_attr(
                "valid_min",
                np.uint32(1577836800),
                ds_name="/image_attributes/icu_time_sec",
            )
            # timestamp of 2024-01-01T00:00:00+00:00
            l1a.set_attr(
                "valid_max",
                np.uint32(1704067200),
                ds_name="/image_attributes/icu_time_sec",
            )
            l1a.set_attr(
                "units",
                "seconds since 1970-01-01 00:00:00",
                ds_name="/image_attributes/icu_time_sec",
            )
        l1a.set_dset(
            "/image_attributes/icu_time_subsec",
            self.science.tstamp["sub_sec"][sci_mask],
        )
        ref_date = self.reference_date
        l1a.set_dset(
            "/image_attributes/image_time",
            [
                (x - ref_date).total_seconds()
                for x in self.science.tstamp["dt"][sci_mask]
            ],
        )
        l1a.set_dset(
            "/image_attributes/image_ID",
            np.bitwise_and(self.science.hdr["sequence"][sci_mask], 0x3FFF),
        )
        l1a.set_dset(
            "/image_attributes/binning_table", self.science.binning_table()[sci_mask]
        )
        l1a.set_dset(
            "/image_attributes/digital_offset", self.science.digital_offset()[sci_mask]
        )
        l1a.set_dset(
            "/image_attributes/exposure_time",
            self.science.exposure_time()[sci_mask] / 1000,
        )
        l1a.set_dset(
            "/image_attributes/nr_coadditions",
            self.science.tlm["REG_NCOADDFRAMES"][sci_mask],
        )
