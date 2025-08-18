#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""`SPXtlm`, class to read/access PACE/SPEXone telemetry data."""

from __future__ import annotations

__all__ = ["SPXtlm"]

import datetime as dt
import logging
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from .hkt_io import HKTio
from .lib import pyspex_version
from .lib.hk_tlm import HKtlm
from .lib.leap_sec import get_leap_seconds
from .lib.sci_tlm import DET_CONSTS, SCItlm
from .lv0_lib import dump_hkt, dump_science, read_lv0_data

if TYPE_CHECKING:
    from dataclasses import dataclass

    import numpy.typing as npt


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
        subtype = ""
        if config.eclipse is None:
            subtype = "_OCAL"
        elif config.eclipse:
            subtype = "_CAL" if mode == "full" else "_DARK"

        prod_ver = (
            "" if config.processing_version == 1 else f".V{config.processing_version:d}"
        )
        return config.outdir / (
            f"PACE_SPEXONE{subtype}"
            f".{coverage[0].strftime('%Y%m%dT%H%M%S'):15s}"
            f".L1A{prod_ver}.nc"
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


# - class SPXtlm ----------------------------
class SPXtlm:
    """Access/convert parameters of SPEXone Science telemetry data.

    Notes
    -----
    This class has the following methods::

     - set_coverage(coverage: tuple[datetime, datetime] | None,
                    extent: bool = False) -> None
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


    Examples
    --------
    Read data from SPEXone level-0 products

    >>> from pyspex.tlm import SPXtlm
    >>> spx = SPXtlm()
    >>> spx.from_lv0(list_l0_products, tlm_type='hk')
    # returns list of TcAccept, TcReject, TcExecute, TcFail and EventRp
    >>> spx.nomhk.events
    # return detector images
    >>> spx.from_lv0(list_l0_products, tlm_type='sci')
    >>> spx.science.images

    Read data from a SPEXone level-1a product

    >>> from pyspex.tlm import SPXtlm
    >>> spx = SPXtlm()
    >>> spx.from_l1a(l1a_product, tlm_type='sci', mps_id=47)
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
        self.mode = "all"
        self.logger = logging.getLogger(__name__)
        self.file_list: list[Path] | None = None
        self._coverage: list[dt.datetime, dt.datetime] | None = None
        self.nomhk: HKtlm = HKtlm()
        self.science: SCItlm = SCItlm()

    @property
    def coverage(self: SPXtlm) -> list[dt.datetime, dt.datetime] | None:
        """Return data time_coverage."""
        return self._coverage

    def set_coverage(
        self: SPXtlm,
        coverage_new: tuple[dt.datetime, dt.datetime] | None,
        extent: bool = False,
    ) -> None:
        """Set, reset or update the class attribute `coverage`.

        Parameters
        ----------
        coverage_new :  list[dt.datetime, dt.datetime] | None,
           Provide new time_coverage range, or reset when coverage_new is None
        extent :  bool, default=False
           Extent existing time_coverage range with coverage_new

        Examples
        --------
        Reset self.coverage to a new time_coverage:
        > self.set_coverage(None)
        > self.set_coverage(coverage_new)

        Extent self.coverage with a new time_coverage
        if self.coverage is not None else self.coverage will be coverage_new:
        > self.set_coverage(coverage_new, extent=True)

        This will not change self.coverage to coverage_new
        when self.coverage is not None:
        > self.set_coverage(coverage_new)

        """
        if self._coverage is None or coverage_new is None:
            self._coverage = coverage_new
            return

        if not extent:
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

    def sel(self: SPXtlm, mask: npt.NDArray[bool]) -> SPXtlm:
        """Return subset of SPXtlm object using a mask array."""
        spx = copy(self)
        spx.set_coverage(None)
        spx.science = self.science.sel(mask)

        # set Science time_coverage_range
        indices = mask.nonzero()[0]
        master_cycle = dt.timedelta(milliseconds=self.science.master_cycle(-1))
        spx.set_coverage(None)
        if len(indices) == 1:
            spx.set_coverage(
                (
                    spx.science.tstamp[0]["dt"],
                    spx.science.tstamp[0]["dt"] + master_cycle,
                )
            )
        else:
            spx.set_coverage(
                (
                    spx.science.tstamp[0]["dt"],
                    spx.science.tstamp[-1]["dt"] + master_cycle,
                )
            )
        # select nomhk data within Science time_coverage_range
        hk_tstamps = np.array(
            [x.replace(tzinfo=None) for x in self.nomhk.tstamp], dtype="datetime64"
        )
        dt_min = np.datetime64(spx.coverage[0].replace(tzinfo=None)) - np.timedelta64(
            1, "s"
        )
        dt_max = np.datetime64(spx.coverage[1].replace(tzinfo=None)) + np.timedelta64(
            1, "s"
        )
        spx.nomhk = self.nomhk.sel((hk_tstamps >= dt_min) & (hk_tstamps <= dt_max))
        return spx

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

        # check number of telemetry data-packages
        ccsds_hk = HKTio(flnames).housekeeping(instrument)
        if not ccsds_hk:
            return

        # perform dump of telemetry data-packages
        if dump:
            dump_hkt(flnames[0].stem + "_hkt.dump", ccsds_hk)
            return

        # check if TAI timestamp is valid
        ii = len(ccsds_hk) // 2
        if (tai_sec := ccsds_hk[ii]["hdr"]["tai_sec"][0]) == 0:
            return

        # set epoch
        epoch = dt.datetime(1958, 1, 1, tzinfo=dt.UTC)
        epoch -= dt.timedelta(seconds=get_leap_seconds(float(tai_sec)))
        self.nomhk.extract_l0_hk(ccsds_hk, epoch)

        # reject nomHK records before or after a big time-jump
        _mm = np.diff(self.nomhk.tstamp) > dt.timedelta(days=1)
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
            (self.nomhk.tstamp[0], self.nomhk.tstamp[-1] + dt.timedelta(seconds=1))
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
            if (tai_sec := ccsds_hk[ii]["hdr"]["tai_sec"][0]) == 0:
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
                _mm = np.diff(self.science.tstamp["dt"]) > dt.timedelta(days=1)
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
                master_cycle = dt.timedelta(milliseconds=self.science.master_cycle(-1))
                self.set_coverage(
                    (
                        self.science.tstamp["dt"][0],
                        self.science.tstamp["dt"][-1] + master_cycle,
                    )
                )

        # collected NomHK telemetry data
        if tlm_type != "sci" and ccsds_hk:
            self.nomhk.extract_l0_hk(ccsds_hk, epoch)
            if self.nomhk.size == 0:
                self.logger.info("no valid housekeeping package found")
                return

            # reject nomHK records before or after a big time-jump
            _mm = np.diff(self.nomhk.tstamp) > dt.timedelta(days=1)
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
                (self.nomhk.tstamp[0], self.nomhk.tstamp[-1] + dt.timedelta(seconds=1))
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
                    master_cycle = dt.timedelta(
                        milliseconds=self.science.master_cycle(ii)
                    )
                    self.set_coverage((tstamp[0], tstamp[-1] + master_cycle))

            # collected NomHk telemetry data
            if tlm_type != "sci":
                self.nomhk.extract_l1a_hk(fid, mps_id)
                if self.nomhk.size == 0:
                    return
                self.set_coverage(
                    (
                        self.nomhk.tstamp[0],
                        self.nomhk.tstamp[-1] + dt.timedelta(seconds=1),
                    )
                )

    def full(self: SPXtlm) -> SPXtlm:
        """Select full-frame measurements."""
        self.mode = "full"
        if self.science.size == 0:
            return self

        sci_mask = (self.science.tstamp["tai_sec"] > TSTAMP_MIN) & (
            self.science.tlm["IMRLEN"] == FULLFRAME_BYTES
        )
        if np.all(~sci_mask):
            return SPXtlm()  # return empy object
        if np.all(sci_mask):
            return self  # return original object
        self.logger.debug("Rejected %d binned Science images", np.sum(~sci_mask))

        return copy(self).sel(sci_mask)

    def binned(self: SPXtlm) -> SPXtlm:
        """Select binned images from data."""
        self.mode = "binned"
        if self.science.size == 0:
            return self

        sci_mask = (self.science.tstamp["tai_sec"] > TSTAMP_MIN) & (
            self.science.tlm["IMRLEN"] < FULLFRAME_BYTES
        )
        if np.all(~sci_mask):
            return SPXtlm()  # return empy object
        if np.all(sci_mask):
            return self  # return original object
        self.logger.debug("Rejected %d full-frame Science images", np.sum(~sci_mask))

        return copy(self).sel(sci_mask)
