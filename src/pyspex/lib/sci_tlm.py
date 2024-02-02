#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2023-2024 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""`SCItlm`, class to read/access PACE/SPEXone telemetry data."""
from __future__ import annotations

__all__ = ["SCItlm"]

import datetime as dt
import logging
from typing import TYPE_CHECKING

import numpy as np

from .ccsds_hdr import CCSDShdr
from .leap_sec import get_leap_seconds
from .tlm_utils import convert_hk

if TYPE_CHECKING:
    import h5py
    import numpy.typing as npt


# - global parameters -----------------------
DET_CONSTS = {
    "dimRow": 2048,
    "dimColumn": 2048,
    "dimFullFrame": 2048 * 2048,
    "DEM_frequency": 10,  # [MHz]
    "FTI_science": 1000 / 15,  # [ms]
    "FTI_diagnostic": 240.0,  # [ms]
    "FTI_margin": 212.4,  # [ms]
    "overheadTime": 0.4644,  # [ms]
    "FOT_length": 20,
}
FULLFRAME_BYTES = 2 * DET_CONSTS["dimFullFrame"]

TSTAMP_MIN = 1561939200  # 2019-07-01T00:00:00+00:00
TSTAMP_TYPE = np.dtype([("tai_sec", int), ("sub_sec", int), ("dt", "O")])

module_logger = logging.getLogger("pyspex.lib.sci_tlm")


# - helper functions ------------------------
def subsec2musec(sub_sec: int) -> int:
    """Return subsec as microseconds."""
    return 100 * int(sub_sec / 65536 * 10000)


def mask2slice(mask: npt.NDArray[bool]) -> None | slice | tuple | npt.NDArray[bool]:
    """Try to slice (faster), instead of boolean indexing (slow)."""
    if np.all(~mask):
        return None
    if np.all(mask):
        return np.s_[:]  # read everything

    indx = np.nonzero(mask)[0]
    if np.all(np.diff(indx) == 1):
        # perform start-stop indexing
        return np.s_[indx[0] : indx[-1] + 1]

    # perform boolean indexing
    return mask


# - class SCItlm ----------------------------
class SCItlm:
    """Class to handle SPEXone Science-telemetry packets."""

    def __init__(self: SCItlm) -> None:
        """Initialize SCItlm object."""
        self.hdr: np.ndarray | None = None
        self.tlm: np.ndarray | None = None
        self.tstamp: np.ndarray | None = None
        self.images: tuple[np.ndarray, ...] | tuple[()] = ()

    def init_attrs(self: SCItlm) -> None:
        """Initialize class attributes."""
        self.hdr = None
        self.tlm = None
        self.tstamp = None
        self.images = ()

    @property
    def size(self: SCItlm) -> int:
        """Return number of elements."""
        return 0 if self.tlm is None else len(self.tlm)

    def copy(self: SCItlm) -> SCItlm:
        """Return deep-copy of SCItlm object."""
        sci = SCItlm()
        sci.hdr = self.hdr.copy()
        sci.tlm = self.tlm.copy()
        sci.tstamp = self.tstamp.copy()
        sci.images = tuple(x for x in self.images)

    def sel(self: SCItlm, mask: np.NDArray[bool]) -> SCItlm:
        """Return subset of SCItlm object using a mask array."""
        sci = SCItlm()
        sci.hdr = self.hdr[mask]
        sci.tlm = self.tlm[mask]
        sci.tstamp = self.tstamp[mask]
        sci.images = tuple(x for x, y in zip(self.images, mask, strict=True) if y)
        return sci

    def extract_l0_sci(self: SCItlm, ccsds_sci: tuple, epoch: dt.datetime) -> int:
        """Extract SPEXone level-0 Science-telemetry data.

        Parameters
        ----------
        ccsds_sci :  tuple[np.ndarray, ...]
           SPEXone level-0 Science-telemetry packets
        epoch :  dt.datetime
           Epoch of the telemetry packets (1958 or 1970)

        Returns
        -------
        int
            number of detector frames

        """
        self.init_attrs()
        if not ccsds_sci:
            return 0

        n_frames = 0
        hdr_dtype = None
        hk_dtype = None
        found_start_first = False
        for buf in ccsds_sci:
            ccsds_hdr = CCSDShdr(buf["hdr"][0])
            if ccsds_hdr.grouping_flag == 1:
                found_start_first = True
                if n_frames == 0:
                    hdr_dtype = buf["hdr"].dtype
                    hk_dtype = buf["hk"].dtype
                    continue

            if not found_start_first:
                continue

            if ccsds_hdr.grouping_flag == 2:
                found_start_first = False
                n_frames += 1

        # do we have any complete detector images (Note ccsds_sci not empty!)?
        # print(f"n_frames: {n_frames}")
        if n_frames == 0:
            return 0

        # allocate memory
        self.hdr = np.empty(n_frames, dtype=hdr_dtype)
        self.tlm = np.empty(n_frames, dtype=hk_dtype)
        self.tstamp = np.empty(n_frames, dtype=TSTAMP_TYPE)

        # extract data from ccsds_sci
        ii = 0
        img = None
        found_start_first = False
        for buf in ccsds_sci:
            ccsds_hdr = CCSDShdr(buf["hdr"][0])
            if ccsds_hdr.grouping_flag == 1:
                found_start_first = True
                self.hdr[ii] = buf["hdr"]
                self.tlm[ii] = buf["hk"]
                self.tstamp[ii] = (
                    buf["icu_tm"]["tai_sec"][0],
                    buf["icu_tm"]["sub_sec"][0],
                    epoch
                    + dt.timedelta(
                        seconds=int(buf["icu_tm"]["tai_sec"][0]),
                        microseconds=subsec2musec(buf["icu_tm"]["sub_sec"][0]),
                    ),
                )
                img = (buf["frame"][0],)
                continue

            if not found_start_first:
                continue

            if ccsds_hdr.grouping_flag == 0:
                img += (buf["frame"][0],)
            elif ccsds_hdr.grouping_flag == 2:
                found_start_first = False
                img += (buf["frame"][0],)
                self.images += (np.concatenate(img),)
                ii += 1
                if ii == n_frames:
                    break

        return n_frames

    def extract_l1a_sci(self: SCItlm, fid: h5py.File, mps_id: int | None) -> None:
        """Extract data from SPEXone level-1a Science-telemetry packets.

        Parameters
        ----------
        fid :  h5py.File
           File pointer to a SPEXone level-1a product
        mps_id : int, optional
           Select data performed with MPS equals 'mps_id'

        """
        # pylint: disable=no-member
        self.init_attrs()

        # read science telemetry
        dset = fid["/science_data/detector_telemetry"]
        if mps_id is None:
            data_sel = np.s_[:]
        else:
            data_sel = mask2slice(dset.fields("MPS_ID")[:] == mps_id)
            if data_sel is None:
                return
        self.tlm = fid["/science_data/detector_telemetry"][data_sel]

        # determine time-stamps
        dset = fid["/image_attributes/icu_time_sec"]
        seconds = dset[data_sel]
        try:
            _ = dset.attrs["units"].index(b"1958")
        except ValueError:
            epoch = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)
        else:
            epoch = dt.datetime(1958, 1, 1, tzinfo=dt.UTC)
            epoch -= dt.timedelta(seconds=get_leap_seconds(seconds[0]))
        subsec = fid["/image_attributes/icu_time_subsec"][data_sel]

        _dt = []
        for ii, sec in enumerate(seconds):
            msec_offs = self.readout_offset(ii)
            _dt.append(
                epoch
                + dt.timedelta(
                    seconds=int(sec),
                    milliseconds=-msec_offs,
                    microseconds=subsec2musec(subsec[ii]),
                )
            )

        self.tstamp = np.empty(len(seconds), dtype=TSTAMP_TYPE)
        self.tstamp["tai_sec"] = seconds
        self.tstamp["sub_sec"] = subsec
        self.tstamp["dt"] = _dt

        # read image data
        self.images = fid["/science_data/detector_images"][data_sel, :]

    def adc_gain(self: SCItlm, indx: int | None = None) -> np.ndarray:
        """Return ADC gain [Volt]."""
        if indx is None:
            indx = np.s_[:]
        return self.tlm["DET_ADCGAIN"][indx]

    def pga_gain(self: SCItlm, indx: int | None = None) -> np.ndarray:
        """Return PGA gain [Volt]."""
        if indx is None:
            indx = np.s_[:]

        # need first bit of address 121
        reg_pgagainfactor = self.tlm["DET_BLACKCOL"][indx] & 0x1

        reg_pgagain = self.tlm["DET_PGAGAIN"][indx]

        return (1 + 0.2 * reg_pgagain) * 2**reg_pgagainfactor

    def exposure_time(self: SCItlm, indx: int | None = None) -> np.ndarray:
        """Return exposure time [ms]."""
        if indx is None:
            indx = np.s_[:]
        return 129e-4 * (
            0.43 * self.tlm["DET_FOTLEN"][indx] + self.tlm["DET_EXPTIME"][indx]
        )

    def frame_period(self: SCItlm, indx: int) -> float:
        """Return frame period of detector measurement [ms]."""
        n_coad = self.tlm["REG_NCOADDFRAMES"][indx]
        # binning mode
        if self.tlm["REG_FULL_FRAME"][indx] == 2:
            return float(n_coad * DET_CONSTS["FTI_science"])

        # full-frame mode
        return float(
            n_coad
            * np.clip(
                DET_CONSTS["FTI_margin"]
                + DET_CONSTS["overheadTime"]
                + self.exposure_time(indx),
                a_min=DET_CONSTS["FTI_diagnostic"],
                a_max=None,
            )
        )

    def readout_offset(self: SCItlm, indx: int) -> float:
        """Return offset wrt start-of-integration [ms]."""
        n_coad = self.tlm["REG_NCOADDFRAMES"][indx]
        n_frm = (
            n_coad + 3
            if self.tlm["IMRLEN"][indx] == FULLFRAME_BYTES
            else 2 * n_coad + 2
        )

        return n_frm * self.frame_period(indx)

    def binning_table(self: SCItlm) -> np.ndarray:
        """Return binning table identifier (zero for full-frame images)."""
        bin_tbl = np.zeros(len(self.tlm), dtype="i1")
        _mm = (self.tlm["IMRLEN"] == FULLFRAME_BYTES) | (self.tlm["IMRLEN"] == 0)
        if np.sum(_mm) == len(self.tlm):
            return bin_tbl

        bin_tbl_start = self.tlm["REG_BINNING_TABLE_START"]
        bin_tbl[~_mm] = 1 + (bin_tbl_start[~_mm] - 0x80000000) // 0x400000
        return bin_tbl

    def digital_offset(self: SCItlm) -> np.ndarray:
        """Return digital offset including ADC offset [count]."""
        buff = self.tlm["DET_OFFSET"].astype("i4")
        buff[buff >= 8192] -= 16384

        return buff + 70

    def convert(self: SCItlm, key: str) -> np.ndarray:
        """Convert telemetry parameter to physical units.

        Parameters
        ----------
        key :  str
           Name of telemetry parameter

        Returns
        -------
        np.ndarray

        """
        if key.upper() not in self.tlm.dtype.names:
            raise KeyError(
                f"Parameter: {key.upper()} not found" f" in {self.tlm.dtype.names}"
            )

        raw_data = np.array([x[key.upper()] for x in self.tlm])
        return convert_hk(key.upper(), raw_data)
