#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2023-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""`SCItlm`, class to read/access PACE/SPEXone telemetry data."""

from __future__ import annotations

__all__ = ["DET_CONSTS", "SCItlm"]

import datetime as dt
import logging
from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from .ccsds_hdr import CCSDShdr
from .tlm_utils import convert_hk

if TYPE_CHECKING:
    import h5py
    from numpy.typing import ArrayLike, NDArray


# - global parameters -----------------------
DET_CONSTS = {
    "dimRow": 2048,
    "dimColumn": 2048,
    "dimFullFrame": 2048 * 2048,
    "DEM_frequency": 10,  # [MHz]
    "FOT_diagnostic": 0.4644,  # [ms]
    "FTI_diagnostic": 240.0,  # [ms]
    "FTI_margin": 212.4,  # [ms]
    "FOT_science": 0.3096,  # [ms]
    "FTI_science": 1000 / 15,  # [ms]
    "FOT_length": 20,
}
FULLFRAME_BYTES = 2 * DET_CONSTS["dimFullFrame"]

TSTAMP_MIN = 1561939200  # 2019-07-01T00:00:00+00:00
TSTAMP_TYPE = np.dtype([("tai_sec", int), ("sub_sec", int), ("dt", "O")])

module_logger = logging.getLogger("pyspex.lib.sci_tlm")


# - helper functions ------------------------
def subsec2musec(sub_sec: int) -> int:
    """Return subsec as microseconds."""
    return int(1e6 * sub_sec / 65536)


def mask2slice(mask: NDArray[bool]) -> None | slice | tuple | NDArray[bool]:
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
        self.hdr: NDArray | None = None
        self.tlm: NDArray | None = None
        self.tstamp: NDArray | None = None
        self.images: tuple[NDArray, ...] | tuple[()] = ()

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
        sci.images = copy(self.images)
        return sci

    def sel(self: SCItlm, mask: ArrayLike[bool]) -> SCItlm:
        """Return subset of SCItlm object using a mask array."""
        sci = SCItlm()
        if self.hdr is not None:
            sci.hdr = self.hdr[mask]
        if self.tlm is not None:
            sci.tlm = self.tlm[mask]
            sci.tstamp = self.tstamp[mask]
            sci.images = tuple(self.images[ii] for ii in mask.nonzero()[0])
        return sci

    def vsel(self: SCItlm, mask: ArrayLike[bool]) -> SCItlm:
        """Return subset of SCItlm object generated using method vstack.

        Notes
        -----
        The methods `append` and `vsel` only work for complete measurements
        with the same MPS.

        """
        sci = SCItlm()
        if self.hdr is not None:
            sci.hdr = self.hdr[mask]
        if self.tlm is not None:
            sci.tlm = self.tlm[mask]
            sci.tstamp = self.tstamp[mask]
            sci.images = self.images[0][mask, :]
        return sci

    def append(self: SCItlm, sci: SCItlm) -> None:
        """Append one SCItlm object to the current.

        Notes
        -----
        The methods `append` and `vsel` only work for complete measurements
        with the same MPS.

        """
        self.hdr = sci.hdr if self.hdr is None else np.append(self.hdr, sci.hdr)
        self.tlm = sci.tlm if self.tlm is None else np.append(self.tlm, sci.tlm)
        self.tstamp = (
            sci.tstamp if self.tstamp is None else np.append(self.tstamp, sci.tstamp)
        )
        if isinstance(sci.images, tuple):
            self.images = (
                (sci.images[0],)
                if len(self.images) == 0
                else (np.concatenate((self.images[0], sci.images[0])),)
            )
        else:
            self.images = (
                sci.images
                if len(self.images) == 0
                else np.concatenate((self.images, sci.images))
            )

    def extract_l0_sci(
        self: SCItlm, ccsds_sci: tuple[NDArray], epoch: dt.datetime
    ) -> int:
        """Extract SPEXone level-0 Science-telemetry data.

        Parameters
        ----------
        ccsds_sci :  tuple[NDArray, ...]
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
                        milliseconds=-self.readout_offset(ii),
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
                img = np.concatenate(img)
                if img.size == self.tlm[ii]["IMRLEN"] // 2:
                    self.images += (img,)
                    ii += 1
                else:
                    n_frames -= 1
                if ii == n_frames:
                    break

        # adjust number of frames for corrupted images
        if ii != self.hdr.size:
            self.hdr = self.hdr[:ii]
            self.tlm = self.tlm[:ii]
            self.tstamp = self.tstamp[:ii]

        return n_frames

    def extract_l1a_sci(self: SCItlm, fid: h5py.File, mps_id: int | None) -> None:
        """Extract data from SPEXone level-1a Science-telemetry packets.

        Parameters
        ----------
        fid :  h5py.File
           A HDF5 file pointer to a SPEXone level-1a product
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
        dset = fid["/image_attributes/image_time"]
        indx = dset.attrs["units"].decode().find("20")
        ref_date = dt.datetime.fromisoformat(
            dset.attrs["units"].decode()[indx:] + "+00:00"
        )
        self.tstamp = np.empty(len(dset[data_sel]), dtype=TSTAMP_TYPE)
        self.tstamp["tai_sec"] = fid["/image_attributes/icu_time_sec"][data_sel]
        self.tstamp["sub_sec"] = fid["/image_attributes/icu_time_subsec"][data_sel]
        self.tstamp["dt"] = [
            ref_date + dt.timedelta(seconds=float(x)) for x in dset[data_sel]
        ]

        # read image data
        self.images = (fid["/science_data/detector_images"][data_sel, :],)

    def adc_gain(self: SCItlm, indx: int | None = None) -> NDArray[np.int32]:
        """Return ADC gain [Volt]."""
        if indx is None:
            indx = np.s_[:]
        return self.tlm["DET_ADCGAIN"][indx]

    def pga_gain(self: SCItlm, indx: int | None = None) -> NDArray[float]:
        """Return PGA gain [Volt]."""
        if indx is None:
            indx = np.s_[:]

        # need first bit of address 121
        reg_pgagainfactor = self.tlm["DET_BLACKCOL"][indx] & 0x1

        reg_pgagain = self.tlm["DET_PGAGAIN"][indx]

        return (1 + 0.2 * reg_pgagain) * 2**reg_pgagainfactor

    def exposure_time(self: SCItlm, indx: int | None = None) -> float | NDArray[float]:
        """Return exposure time [ms]."""
        if indx is None:
            indx = np.s_[:]
        return 129e-4 * (
            0.43 * self.tlm["DET_FOTLEN"][indx] + self.tlm["DET_EXPTIME"][indx]
        )

    def frame_period(self: SCItlm, indx: int | None = None) -> float | NDArray[float]:
        """Return frame period of detector measurement [ms]."""
        if indx is None:
            res = np.zeros(self.tlm.size, dtype="f8")
            _mm = (self.tlm["REG_FULL_FRAME"] & 0x3) == 2
            # binning mode
            if np.sum(_mm) > 0:
                res[_mm] = DET_CONSTS["FTI_science"]

            # full-frame mode
            _mm = ~_mm
            if np.sum(_mm) > 0:
                res[_mm] = np.clip(
                    DET_CONSTS["FTI_margin"]
                    + DET_CONSTS["FOT_diagnostic"]
                    + self.exposure_time(_mm),
                    a_min=DET_CONSTS["FTI_diagnostic"],
                    a_max=None,
                )
            return res

        # binning mode
        if (self.tlm["REG_FULL_FRAME"][indx] & 0x3) == 2:
            return DET_CONSTS["FTI_science"]

        # full-frame mode
        return np.clip(
            DET_CONSTS["FTI_margin"]
            + DET_CONSTS["FOT_diagnostic"]
            + self.exposure_time(indx),
            a_min=DET_CONSTS["FTI_diagnostic"],
            a_max=None,
        )

    def master_cycle(self: SCItlm, indx: int) -> float:
        """Return master-cycle period."""
        return self.tlm["REG_NCOADDFRAMES"][indx] * self.frame_period(indx)

    def readout_offset(self: SCItlm, indx: int) -> float:
        """Return offset wrt start-of-integration [ms]."""
        n_coad = self.tlm["REG_NCOADDFRAMES"][indx]
        n_frm = (
            n_coad + 1
            if self.tlm["IMRLEN"][indx] == FULLFRAME_BYTES
            else 2 * n_coad + 1
        )
        return n_frm * self.frame_period(indx)

    def binning_table(self: SCItlm) -> NDArray[np.int8]:
        """Return binning table identifier (zero for full-frame images)."""
        bin_tbl = np.zeros(len(self.tlm), dtype="i1")
        _mm = (self.tlm["IMRLEN"] == FULLFRAME_BYTES) | (self.tlm["IMRLEN"] == 0)
        if np.sum(_mm) == len(self.tlm):
            return bin_tbl

        bin_tbl_start = self.tlm["REG_BINNING_TABLE_START"]
        bin_tbl[~_mm] = 1 + (bin_tbl_start[~_mm] - 0x80000000) // 0x400000
        return bin_tbl

    def digital_offset(self: SCItlm) -> NDArray[np.int32]:
        """Return digital offset including ADC offset [count]."""
        buff = self.tlm["DET_OFFSET"].astype("i4")
        buff[buff >= 8192] -= 16384

        return buff + 70

    def convert(self: SCItlm, key: str) -> NDArray[float]:
        """Convert telemetry parameter to physical units.

        Parameters
        ----------
        key :  str
           Name of telemetry parameter

        Returns
        -------
        NDArray[float]

        """
        parm = key.upper()
        raw_data = np.array([x[parm] for x in self.tlm])
        return convert_hk(parm, raw_data)
