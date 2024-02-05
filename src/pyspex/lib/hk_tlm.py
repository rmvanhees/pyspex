#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2023-2024 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""`HKtlm`, class to read/access PACE/SPEXone telemetry data."""
from __future__ import annotations

__all__ = ["HKtlm"]

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np

from .ccsds_hdr import CCSDShdr
from .tlm_utils import convert_hk
from .tmtc_def import tmtc_dtype

if TYPE_CHECKING:
    import h5py
    import numpy.typing as npt


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


# - class HKtlm ----------------------------
class HKtlm:
    """Class to handle SPEXone housekeeping telemetry packets."""

    def __init__(self: HKtlm) -> None:
        """Initialize HKtlm object."""
        self.hdr: np.ndarray | None = None
        self.tlm: np.ndarray | None = None
        self.tstamp: list[dt.datetime, ...] | list = []
        self.events: list[np.ndarray, ...] | list = []

    def init_attrs(self: HKtlm) -> None:
        """Initialize class attributes."""
        self.hdr = None
        self.tlm = None
        self.tstamp = []
        self.events = []

    @property
    def size(self: HKtlm) -> int:
        """Return number of elements."""
        return 0 if self.tlm is None else len(self.tlm)

    def copy(self: HKtlm) -> HKtlm:
        """Return deep-copy of HKtlm object."""
        hkt = HKtlm()
        hkt.hdr = self.hdr.copy()
        hkt.tlm = self.tlm.copy()
        hkt.tstamp = self.tstamp.copy()
        hkt.events = self.events.copy()
        return hkt

    def sel(self: HKtlm, mask: np.NDArray[bool]) -> HKtlm:
        """Return subset of HKtlm object using a mask array."""
        hkt = HKtlm()
        hkt.hdr = self.hdr[mask]
        hkt.tlm = self.tlm[mask]
        hkt.tstamp = [x for x, y in zip(self.tstamp, mask, strict=True) if y]
        hkt.events = self.events.copy()
        return hkt

    def extract_l0_hk(self: HKtlm, ccsds_hk: tuple, epoch: dt.datetime) -> None:
        """Extract data from SPEXone level-0 housekeeping telemetry packets.

        Parameters
        ----------
        ccsds_hk :  tuple[np.ndarray, ...]
           SPEXone level-0 housekeeping telemetry packets
        epoch :  dt.datetime
           Epoch of the telemetry packets (1958 or 1970)

        """
        self.init_attrs()
        if not ccsds_hk:
            return

        self.hdr = np.empty(len(ccsds_hk), dtype=ccsds_hk[0]["hdr"].dtype)
        self.tlm = np.empty(len(ccsds_hk), dtype=tmtc_dtype(0x320))
        ii = 0
        for buf in ccsds_hk:
            ccsds_hdr = CCSDShdr(buf["hdr"][0])

            # Catch TcAccept, TcReject, TcExecute, TcFail and EventRp as events
            if ccsds_hdr.apid != 0x320 or buf["hdr"]["tai_sec"] < len(ccsds_hk):
                if 0x331 <= ccsds_hdr.apid <= 0x335:
                    self.events.append(buf)
                continue

            self.hdr[ii] = buf["hdr"]
            self.tlm[ii] = buf["hk"]
            self.tstamp.append(
                epoch
                + dt.timedelta(
                    seconds=int(buf["hdr"]["tai_sec"][0]),
                    microseconds=subsec2musec(buf["hdr"]["sub_sec"][0]),
                )
            )
            ii += 1

        self.hdr = self.hdr[:ii]
        self.tlm = self.tlm[:ii]

        # These values are originally stored in little-endian, but
        # Numpy does not accept a mix of little & big-endian values
        # in a structured array.
        self.tlm["HTR1_CALCPVAL"][:] = self.tlm["HTR1_CALCPVAL"].byteswap()
        self.tlm["HTR2_CALCPVAL"][:] = self.tlm["HTR2_CALCPVAL"].byteswap()
        self.tlm["HTR3_CALCPVAL"][:] = self.tlm["HTR3_CALCPVAL"].byteswap()
        self.tlm["HTR4_CALCPVAL"][:] = self.tlm["HTR4_CALCPVAL"].byteswap()
        self.tlm["HTR1_CALCIVAL"][:] = self.tlm["HTR1_CALCIVAL"].byteswap()
        self.tlm["HTR2_CALCIVAL"][:] = self.tlm["HTR2_CALCIVAL"].byteswap()
        self.tlm["HTR3_CALCIVAL"][:] = self.tlm["HTR3_CALCIVAL"].byteswap()
        self.tlm["HTR4_CALCIVAL"][:] = self.tlm["HTR4_CALCIVAL"].byteswap()

    def extract_l1a_hk(self: HKtlm, fid: h5py.File, mps_id: int | None) -> None:
        """Extract data from SPEXone level-1a housekeeping telemetry packets.

        Parameters
        ----------
        fid :  h5py.File
           File pointer to a SPEXone level-1a product
        mps_id : int, optional
           Select data performed with MPS equals 'mps_id'

        """
        self.init_attrs()

        # pylint: disable=no-member
        dset = fid["/engineering_data/NomHK_telemetry"]
        if mps_id is None:
            data_sel = np.s_[:]
        else:
            data_sel = mask2slice(dset.fields("MPS_ID")[:] == mps_id)
            if data_sel is None:
                return
        self.tlm = dset[data_sel]

        dset = fid["/engineering_data/HK_tlm_time"]
        ref_date = dset.attrs["units"].decode()[14:] + "+00:00"
        epoch = dt.datetime.fromisoformat(ref_date)
        for sec in dset[data_sel]:
            self.tstamp.append(epoch + dt.timedelta(seconds=sec))

    def convert(self: HKtlm, key: str) -> np.ndarray:
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
