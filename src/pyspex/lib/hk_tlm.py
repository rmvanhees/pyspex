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
"""`HKtlm`, class to read/access PACE/SPEXone telemetry data."""

from __future__ import annotations

__all__ = ["HKtlm"]

import datetime as dt
from copy import copy
from typing import TYPE_CHECKING

import numpy as np

from .ccsds_hdr import CCSDShdr
from .tlm_utils import CONV_DICT, HkFlagging, convert_hk
from .tmtc_def import tmtc_dtype

if TYPE_CHECKING:
    import h5py
    from numpy.typing import NDArray


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


# - class HKtlm ----------------------------
class HKtlm:
    """Class to handle SPEXone housekeeping telemetry packets."""

    def __init__(self: HKtlm) -> None:
        """Initialize HKtlm object."""
        self.hdr: NDArray | None = None
        self.tlm: NDArray | None = None
        self.tstamp: list[dt.datetime, ...] | list = []
        self.events: list[NDArray, ...] | list = []

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
        hkt.tstamp = copy(self.tstamp)
        hkt.events = copy(self.events)
        return hkt

    def sel(self: HKtlm, mask: NDArray[bool]) -> HKtlm:
        """Return subset of HKtlm object using a mask array."""
        hkt = HKtlm()
        if self.hdr is not None:
            hkt.hdr = self.hdr[mask]
        if self.tlm is not None:
            hkt.tlm = self.tlm[mask]
            hkt.tstamp = [x for x, y in zip(self.tstamp, mask, strict=True) if y]
            hkt.events = self.events.copy()
        return hkt

    def extract_l0_hk(self: HKtlm, ccsds_hk: tuple, epoch: dt.datetime) -> None:
        """Extract data from SPEXone level-0 housekeeping telemetry packets.

        Parameters
        ----------
        ccsds_hk :  tuple[NDArray, ...]
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
           A HDF5 file pointer to a SPEXone level-1a product
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

    def convert(self: HKtlm, key: str) -> NDArray:
        """Convert telemetry parameter to physical units.

        Parameters
        ----------
        key :  str
           Name of telemetry parameter

        Returns
        -------
        NDArray

        """
        parm = key.upper()
        if parm.endswith("_POWER"):
            # HTR?_POWER is not defined in the telemetry structure, HTR?_I is available
            # However, convert_hk will convert raw heater currents to Watt
            raw_data = np.array([x[parm.replace("_POWER", "_I")] for x in self.tlm])
        else:
            raw_data = np.array([x[parm] for x in self.tlm])
        return convert_hk(parm, raw_data)

    def check(self: HKtlm, key: str) -> NDArray:
        """Check of parameter is out-of-range or changed of value."""
        try:
            values = self.convert(key)
        except KeyError as exc:
            raise RuntimeError from exc
        res = np.full(values.size, HkFlagging.NOMINAL)

        valid_range = CONV_DICT[key.upper()]["range"]
        if valid_range is None:
            # if no range is provided for key then check where its value has changed
            res[np.diff(values) != 0] = HkFlagging.CHANGED
            return res

        # flag too small values (value of flag depend on units of parameter)
        if (flag := HkFlagging.get_flag(key.upper(), too_low=True)) is not None:
            res[values < valid_range[0]] = flag
        # flag too large values (value of flag depend on units of parameter)
        if (flag := HkFlagging.get_flag(key.upper(), too_low=False)) is not None:
            res[values > valid_range[1]] = flag

        return res
