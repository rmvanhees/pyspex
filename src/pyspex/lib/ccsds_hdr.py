#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Read Primary and Secondary CCSDS headers and obtain their parameters."""

from __future__ import annotations

__all__ = ["CCSDShdr"]

import datetime as dt
from typing import TYPE_CHECKING

import numpy as np

from .tmtc_def import tmtc_dtype

if TYPE_CHECKING:
    import numpy.typing as npt


class CCSDShdr:
    """Read CCSDS telemetry packet structure.

    Which consists of the primary header: version, type, apid, grouping flag,
    sequence count and packet length, and the secondary header: tai_sec and
    sub_sec.

    """

    def __init__(self: CCSDShdr, hdr: np.ndarray | None = None) -> None:
        """Initialise the class instance.

        Parameters
        ----------
        hdr :  np.ndarray, optional
           CCSDS primary and secondary headers

        """
        self.__dtype = None
        self.__hdr = None
        if hdr is not None:
            self.__hdr = hdr
            self.__dtype = hdr.dtype

    def _tm_raw_(self: CCSDShdr) -> np.dtype:  # ApID unknown
        """Return data-type of unknown packet, just a header and byte data."""
        return np.dtype(
            [("hdr", self.__hdr.dtype), ("Data", "u1", (self.__hdr["length"] - 5))]
        )

    def _tm_800_(self: CCSDShdr) -> np.dtype:  # ApID = 0x320
        """Return data-type of NomHk packet."""
        return np.dtype([("hdr", self.__hdr.dtype), ("hk", tmtc_dtype(0x320))])

    def _tm_802_(self: CCSDShdr) -> np.dtype:  # ApID = 0x322
        """Return data-type of DemHk packet."""
        return np.dtype([("hdr", self.__hdr.dtype), ("hk", tmtc_dtype(0x322))])

    def _tm_817_(self: CCSDShdr) -> np.dtype:  # ApID = 0x331
        """Return data-type of TcAccept packet."""
        return np.dtype(
            [("hdr", self.__hdr.dtype), ("TcPacketId", ">u2"), ("TcSeqControl", ">u2")]
        )

    def _tm_818_(self: CCSDShdr) -> np.dtype:  # ApID = 0x332
        """Return data-type of TcReject packet."""
        return np.dtype(
            [
                ("hdr", self.__hdr.dtype),
                ("TcPacketId", ">u2"),
                ("TcSeqControl", ">u2"),
                ("TcRejectCode", ">u2"),
                ("RejectParameter1", ">u2"),
                ("RejectParameter2", ">u2"),
            ]
        )

    def _tm_819_(self: CCSDShdr) -> np.dtype:  # ApID = 0x333
        """Return data-type of TcExecute packet."""
        return np.dtype(
            [("hdr", self.__hdr.dtype), ("TcPacketId", ">u2"), ("TcSeqControl", ">u2")]
        )

    def _tm_820_(self: CCSDShdr) -> np.dtype:  # ApID = 0x334
        """Return data-type of TcFail packet."""
        return np.dtype(
            [
                ("hdr", self.__hdr.dtype),
                ("TcPacketId", ">u2"),
                ("TcSeqControl", ">u2"),
                ("TcFailCode", ">u2"),
                ("FailParameter1", ">u2"),
                ("FailParameter2", ">u2"),
            ]
        )

    def _tm_821_(self: CCSDShdr) -> np.dtype:  # ApID = 0x335
        """Return data-type of EventRp packet."""
        return np.dtype(
            [
                ("hdr", self.__hdr.dtype),
                ("Event_ID", "u1"),
                ("Event_Sev", "u1"),
                ("Word1", ">u2"),
                ("Word2", ">u2"),
                ("Word3", ">u2"),
                ("Word4", ">u2"),
                ("Word5", ">u2"),
                ("Word6", ">u2"),
                ("Word7", ">u2"),
                ("Word8", ">u2"),
            ]
        )

    def _tm_832_(self: CCSDShdr) -> np.dtype:  # ApID = 0x340
        """Return data-type of MemDump packet."""
        if self.__hdr["length"] - 15 == 1:
            return np.dtype(
                [
                    ("hdr", self.__hdr.dtype),
                    ("Image_ID", "u1"),
                    ("_FillerByte", "u1"),
                    ("Address32", ">u4"),
                    ("Length", ">u4"),
                    ("Data", "u1"),
                ]
            )

        return np.dtype(
            [
                ("hdr", self.__hdr.dtype),
                ("Image_ID", "u1"),
                ("_FillerByte", "u1"),
                ("Address32", ">u4"),
                ("Length", ">u4"),
                (
                    "Data",
                    "u1",
                    (self.__hdr["length"] - 15),
                ),
            ]
        )

    def _tm_833_(self: CCSDShdr) -> np.dtype:  # ApID = 0x341
        """Return data-type of MemCheckRp packet."""
        return np.dtype(
            [
                ("hdr", self.__hdr.dtype),
                ("Image_ID", "u1"),
                ("_FillerByte", "u1"),
                ("Address32", ">u4"),
                ("Length", ">u4"),
                ("CheckSum", ">u4"),
            ]
        )

    def _tm_826_(self: CCSDShdr) -> np.dtype:  # ApID = 0x33A
        """Return data-type of MonListRp packet."""
        mon_dtype = np.dtype(
            [
                ("Mon1_EnSts", ">u2"),
                ("Mon1_ParID", ">u2"),
                ("Mon1_Int", "u1"),
                ("Mon1_NofSampl", "u1"),
                ("Mon1_CheckType", ">u2"),
                ("Mon1_LowOrExpCheckVal", ">u4"),
                ("Mon1_LowOrExpCheckRpId", ">u4"),
                ("Mon1_UppOrExpCheckVal", ">u4"),
                ("Mon1_UppOrExpCheckRpId", ">u4"),
            ]
        )
        num = (self.__hdr["length"] - 5) // mon_dtype.itemsize
        return np.dtype([("hdr", self.__hdr.dtype), ("Report", mon_dtype, (num,))])

    def _tm_827_(self: CCSDShdr) -> np.dtype:  # ApID = 0x33B
        """Return data-type of EvRpListRp packet."""
        return np.dtype(
            [
                ("hdr", self.__hdr.dtype),
                (
                    "Data",
                    "u1",
                    (self.__hdr["length"] - 5),
                ),
            ]
        )

    def _tm_828_(self: CCSDShdr) -> np.dtype:  # ApID = 0x33C
        """Return data-type of MpsTableRp packet."""
        return np.dtype(
            [
                ("hdr", self.__hdr.dtype),
                ("MPS_ID", "u1"),
                ("MPS_VER", "u1"),
                ("FTO", ">u2"),
                ("FTI", ">u2"),
                ("FTC", ">u2"),
                ("IMRO", ">u2"),
                ("IMRSA_A", ">u4"),
                ("IMRSA_B", ">u4"),
                ("IMRLEN", ">u4"),
                ("PKTLEN", ">u2"),
                ("TMRO", ">u2"),
                ("TMRI", ">u2"),
                ("IMDMODE", "u1"),
                ("I_LVDS", "u1"),
                ("_Filler1", ">u2"),
                ("_Filler2", ">u2"),
                ("_Filler3", ">u2"),
                ("DEM_RST", "u1"),
                ("DEM_CMV_CTRL", "u1"),
                ("COADD", "u1"),
                ("DEM_IGEN", "u1"),
                ("FRAME_MODE", "u1"),
                ("OUTPMODE", "u1"),
                ("BIN_TBL", ">u4"),
                ("COADD_BUF", ">u4"),
                ("COADD_RESA", ">u4"),
                ("COADD_RESB", ">u4"),
                ("FRAME_BUFA", ">u4"),
                ("FRAME_BUFB", ">u4"),
                ("LINE_ENA", ">u4"),
                ("NUMLIN", ">u2"),
                ("STR1", ">u2"),
                ("STR2", ">u2"),
                ("STR3", ">u2"),
                ("STR4", ">u2"),
                ("STR5", ">u2"),
                ("STR6", ">u2"),
                ("STR7", ">u2"),
                ("STR8", ">u2"),
                ("NUMLIN1", ">u2"),
                ("NUMLIN2", ">u2"),
                ("NUMLIN3", ">u2"),
                ("NUMLIN4", ">u2"),
                ("NUMLIN5", ">u2"),
                ("NUMLIN6", ">u2"),
                ("NUMLIN7", ">u2"),
                ("NUMLIN8", ">u2"),
                ("SUBS", ">u2"),
                ("SUBA", ">u2"),
                ("MONO", "u1"),
                ("IMFLP", "u1"),
                ("EXP_CTRL", "u1"),
                ("_FillerByte4", "u1"),
                ("EXP_TIME", ">u4"),
                ("EXP_STEP", ">u4"),
                ("EXP_KP1", ">u4"),
                ("EXP_KP2", ">u4"),
                ("NRSLOPE", "u1"),
                ("EXP_SEQ", "u1"),
                ("EXP_TIME2", ">u4"),
                ("EXP_STEP2", ">u4"),
                ("NUMFR", ">u2"),
                ("FOTLEN", "u1"),
                ("_FillerByte5", "u1"),
                ("ILVDSRCVR", "u1"),
                ("CALIB", "u1"),
                ("TRAINPTRN", ">u2"),
                ("CHENA", ">u4"),
                ("ICOL", "u1"),
                ("ICOLPR", "u1"),
                ("IADC", "u1"),
                ("IAMP", "u1"),
                ("VTFL1", "u1"),
                ("VTFL2", "u1"),
                ("VTFL3", "u1"),
                ("VRSTL", "u1"),
                ("VPRECH", "u1"),
                ("VREF", "u1"),
                ("VRAMP1", "u1"),
                ("VRAMP2", "u1"),
                ("OFFSET", ">u2"),
                ("PGAGAIN", "u1"),
                ("ADCGAIN", "u1"),
                ("TDIG1", "u1"),
                ("TDIG2", "u1"),
                ("BITMODE", "u1"),
                ("ADCRES", "u1"),
                ("PLLENA", "u1"),
                ("PLLINFRE", "u1"),
                ("PLLBYP", "u1"),
                ("PLLRATE", "u1"),
                ("PLLLOAD", "u1"),
                ("DETDUM", "u1"),
                ("BLACKCOL", "u1"),
                ("VBLACKSUN", "u1"),
                ("_Filler6", ">u4"),
                ("_Filler7", ">u4"),
            ]
        )

    def _tm_829_(self: CCSDShdr) -> np.dtype:  # ApID = 0x33D
        """Return data-type of ThemTableRp packet."""
        return np.dtype(
            [
                ("hdr", self.__hdr.dtype),
                ("HTR_1_IsEna", "u1"),
                ("HTR_1_AtcCorMan", "u1"),
                ("HTR_1_THMCH", "u1"),
                ("_FillerByte1", "u1"),
                ("HTR_1_ManOutput", ">u2"),
                ("HTR_1_ATC_SP", ">u4"),
                ("HTR_1_ATC_P", ">u4"),
                ("HTR_1_ATC_I", ">u4"),
                ("HTR_1_ATC_I_INIT", ">u4"),
                ("HTR_2_IsEna", "u1"),
                ("HTR_2_AtcCorMan", "u1"),
                ("HTR_2_THMCH", "u1"),
                ("_FillerByte2", "u1"),
                ("HTR_2_ManOutput", ">u2"),
                ("HTR_2_ATC_SP", ">u4"),
                ("HTR_2_ATC_P", ">u4"),
                ("HTR_2_ATC_I", ">u4"),
                ("HTR_2_ATC_I_INIT", ">u4"),
                ("HTR_3_IsEna", "u1"),
                ("HTR_3_AtcCorMan", "u1"),
                ("HTR_3_THMCH", "u1"),
                ("_FillerByte3", "u1"),
                ("HTR_3_ManOutput", ">u2"),
                ("HTR_3_ATC_SP", ">u4"),
                ("HTR_3_ATC_P", ">u4"),
                ("HTR_3_ATC_I", ">u4"),
                ("HTR_3_ATC_I_INIT", ">u4"),
                ("HTR_4_IsEna", "u1"),
                ("HTR_4_AtcCorMan", "u1"),
                ("HTR_4_THMCH", "u1"),
                ("_FillerByte4", "u1"),
                ("HTR_4_ManOutput", ">u2"),
                ("HTR_4_ATC_SP", ">u4"),
                ("HTR_4_ATC_P", ">u4"),
                ("HTR_4_ATC_I", ">u4"),
                ("HTR_4_ATC_I_INIT", ">u4"),
            ]
        )

    @property
    def hdr(self: CCSDShdr) -> np.ndarray:
        """Structured array holding the CCSDS header."""
        return self.__hdr

    @property
    def dtype(self: CCSDShdr) -> np.dtype:
        """Return numpy date-type of CCSDS headers."""
        return self.__dtype

    @property
    def version(self: CCSDShdr) -> npt.NDArray[int]:
        """Return zero to indicate that this is a version 1 packet."""
        return (self.__hdr["type"] >> 13) & 0x7

    @property
    def type(self: CCSDShdr) -> npt.NDArray[int]:
        """Return zero to indicate that this is a telemetry packet."""
        return (self.__hdr["type"] >> 12) & 0x1

    @property
    def apid(self: CCSDShdr) -> npt.NDArray[int]:
        """Return ApID: an identifier for this telemetry packet.

        Notes
        -----
        SPEXone uses the following APIDs:

        - 0x320:  NomHk
        - 0x322:  DemHk
        - 0x331:  TcAccept
        - 0x332:  TcReject
        - 0x333:  TcExecute
        - 0x334:  TcFail
        - 0x335:  EventRp
        - 0x33A:  MonListRp
        - 0x33B:  EvRpListRp
        - 0x33C:  MpsTableRp
        - 0x33D:  ThermTableRp
        - 0x340:  MemDump
        - 0x341:  MemCheckRp
        - 0x350:  Science

        """
        return self.__hdr["type"] & 0x7FF

    @property
    def grouping_flag(self: CCSDShdr) -> npt.NDArray[int]:
        """Data packages can be segmented.

        Note:
        ----
        This flag is encoded as::

         00 : continuation segment
         01 : first segment
         10 : last segment
         11 : unsegmented

        """
        return (self.__hdr["sequence"] >> 14) & 0x3

    @property
    def sequence(self: CCSDShdr) -> npt.NDArray[int]:
        """Return the sequence counter.

        This counter is incremented with each consecutive packet of a
        particular ApID. This value will roll over to 0 after 0x3FF is reached.
        """
        return self.__hdr["sequence"] & 0x3FFF

    @property
    def packet_size(self: CCSDShdr) -> npt.NDArray[int]:
        """Returns the CCSDS packet-length.

        Which is equal to the number of bytes in the Secondary header plus
        User Data minus 1.
        """
        return self.__hdr["length"]

    @property
    def tai_sec(self: CCSDShdr) -> npt.NDArray[int]:
        """Seconds since 1958 (TAI)."""
        return self.__hdr["tai_sec"]

    @property
    def sub_sec(self: CCSDShdr) -> npt.NDArray[int]:
        """Sub-seconds (1 / 2**16)."""
        return self.__hdr["sub_sec"]

    @property
    def data_dtype(self: CCSDShdr) -> np.dtype:
        """Return numpy data-type of CCSDS User Data."""
        method = getattr(self, f"_tm_{self.apid:d}_", None)
        return self._tm_raw_() if method is None else method()

    def tstamp(self: CCSDShdr, epoch: dt.datetime) -> dt.datetime:
        """Return time of the telemetry packet.

        Parameters
        ----------
        epoch :  datetime
           Provide the UTC epoch of the time (thus corrected for leap seconds)

        """
        return epoch + dt.timedelta(
            seconds=int(self.tai_sec),
            microseconds=100 * int(self.sub_sec / 65536 * 10000),
        )

    def read(self: CCSDShdr, file_format: str, buffer: bytes, offs: int = 0) -> None:
        """Read CCSDS primary and secondary headers from data.

        Parameters
        ----------
        file_format :  {'raw', 'dsb' or 'st3'}
           File format of this level 0 product
        buffer :  buffer_like
           Array of type (unsigned) byte.
        offs :  int, default=0
           Start reading the buffer from this offset (in bytes)

        Notes
        -----
        SPEXone level-0 file-formats:
        'raw'
           data has no file header and standard CCSDS packet headers
        'st3'
           data has no file header and ITOS + spacewire + CCSDS packet headers
        'dsb'
           data has a cFE file-header and spacewire + CCSDS packet headers

        """
        if file_format == "dsb":
            hdr_dtype = np.dtype(
                [
                    ("spacewire", "u1", (2,)),
                    ("type", ">u2"),
                    ("sequence", ">u2"),
                    ("length", ">u2"),
                    ("tai_sec", ">u4"),
                    ("sub_sec", ">u2"),
                ]
            )
        elif file_format == "raw":
            hdr_dtype = np.dtype(
                [
                    ("type", ">u2"),
                    ("sequence", ">u2"),
                    ("length", ">u2"),
                    ("tai_sec", ">u4"),
                    ("sub_sec", ">u2"),
                ]
            )
        elif file_format == "st3":
            hdr_dtype = np.dtype(
                [
                    ("itos_hdr", ">u2", (8,)),
                    ("spacewire", "u1", (2,)),
                    ("type", ">u2"),
                    ("sequence", ">u2"),
                    ("length", ">u2"),
                    ("tai_sec", ">u4"),
                    ("sub_sec", ">u2"),
                ]
            )
        else:
            raise ValueError("Unknown file_format, should be dsb, raw or st3")

        self.__hdr = np.frombuffer(buffer, count=1, offset=offs, dtype=hdr_dtype)[0]
        self.__dtype = hdr_dtype
