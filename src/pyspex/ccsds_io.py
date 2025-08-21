#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Contains the class `CCSDSio` to read SPEXone telemetry packets."""

from __future__ import annotations

__all__ = ["CCSDSio", "hk_sec_of_day", "img_sec_of_day"]

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Self

import numpy as np

from .lib.tmtc_def import tmtc_dtype

# - global parameters ------------------------------
FULLFRAME_BYTES = 2 * 2048 * 2048

# Define parameters of Primary header
#  - Packet type     (3 bits): Version No.
#                              Indicates this is a CCSDS version 1 packet
#                     (1 bit): Type indicator
#                              Indicates this is a telemetry packet
#                     (1 bit): Secondary flag
#                              Indicate presence of Secondary header
#                   (11 bits): ApID
#                              SPEXone ApID [0x320 - 0x351] or 2047
#
#  - Packet Sequence (2 bits): Grouping flag
#                              00 continuation packet-data segment
#                              01 first packet-data segment
#                              10 last packet-data segment
#                              11 packet-data unsegmented
#                   (14 bits): Counter per ApID, rollover to 0 at 0x3FFF
#  - Packet length  (16 bits): size of packet data in bytes (always odd)
#                              (secondary header + User data) - 1
#  - Packet timestamp:         Secondary header
#                   (32 bits): seconds (1-1-1970 UTC)
#                   (16 bits): sub-seconds (1/2 ** 16)

HDR_DTYPE = np.dtype(
    [
        ("type", ">u2"),
        ("sequence", ">u2"),
        ("length", ">u2"),
        ("tai_sec", ">u4"),
        ("sub_sec", ">u2"),
    ]
)

TIME_DTYPE = np.dtype([("tai_sec", ">u4"), ("sub_sec", ">u2")])

SCIHK_DTYPE = tmtc_dtype(0x350)

# - Error messages ------------------------
MSG_SKIP_FRAME = "[WARNING]: rejected a frame because it's incomplete"
MSG_INVALID_APID = (
    "[WARNING]: found one or more telemetry packages with an invalid APID"
)
MSG_CORRUPT_APID = "corrupted segments - detected APID 1 after <> 2"
MSG_CORRUPT_FRAME = "corrupted segments - previous frame not closed"


# - functions -----------------------------
def img_sec_of_day(
    img_sec: np.ndarray, img_subsec: np.ndarray, img_hk: np.ndarray
) -> tuple[datetime, float | Any]:
    """Convert Image CCSDS timestamp to seconds after midnight.

    Parameters
    ----------
    img_sec : numpy array (dtype='u4')
        Seconds since 1970-01-01 (or 1958-01-01)
    img_subsec : numpy array (dtype='u2')
        Sub-seconds as (1 / 2**16) seconds
    img_hk :  numpy array
        DemHK telemetry packages

    Returns
    -------
    tuple
        reference day: datetime, sec_of_day: numpy.ndarray

    """
    # determine for the first timestamp the offset with last midnight [seconds]
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    tstamp0 = epoch + timedelta(seconds=int(img_sec[0]))
    ref_day = datetime(
        year=tstamp0.year, month=tstamp0.month, day=tstamp0.day, tzinfo=UTC
    )
    # seconds since midnight
    offs_sec = (ref_day - epoch).total_seconds()

    # Determine offset wrt start-of-integration (IMRO + 1)
    # Where the default is defined as IMRO:
    #  [full-frame] COADDD + 2  (no typo, this is valid for the later MPS's)
    #  [binned] 2 * COADD + 1   (always valid)
    offs_msec = 0
    if img_hk["ICUSWVER"][0] > 0x123:
        imro = np.empty(img_hk.size, dtype=float)
        _mm = img_hk["IMRLEN"] == FULLFRAME_BYTES
        imro[_mm] = img_hk["REG_NCOADDFRAMES"][_mm] + 2
        imro[~_mm] = 2 * img_hk["REG_NCOADDFRAMES"][~_mm] + 1
        offs_msec = img_hk["FTI"] * (imro + 1) / 10

    # return seconds since midnight
    return ref_day, img_sec - offs_sec + img_subsec / 65536 - offs_msec / 1000


def hk_sec_of_day(
    ccsds_sec: np.ndarray, ccsds_subsec: np.ndarray, ref_day: datetime | None = None
) -> np.ndarray:
    """Convert CCSDS timestamp to seconds after midnight.

    Parameters
    ----------
    ccsds_sec : numpy array (dtype='u4')
        Seconds since 1970-01-01 (or 1958-01-01)
    ccsds_subsec : numpy array (dtype='u2')
        Sub-seconds as (1 / 2**16) seconds
    ref_day : datetime.datetime, optional
        `date` object represent a date (year, month, day)

    Returns
    -------
    numpy.ndarray with sec_of_day

    """
    # determine for the first timestamp the offset with last midnight [seconds]
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    if ref_day is None:
        tstamp0 = epoch + timedelta(seconds=int(ccsds_sec[0]))
        ref_day = datetime(
            year=tstamp0.year, month=tstamp0.month, day=tstamp0.day, tzinfo=UTC
        )
    offs_sec = (ref_day - epoch).total_seconds()

    # return seconds since midnight
    return ccsds_sec - offs_sec + ccsds_subsec / 65536


# - class CCSDSio -------------------------
class CCSDSio:
    """Read SPEXone telemetry packets.

    Parameters
    ----------
    file_list: iterator to strings
        list of file-names, where each file contains parts of a measurement

    Notes
    -----
    The formats of the PACE telemetry packets are following the standards:
    CCSDS-131.0-B-3, CCSDS-132.0-B-2 and CCSDS-133.0-B-1.

    This module is currenty restriced to telementry packets with APID:
    0x350 (Science), 0x320 (NomHK) and 0x322 (DemHK).

    A telemtry packet consist of a PRIMARY HEADER, SECONDARY HEADER
    (consist of a timestamp) and USER DATA with the actual telemetry
    packet data.

    Doc: TMTC handbook (SPX1-TN-005), issue 12, 15-May-2020

    The files with science and telemetry data needs to be in chronological
    order. However, you may mix science and housekeeping data as long as
    science data are chronological and housekeeping data are chronological.

    Examples
    --------
    >>> packets = ()
    >>> with CCSDSio(['file1', 'file2', 'file3']) as ccsds:
    >>>     while True:
    >>>         # read one telemetry packet at a time
    >>>         packet = ccsds.read_packet()
    >>>         if packet is None:
    >>>             # now we have read all files
    >>>             break
    >>>
    >>>         packets += (packet[0],)
    >>>
    >>>     # combine segmented Science packages
    >>>     science_tm = ccsds.science_tm(packets)
    >>>     # now you may want to collect the engineering packages

    """

    def __init__(self: CCSDSio, file_list: list) -> None:
        """Initialize access to a SPEXone Level-0 product (CCSDS format)."""
        # initialize class attributes
        self.file_list = iter(file_list)
        self.found_invalid_apid = False
        self.__hdr = None
        self.fp = None

        if file_list:
            self.open_next_file()

    def __iter__(self: CCSDSio) -> None:
        """Allow iteration."""
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __enter__(self: CCSDSio) -> Self:
        """Initiate the context manager."""
        return self

    def __exit__(self: CCSDSio, *args: object) -> bool:
        """Exit the context manager."""
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self: CCSDSio) -> None:
        """Close resources."""
        if self.fp is not None:
            if self.found_invalid_apid:
                print(MSG_INVALID_APID)
            self.found_invalid_apid = False
            self.fp.close()

    # ---------- define some class properties ----------
    @property
    def version_no(self: CCSDSio) -> int | None:
        """Returns CCSDS version number."""
        if self.__hdr is None:
            return None

        return (self.__hdr["type"] >> 13) & 0x7

    @property
    def type_indicator(self: CCSDSio) -> int | None:
        """Returns type of telemetry packet."""
        if self.__hdr is None:
            return None

        return (self.__hdr["type"] >> 12) & 0x1

    @property
    def secnd_hdr_flag(self: CCSDSio) -> bool | None:
        """Returns flag indicating presence of a secondary header."""
        if self.__hdr is None:
            return None

        return (self.__hdr["type"] >> 11) & 0x1

    @property
    def ap_id(self: CCSDSio) -> int | None:
        """Returns SPEXone ApID."""
        if self.__hdr is None:
            return None

        return self.__hdr["type"] & 0x7FF

    @property
    def grouping_flag(self: CCSDSio) -> int | None:
        """Returns grouping flag.

        The meaning of the grouping flag values are::

          00 continuation packet-data segment
          01 first packet-data segment
          10 last packet-data segment
          11 packet-data unsegmented

        """
        if self.__hdr is None:
            return None

        return (self.__hdr["sequence"] >> 14) & 0x3

    @property
    def sequence_count(self: CCSDSio) -> int | None:
        """Returns sequence counter, rollover to zero at 0x3FFF."""
        if self.__hdr is None:
            return None

        return self.__hdr["sequence"] & 0x3FFF

    @property
    def packet_length(self: CCSDSio) -> int | None:
        """Returns size of packet data in bytes.

        Value equals secondary header + user data (always odd)
        """
        if self.__hdr is None:
            return None

        return self.__hdr["length"]

    # ---------- define empty telemetry packet ----------
    def open_next_file(self: CCSDSio) -> None:
        """Open next file from file_list."""
        flname = next(self.file_list)
        if not Path(flname).is_file():
            raise FileNotFoundError(f"{flname} does not exist")

        self.close()
        # pylint: disable=consider-using-with
        self.fp = open(flname, "rb")  # noqa: SIM115

    @staticmethod
    def fix_dem_hk24(dem_hk: np.ndarray) -> np.ndarray:
        """Correct 32-bit values in the DemHk.

        Which originate from 24-bit values of the detector register parameters.

        Parameters
        ----------
        dem_hk : numpy.ndarray
           SPEXone DEM housekeeping packages

        Returns
        -------
        numpy.ndarray
           SPEXone DEM housekeeping packages

        """
        for key in [
            "DET_EXPTIME",
            "DET_EXPSTEP",
            "DET_KP1",
            "DET_KP2",
            "DET_EXPTIME2",
            "DET_EXPSTEP2",
        ]:
            dem_hk[key] = dem_hk[key] >> 8

        return dem_hk

    @staticmethod
    def fix_sci_hk24(sci_hk: np.ndarray) -> np.ndarray:
        """Correct 32-bit values in the Science HK.

        Which originate from 24-bit values in the detector register parameters.
        In addition::

         - copy the first 4 bytes of DET_CHENA to DET_ILVDS
         - parameter 'REG_BINNING_TABLE_START' was writen in little-endian

        Parameters
        ----------
        sci_hk : numpy.ndarray
           SPEXone Science telemetry packages

        Returns
        -------
        numpy.ndarray
           SPEXone Science telemetry packages

        """
        if np.all(sci_hk["ICUSWVER"] < 0x129):
            key = "REG_BINNING_TABLE_START"
            sci_hk[key] = np.ndarray(
                shape=sci_hk.shape, dtype="<u4", buffer=sci_hk[key]
            )

        sci_hk["DET_ILVDS"] = sci_hk["DET_CHENA"] & 0xF

        for key in [
            "TS1_DEM_N_T",
            "TS2_HOUSING_N_T",
            "TS3_RADIATOR_N_T",
            "TS4_DEM_R_T",
            "TS5_HOUSING_R_T",
            "TS6_RADIATOR_R_T",
            "LED1_ANODE_V",
            "LED1_CATH_V",
            "LED1_I",
            "LED2_ANODE_V",
            "LED2_CATH_V",
            "LED2_I",
            "ADC1_VCC",
            "ADC1_REF",
            "ADC1_T",
            "ADC2_VCC",
            "ADC2_REF",
            "ADC2_T",
            "DET_EXPTIME",
            "DET_EXPSTEP",
            "DET_KP1",
            "DET_KP2",
            "DET_EXPTIME2",
            "DET_EXPSTEP2",
            "DET_CHENA",
        ]:
            sci_hk[key] = sci_hk[key] >> 8

        return sci_hk

    def __rd_science(self: CCSDSio, hdr: np.ndarray) -> np.ndarray:
        """Read Science telemetry packet.

        Parameters
        ----------
        hdr :  numpy.ndarray
           CCSDS header information

        Returns
        -------
        numpy.ndarray
           SPEXone Science telemetry packages

        """
        num_bytes = self.packet_length - TIME_DTYPE.itemsize + 1
        packet = np.empty(
            1,
            dtype=np.dtype(
                [
                    ("packet_header", HDR_DTYPE),
                    ("science_hk", SCIHK_DTYPE),
                    ("icu_time", TIME_DTYPE),
                    ("image_data", "O"),
                ]
            ),
        )
        packet["packet_header"] = hdr

        # first segment or unsegmented data packet provides Science_HK
        if self.grouping_flag in (1, 3):
            packet["science_hk"] = self.fix_sci_hk24(
                np.fromfile(self.fp, count=1, dtype=SCIHK_DTYPE)
            )
            num_bytes -= SCIHK_DTYPE.itemsize
            packet["icu_time"] = np.fromfile(self.fp, count=1, dtype=TIME_DTYPE)
            num_bytes -= TIME_DTYPE.itemsize

        # read detector image data
        packet["image_data"][0] = np.fromfile(
            self.fp, dtype=">u2", count=num_bytes // 2
        )
        return packet

    def __rd_nomhk(self: CCSDSio, hdr: np.ndarray) -> np.ndarray:
        """Read NomHK telemetry packet.

        Parameters
        ----------
        hdr :  numpy.ndarray
           CCSDS header information

        Returns
        -------
        numpy.ndarray
           SPEXone Nominal housekeeping packages

        """
        packet = np.empty(
            1,
            dtype=np.dtype(
                [("packet_header", HDR_DTYPE), ("nominal_hk", tmtc_dtype(0x320))]
            ),
        )
        packet["packet_header"] = hdr
        packet["nominal_hk"] = np.fromfile(self.fp, count=1, dtype=tmtc_dtype(0x320))
        return packet

    def __rd_demhk(self: CCSDSio, hdr: np.ndarray) -> np.ndarray:
        """Read DemHK telemetry packet.

        Parameters
        ----------
        hdr :  numpy.ndarray
           CCSDS header information

        Returns
        -------
        numpy.ndarray
           SPEXone detector housekeeping packages

        """
        packet = np.empty(
            1,
            dtype=np.dtype(
                [("packet_header", HDR_DTYPE), ("detector_hk", tmtc_dtype(0x322))]
            ),
        )
        packet["packet_header"] = hdr
        packet["detector_hk"] = self.fix_dem_hk24(
            np.fromfile(self.fp, count=1, dtype=tmtc_dtype(0x322))
        )
        return packet

    def __rd_tc_accept(self: CCSDSio, _: np.ndarray) -> np.ndarray:
        """Read/dump TcAccept packet."""
        self.fp.seek(-1 * HDR_DTYPE.itemsize, 1)
        packet = np.fromfile(
            self.fp,
            count=1,
            dtype=np.dtype(
                [
                    ("packet_header", HDR_DTYPE),
                    ("TcPacketId", ">u2"),
                    ("TcSeqControl", ">u2"),
                ]
            ),
        )
        # print(
        # "[TcAccept]: ", self, packet["TcPacketId"][0], packet["TcSeqControl"][0]
        # )
        return packet

    def __rd_tc_execute(self: CCSDSio, _: np.ndarray) -> np.ndarray:
        """Read/dump TcExecute packet."""
        self.fp.seek(-1 * HDR_DTYPE.itemsize, 1)
        packet = np.fromfile(
            self.fp,
            count=1,
            dtype=np.dtype(
                [
                    ("packet_header", HDR_DTYPE),
                    ("TcPacketId", ">u2"),
                    ("TcSeqControl", ">u2"),
                ]
            ),
        )
        # print(
        # "[TcExecute]:", self, packet["TcPacketId"][0], packet["TcSeqControl"][0]
        # )
        return packet

    def __rd_tc_fail(self: CCSDSio, _: np.ndarray) -> np.ndarray:
        """Read/dump TcFail packet."""
        self.fp.seek(-1 * HDR_DTYPE.itemsize, 1)
        packet = np.fromfile(
            self.fp,
            count=1,
            dtype=np.dtype(
                [
                    ("packet_header", HDR_DTYPE),
                    ("TcPacketId", ">u2"),
                    ("TcSeqControl", ">u2"),
                    ("TcFailCode", ">u2"),
                    ("FailParameter1", ">u2"),
                    ("FailParameter2", ">u2"),
                ]
            ),
        )
        # print(
        #    "[TcFail]:   ",
        #    self,
        #    packet["TcPacketId"][0],
        #    packet["TcSeqControl"][0],
        #    bin(packet["TcFailCode"][0]),
        #    packet["FailParameter1"][0],
        #    packet["FailParameter2"][0],
        # )
        return packet

    def __rd_tc_reject(self: CCSDSio, _: np.ndarray) -> np.ndarray:
        """Read/dump TcReject packet."""
        self.fp.seek(-1 * HDR_DTYPE.itemsize, 1)
        packet = np.fromfile(
            self.fp,
            count=1,
            dtype=np.dtype(
                [
                    ("packet_header", HDR_DTYPE),
                    ("TcPacketId", ">u2"),
                    ("TcSeqControl", ">u2"),
                    ("TcRejectCode", ">u2"),
                    ("RejectParameter1", ">u2"),
                    ("RejectParameter2", ">u2"),
                ]
            ),
        )
        # print(
        #    "[TcReject]: ",
        #    self,
        #    packet["TcPacketId"][0],
        #    packet["TcSeqControl"][0],
        #    bin(packet["TcRejectCode"][0]),
        #    packet["RejectParameter1"][0],
        #    packet["RejectParameter2"][0],
        # )
        return packet

    def __rd_other(self: CCSDSio, hdr: np.ndarray) -> np.ndarray | None:
        """Read other telemetry packet."""
        num_bytes = self.packet_length - TIME_DTYPE.itemsize + 1
        if not 0x320 <= self.ap_id <= 0x350:
            self.found_invalid_apid = True
            self.fp.seek(num_bytes, 1)
            return None

        packet = np.empty(
            1,
            dtype=np.dtype(
                [("packet_header", HDR_DTYPE), ("raw_data", "u1", num_bytes)]
            ),
        )
        packet["packet_header"] = hdr
        packet["raw_data"] = np.fromfile(self.fp, count=num_bytes, dtype="u1")
        return packet

    def read_packet(self: CCSDSio) -> np.ndarray | None:
        """Read next telemetry packet.

        Returns
        -------
        numpy.ndarray
           CCSDS packet data

        """
        if self.fp is None:
            return None

        # read primary/secondary header
        hdr = np.fromfile(self.fp, count=1, dtype=HDR_DTYPE)
        if hdr.size == 0:
            try:
                self.open_next_file()
            except StopIteration:
                return None
            except FileNotFoundError:
                return None

            hdr = np.fromfile(self.fp, count=1, dtype=HDR_DTYPE)
            if hdr.size == 0:
                return None

        # save packet header as class attribute
        self.__hdr = hdr[0]

        # read telemetry packet
        rd_func = {
            0x350: self.__rd_science,
            0x320: self.__rd_nomhk,
            0x322: self.__rd_demhk,
            0x331: self.__rd_tc_accept,
            0x332: self.__rd_tc_reject,
            0x333: self.__rd_tc_execute,
            0x334: self.__rd_tc_fail,
        }.get(self.ap_id, self.__rd_other)
        return rd_func(hdr)

    @staticmethod
    def select_tm(packets_in: tuple, ap_id: int) -> tuple:
        """Select telemetry packages on SPEXone ApID.

        Parameters
        ----------
        packets_in: tuple
           SPEXone telemetry packages
        ap_id: int
           SPEXone ApID

        Returns
        -------
        tuple
           selected telemetry packages

        """
        packets = ()
        for packet in packets_in:
            if "packet_header" not in packet.dtype.names:
                continue

            if (packet["packet_header"]["type"] & 0x7FF) == ap_id:
                packets += (packet,)

        return packets

    def science_tm(self: CCSDSio, packets_in: tuple) -> tuple:
        """Combine segmented Science telemetry packages.

        Parameters
        ----------
        packets_in: tuple
           Science or house-keeping telemetry packages

        Returns
        -------
        tuple
           unsegmented Science telemetry packages

        """
        # reject non-Science telemetry packages
        packets = self.select_tm(packets_in, 0x350)
        if not packets:
            return ()

        # check if grouping_flag of first segment equals 1
        #   else reject all segments with grouping_flag != 1
        self.__hdr = packets[0]["packet_header"]
        if self.grouping_flag != 1:
            ii = 0
            for packet in packets:
                self.__hdr = packet["packet_header"]
                if self.grouping_flag == 1:
                    break
                ii += 1

            print(f"[WARNING]: first frame incomplete - skipped {ii} segments")
            packets = packets[ii:]
            if not packets:
                return ()

        # check if grouping_flag of last segment equals 2
        #   else reject all segments after the last segment
        #   with grouping_flag == 2
        self.__hdr = packets[-1]["packet_header"]
        if self.grouping_flag != 2:
            ii = 0
            for packet in packets:
                self.__hdr = packet["packet_header"]
                if self.grouping_flag == 2:
                    break
                ii += 1

            print(f"[WARNING]: last frame incomplete - rejected {ii} segments")
            packets = packets[:-ii]
            if not packets:
                return ()

        res = ()
        offs = 0
        prev_grp_flag = 2
        for packet in packets:
            grouping_flag = (packet["packet_header"]["sequence"] >> 14) & 0x3
            # print(prev_grp_flag, grouping_flag, len(res), offs,
            #      packet['image_data'].size)
            # handle segmented data
            if grouping_flag == 1:  # first segment
                # group_flag of previous package should be 2
                if prev_grp_flag != 2:
                    if packet["image_data"].size in (3853, 7853):
                        print(MSG_SKIP_FRAME)
                        prev_grp_flag = 2
                        offs = 0
                    else:
                        raise RuntimeError(MSG_CORRUPT_APID)

                img_size = packet["science_hk"]["IMRLEN"] // 2
                rec_buff = np.empty(
                    1,
                    dtype=np.dtype(
                        [
                            ("packet_header", HDR_DTYPE),
                            ("science_hk", SCIHK_DTYPE),
                            ("icu_time", TIME_DTYPE),
                            ("image_data", "O"),
                        ]
                    ),
                )[0]
                img_buff = np.empty(img_size, dtype="u2")

                rec_buff["packet_header"] = packet["packet_header"]
                rec_buff["science_hk"] = packet["science_hk"]
                rec_buff["icu_time"] = packet["icu_time"]
                img_buff[offs : offs + packet["image_data"].size] = packet["image_data"]
                offs += packet["image_data"].size
            else:  # continuation or last segment
                # group_flag of previous package should be 0 or 1
                if prev_grp_flag == 2:
                    raise RuntimeError(MSG_CORRUPT_FRAME)

                img_buff[offs : offs + packet["image_data"].size] = packet["image_data"]
                offs += packet["image_data"].size
                if grouping_flag == 2:
                    if offs == img_size:
                        rec_buff["image_data"] = img_buff
                        res += (rec_buff,)
                    else:
                        print(MSG_SKIP_FRAME)
                    offs = 0

            # keep current group flag for next read
            prev_grp_flag = grouping_flag

        return res
