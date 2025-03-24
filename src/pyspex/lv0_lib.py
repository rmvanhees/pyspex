#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Contains a collection of routines to access SPEXone Level-0 data.

Routines to read and write SPEXone CCSDS data:

   `dtype_tmtc`, `read_lv0_data`, `dump_hkt`, `dump_science`

Handy routines to convert CCSDS parameters:

   `hk_sec_of_day`, `img_sec_of_day`
"""

from __future__ import annotations

__all__ = ["CorruptPacketWarning", "dump_hkt", "dump_science", "read_lv0_data"]

import logging
import warnings
from pathlib import Path

import numpy as np

from .lib.ccsds_hdr import CCSDShdr
from .lib.tmtc_def import tmtc_dtype

# - local functions --------------------------------
module_logger = logging.getLogger("pyspex.lv0_lib")


def _cfe_header_(flname: Path) -> np.ndarray:
    """Read cFE file header (only for file_format='dsb')."""
    # define numpy data-type to read the cFE file-header
    dtype_cfe = np.dtype(
        [
            ("ContentType", "S4"),
            ("SubType", "S4"),
            ("FileHeaderLength", ">u4"),
            ("SpacecraftID", "S4"),
            ("ProcessorID", ">u4"),
            ("InstrumentID", "S4"),
            ("TimeSec", ">u4"),
            ("TimeSubSec", ">u4"),
            ("Filename", "S32"),
        ]
    )

    cfe_hdr = np.fromfile(flname, count=1, dtype=dtype_cfe)[0]
    module_logger.debug('content of cFE header "%s"', cfe_hdr)
    return cfe_hdr


def _fix_hk24_(sci_hk: np.ndarray) -> np.ndarray:
    """Correct 32-bit values in the Science HK.

    Which originate from 24-bit values in the detector register parameters.

    In addition::

    - copy the first 4 bytes of 'DET_CHENA' to 'DET_ILVDS'
    - parameter 'REG_BINNING_TABLE_START' was writen in little-endian

    """
    res = sci_hk.copy()
    if sci_hk["ICUSWVER"] < 0x129:
        res["REG_BINNING_TABLE_START"] = sci_hk["REG_BINNING_TABLE_START"].byteswap()

    res["DET_ILVDS"] = sci_hk["DET_CHENA"] & 0xF
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
        res[key] = sci_hk[key] >> 8

    return res


# - main function ----------------------------------
class CorruptPacketWarning(UserWarning):
    """Creating a custom warning."""


def read_lv0_data(
    file_list: list[Path], file_format: str, *, debug: bool = False
) -> tuple[tuple, tuple]:
    """Read level 0 data and return Science and telemetry data.

    Parameters
    ----------
    file_list : list of Path
       list of CCSDS files
    file_format : {'raw', 'st3', 'dsb'}
       type of CCSDS data
    debug : bool, default=False
       run in debug mode

    Returns
    -------
    tuple
         Contains all Science and TmTc CCSDS packages as numpy arrays,
         or None if called with debug is True

    """
    scihk_dtype = tmtc_dtype(0x350)
    icutm_dtype = np.dtype([("tai_sec", ">u4"), ("sub_sec", ">u2")])

    # read level 0 headers and CCSDS data of Science and TmTc data
    ccsds_sci = ()
    ccsds_hk = ()
    for flname in file_list:
        offs = 0
        if file_format == "dsb":
            cfe_hdr = _cfe_header_(flname)
            offs += cfe_hdr["FileHeaderLength"]

        buff_sci = ()  # Use chunking to speed-up memory allocation
        buff_hk = ()
        with open(flname, "rb") as fp:
            module_logger.debug('processing file "%s"', flname)

            # read CCSDS header and user data
            ccsds_data = fp.read()
            while offs < len(ccsds_data):
                try:
                    ccsds_hdr = CCSDShdr()
                    ccsds_hdr.read(file_format, ccsds_data, offs)
                    hdr_dtype = ccsds_hdr.dtype
                except ValueError as exc:
                    module_logger.warning('header read error with "%s".', exc)
                    break

                # check for data corruption (length > 0 and odd)
                if ccsds_hdr.apid != 0x340 and ccsds_hdr.packet_size % 2 == 0:
                    msg = (
                        "corrupted CCSDS packet detected:"
                        f" APID: {ccsds_hdr.apid}"
                        f", grouping_flag: {ccsds_hdr.grouping_flag}"
                        f", itemsize: {hdr_dtype.itemsize}"
                        f", packet_length: {ccsds_hdr.packet_size}"
                        f", file position: {offs}"
                    )
                    warnings.warn(msg, category=CorruptPacketWarning, stacklevel=1)
                    break

                if debug:
                    print(
                        ccsds_hdr.apid,
                        ccsds_hdr.grouping_flag,
                        hdr_dtype.itemsize,
                        ccsds_hdr.packet_size,
                        offs,
                    )
                    if not 0x320 <= ccsds_hdr.apid <= 0x350:
                        break

                    offs += hdr_dtype.itemsize + ccsds_hdr.packet_size - 5
                    continue

                # copy the full CCSDS package
                if ccsds_hdr.apid == 0x350:  # Science APID
                    nbytes = ccsds_hdr.packet_size - 5
                    if ccsds_hdr.grouping_flag == 1:
                        buff = np.empty(
                            1,
                            dtype=np.dtype(
                                [
                                    ("hdr", hdr_dtype),
                                    ("hk", scihk_dtype),
                                    ("icu_tm", icutm_dtype),
                                    ("frame", "O"),
                                ]
                            ),
                        )
                        buff["hdr"] = ccsds_hdr.hdr
                        offs += hdr_dtype.itemsize
                        buff["hk"] = _fix_hk24_(
                            np.frombuffer(
                                ccsds_data, count=1, offset=offs, dtype=scihk_dtype
                            )[0]
                        )
                        offs += scihk_dtype.itemsize
                        buff["icu_tm"] = np.frombuffer(
                            ccsds_data, count=1, offset=offs, dtype=icutm_dtype
                        )[0]
                        offs += icutm_dtype.itemsize
                        nbytes -= scihk_dtype.itemsize + icutm_dtype.itemsize
                    else:
                        buff = np.empty(
                            1, dtype=np.dtype([("hdr", hdr_dtype), ("frame", "O")])
                        )
                        buff["hdr"] = ccsds_hdr.hdr
                        offs += hdr_dtype.itemsize

                    buff["frame"][0] = np.frombuffer(
                        ccsds_data, count=nbytes // 2, offset=offs, dtype=">u2"
                    )
                    buff_sci += (buff.copy(),)
                    offs += nbytes
                elif 0x320 <= ccsds_hdr.apid < 0x350:  # other valid APIDs
                    dtype_tmtc = ccsds_hdr.data_dtype
                    buff = np.frombuffer(
                        ccsds_data, count=1, offset=offs, dtype=dtype_tmtc
                    )
                    buff_hk += (buff,)
                    offs += dtype_tmtc.itemsize
                else:
                    offs += hdr_dtype.itemsize + ccsds_hdr.packet_size - 5
            del ccsds_data

        ccsds_sci += buff_sci
        ccsds_hk += buff_hk

    module_logger.debug("number of Science packages %d", len(ccsds_sci))
    module_logger.debug("number of Engineering packages %d", len(ccsds_hk))

    return ccsds_sci, ccsds_hk


def dump_hkt(flname: str, ccsds_hk: tuple[np.ndarray, ...]) -> None:
    """Dump header info of the SPEXone housekeeping telemetry packets."""

    def msg_320(val: np.ndarray) -> str:
        return f" {val['ICUSWVER']:8x} {val['MPS_ID']:6d}"

    def msg_331(val: np.ndarray) -> str:
        return f" {-1:8x} {-1:6d} {val['TcSeqControl'][0]:12d}"

    def msg_332(val: np.ndarray) -> str:
        return (
            f" {-1:8x} {-1:6d} {val['TcSeqControl'][0]:12d}"
            f" {bin(val['TcRejectCode'][0])}"
            f" 0x{val['RejectParameter1'][0]:x}"
            f" 0x{val['RejectParameter2'][0]:x}"
        )

    def msg_333(val: np.ndarray) -> str:
        return f" {-1:8x} {-1:6d} {val['TcSeqControl'][0]:12d}"

    def msg_334(val: np.ndarray) -> str:
        return (
            f" {-1:8x} {-1:6d} {val['TcSeqControl'][0]:12d}"
            f" {bin(val['TcFailCode'][0])}"
            f" 0x{val['FailParameter1'][0]:x}"
            f" 0x{val['FailParameter2'][0]:x}"
        )

    def msg_335(val: np.ndarray) -> str:
        return (
            f" {-1:8x} {-1:6d} {bin(val['Event_ID'][0])}"
            f" {bin(val['Event_Sev'][0])}"
            f" 0x{val['Word1'][0]:x}"
            f" 0x{val['Word2'][0]:x}"
            f" 0x{val['Word3'][0]:x}"
            f" 0x{val['Word4'][0]:x}"
        )

    with Path(flname).open("w", encoding="ascii") as fp:
        fp.write(
            "APID Grouping Counter Length     TAI_SEC    SUB_SEC"
            " ICUSWVER MPS_ID TcSeqControl TcErrorCode\n"
        )
        for buf in ccsds_hk:
            ccsds_hdr = CCSDShdr(buf["hdr"][0])
            msg = (
                f"{ccsds_hdr.apid:4x} {ccsds_hdr.grouping_flag:8d}"
                f" {ccsds_hdr.sequence:7d} {ccsds_hdr.packet_size:6d}"
                f" {ccsds_hdr.tai_sec:11d} {ccsds_hdr.sub_sec:10d}"
            )

            if ccsds_hdr.apid == 0x320:
                msg_320(buf["hk"][0])
            else:
                method = {
                    0x331: msg_331,
                    0x332: msg_332,
                    0x333: msg_333,
                    0x334: msg_334,
                    0x335: msg_335,
                }.get(ccsds_hdr.apid, None)
                msg += "" if method is None else method(buf)
            fp.write(msg + "\n")


def dump_science(flname: str, ccsds_sci: tuple[np.ndarray, ...]) -> None:
    """Dump telemetry header info (Science)."""
    with Path(flname).open("w", encoding="ascii") as fp:
        fp.write(
            "APID Grouping Counter Length"
            " ICUSWVER MPS_ID  IMRLEN     ICU_SEC ICU_SUBSEC\n"
        )
        for segment in ccsds_sci:
            ccsds_hdr = CCSDShdr(segment["hdr"][0])
            if ccsds_hdr.grouping_flag == 1:
                nom_hk = segment["hk"]
                icu_tm = segment["icu_tm"]
                fp.write(
                    f"{ccsds_hdr.apid:4x}"
                    f" {ccsds_hdr.grouping_flag:8d}"
                    f" {ccsds_hdr.sequence:7d}"
                    f" {ccsds_hdr.packet_size:6d}"
                    f" {nom_hk['ICUSWVER'][0]:8x}"
                    f" {nom_hk['MPS_ID'][0]:6d}"
                    f" {nom_hk['IMRLEN'][0]:7d}"
                    f" {icu_tm['tai_sec'][0]:11d}"
                    f" {icu_tm['sub_sec'][0]:10d}\n"
                )
            else:
                fp.write(
                    f"{ccsds_hdr.apid:4x}"
                    f" {ccsds_hdr.grouping_flag:8d}"
                    f" {ccsds_hdr.sequence:7d}"
                    f" {ccsds_hdr.packet_size:6d}\n"
                )
