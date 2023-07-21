#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Contains a collection of routines to read and write SPEXone CCSDS data:

   `dtype_packet_hdr`, `dtype_tmtc`, `read_cfe_header`, `read_lv0_data`

And handy routines to convert CCSDS parameters:

   `ap_id`, `fix_sub_sec`, `grouping_flag`,
   `hk_sec_of_day`, `img_sec_of_day`, `packet_length`, `sequence`
"""
from __future__ import annotations
__all__ = ['ap_id', 'fix_sub_sec', 'grouping_flag',
           'hk_sec_of_day', 'img_sec_of_day',
           'packet_length', 'sequence',
           'dtype_packet_hdr', 'dtype_tmtc',
           'read_cfe_header', 'read_lv0_data']

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .lib.leap_sec import get_leap_seconds
from .lib.tmtc_def import tmtc_dtype
from .tm_science import TMscience

# - global parameters ------------------------------
FULLFRAME_BYTES = 2 * 2048 * 2048


# - local functions --------------------------------
def ap_id(hdr: np.ndarray) -> int:
    """
    Returns Telemetry APID, the range 0x320 to 0x351 is available to SPEXone

    Parameters
    ----------
    hdr :  np.ndarray
        Structured numpy array with contents of the level 0 CCSDS header

    Returns
    -------
    int
        Telemetry APID

    Notes
    -----
    The following values are recognized:

    - 0x350 : Science
    - 0x320 : NomHk
    - 0x322 : DemHk
    - 0x331 : TcAccept
    - 0x332 : TcReject
    - 0x333 : TcExecute
    - 0x334 : TcFail
    - 0x335 : EventRp

    """
    return hdr['type'] & 0x7FF


def grouping_flag(hdr: np.ndarray) -> int:
    """
    Returns grouping flag

    Parameters
    ----------
    hdr :  np.ndarray
        Structured numpy array with contents of the level 0 CCSDS header

    Returns
    -------
    int
        Grouping flag

    Notes
    -----
    The 2-byte flag is encoded as follows::

       00: continuation packet-data segment
       01: first packet-data segment
       10: last packet-data segment
       11: packet-data unsegmented

    """
    return (hdr['sequence'] >> 14) & 0x3


def sequence(hdr: np.ndarray) -> int:
    """
    Returns sequence counter, rollover to zero at 0x3FFF

    Parameters
    ----------
    hdr :  np.ndarray
        Structured numpy array with contents of the level 0 CCSDS header

    Returns
    -------
    int
        Sequence counter
    """
    return hdr['sequence'] & 0x3FFF


def packet_length(hdr: np.ndarray) -> int:
    """
    Returns size of secondary header + user data - 1 in bytes

    Parameters
    ----------
    hdr :  np.ndarray
        Structured numpy array with contents of the level 0 CCSDS header

    Returns
    -------
    int
        Size of variable part of the CCSDS package

    Notes
    -----
    We always read the primary header and secondary header at once.
    Value range: 7 - 16375
    """
    return hdr['length']


def dtype_packet_hdr(file_format: str) -> np.dtype | None:
    """
    Return definition of the CCSDS packet headers (primary and secondary)

    Parameters
    ----------
    file_format : {'raw', 'dsb' or 'st3'}
        File format of level 0 products

    Returns
    -------
    np.dtype
        numpy dtype of the packet headers or None if file format is unknown

    Notes
    -----

    'raw': data has no file header and standard CCSDS packet headers

    'st3': data has no file header and ITOS + spacewire + CCSDS packet headers

    'dsb': data has a cFE file-header and spacewire + CCSDS packet headers

    """
    if file_format == 'raw':
        return np.dtype([('type', '>u2'),
                         ('sequence', '>u2'),
                         ('length', '>u2'),
                         ('tai_sec', '>u4'),
                         ('sub_sec', '>u2')])

    if file_format == 'dsb':
        return np.dtype([('spacewire', 'u1', (2,)),
                         ('type', '>u2'),
                         ('sequence', '>u2'),
                         ('length', '>u2'),
                         ('tai_sec', '>u4'),
                         ('sub_sec', '>u2')])

    if file_format == 'st3':
        return np.dtype([('itos_hdr', '>u2', (8,)),
                         ('spacewire', 'u1', (2,)),
                         ('type', '>u2'),
                         ('sequence', '>u2'),
                         ('length', '>u2'),
                         ('tai_sec', '>u4'),
                         ('sub_sec', '>u2')])

    return None


def dtype_tmtc(hdr: np.ndarray) -> np.dtype:
    """
    Return definition of a CCSDS TmTc package (given APID)

    Parameters
    ----------
    hdr :  np.ndarray
        Structured numpy array with contents of the level 0 CCSDS header

    Returns
    -------
    np.dtype
        numpy dtype of CCSDS TmTC package or None if APID is not implemented
    """
    return {0x320: np.dtype([('hdr', hdr.dtype),
                             ('hk', tmtc_dtype(0x320))]),
            0x322: np.dtype([('hdr', hdr.dtype),
                             ('hk', tmtc_dtype(0x322))]),
            0x331: np.dtype([('hdr', hdr.dtype),
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2')]),
            0x332: np.dtype([('hdr', hdr.dtype),
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2'),
                             ('TcErrorCode', '>u2'),
                             ('RejectParameter1', '>u2'),
                             ('RejectParameter2', '>u2')]),
            0x333: np.dtype([('hdr', hdr.dtype),
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2')]),
            0x334: np.dtype([('hdr', hdr.dtype),
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2'),
                             ('TcErrorCode', '>u2'),
                             ('FailParameter1', '>u2'),
                             ('FailParameter2', '>u2')])}.get(ap_id(hdr), None)


def _fix_hk24_(sci_hk):
    """
    Correct 32-bit integers in the Science HK which originate from
    24-bit integers in the detector register values

    In addition:

    - copy the first 4 bytes of 'DET_CHENA' to 'DET_ILVDS'
    - parameter 'REG_BINNING_TABLE_START' was writen in little-endian

    """
    res = sci_hk.copy()
    if sci_hk['ICUSWVER'] < 0x129:
        res['REG_BINNING_TABLE_START'] = \
            sci_hk['REG_BINNING_TABLE_START'].byteswap()

    res['DET_ILVDS'] = sci_hk['DET_CHENA'] & 0xf
    for key in ['TS1_DEM_N_T', 'TS2_HOUSING_N_T', 'TS3_RADIATOR_N_T',
                'TS4_DEM_R_T', 'TS5_HOUSING_R_T', 'TS6_RADIATOR_R_T',
                'LED1_ANODE_V', 'LED1_CATH_V', 'LED1_I',
                'LED2_ANODE_V', 'LED2_CATH_V', 'LED2_I',
                'ADC1_VCC', 'ADC1_REF', 'ADC1_T',
                'ADC2_VCC', 'ADC2_REF', 'ADC2_T',
                'DET_EXPTIME', 'DET_EXPSTEP', 'DET_KP1',
                'DET_KP2', 'DET_EXPTIME2', 'DET_EXPSTEP2',
                'DET_CHENA']:
        res[key] = sci_hk[key] >> 8

    return res


def read_cfe_header(flname: Path, verbose: bool = False) -> np.ndarray:
    """Read cFE file header (only for file_format='dsb').
    """
    # define numpy data-type to read the cFE file-header
    dtype_cfe = np.dtype([
        ('ContentType', 'S4'),
        ('SubType', 'S4'),
        ('FileHeaderLength', '>u4'),
        ('SpacecraftID', 'S4'),
        ('ProcessorID', '>u4'),
        ('InstrumentID', 'S4'),
        ('TimeSec', '>u4'),
        ('TimeSubSec', '>u4'),
        ('Filename', 'S32')])

    cfe_hdr = np.fromfile(flname, count=1, dtype=dtype_cfe)[0]
    if verbose:
        print(f'[INFO]: content of cFE header "{cfe_hdr}"')
    return cfe_hdr


def read_lv0_data(file_list: list[Path, ...],
                  file_format: str,
                  debug: bool = False,
                  verbose: bool = False) -> tuple[tuple, tuple]:
    """Read level 0 data and return Science and telemetry data.

    Parameters
    ----------
    file_list : list of Path
       list of CCSDS files
    file_format : {'raw', 'st3', 'dsb'}
       type of CCSDS data
    debug : bool, default=False
       run in debug mode
    verbose : bool, default=False
       be verbose

    Returns
    -------
    tuple
         Contains all Science and TmTc CCSDS packages as numpy arrays,
         or None if called with debug is True
    """
    hdr_dtype = dtype_packet_hdr(file_format)
    scihk_dtype = tmtc_dtype(0x350)
    icutm_dtype = np.dtype([('tai_sec', '>u4'),
                            ('sub_sec', '>u2')])

    # read level 0 headers and CCSDS data of Science and TmTc data
    ccsds_sci = ()
    ccsds_hk = ()
    for flname in file_list:
        offs = 0
        if file_format == 'dsb':
            cfe_hdr = read_cfe_header(flname, verbose)
            offs += cfe_hdr['FileHeaderLength']

        buff_sci = ()          # Use chunking to speed-up memory allocation
        buff_hk = ()
        with open(flname, 'rb') as fp:
            if verbose:
                print(f'[INFO]: processing file "{flname}"')

            # read CCSDS header and user data
            ccsds_data = fp.read()
            while offs < len(ccsds_data):
                try:
                    hdr = np.frombuffer(ccsds_data, count=1, offset=offs,
                                        dtype=hdr_dtype)[0]
                except ValueError as exc:
                    print(f'[WARNING]: header reading error with "{exc}"')
                    break

                # copy the full CCSDS package
                if debug:
                    print(ap_id(hdr), grouping_flag(hdr),
                          hdr_dtype.itemsize, hdr['length'], offs)
                    offs += hdr_dtype.itemsize + hdr['length'] - 5
                elif ap_id(hdr) == 0x350:                   # Science APID
                    nbytes = hdr['length'] - 5
                    if grouping_flag(hdr) == 1:
                        buff = np.empty(1, dtype=np.dtype([
                            ('hdr', hdr_dtype),
                            ('hk', scihk_dtype),
                            ('icu_tm', icutm_dtype),
                            ('frame', 'O')]))
                        buff['hdr'] = hdr
                        offs += hdr_dtype.itemsize
                        buff['hk'] = _fix_hk24_(
                            np.frombuffer(ccsds_data,
                                          count=1, offset=offs,
                                          dtype=scihk_dtype)[0])
                        offs += scihk_dtype.itemsize
                        buff['icu_tm'] = np.frombuffer(ccsds_data,
                                                       count=1, offset=offs,
                                                       dtype=icutm_dtype)[0]
                        offs += icutm_dtype.itemsize
                        nbytes -= (scihk_dtype.itemsize + icutm_dtype.itemsize)
                    else:
                        buff = np.empty(1, dtype=np.dtype([
                            ('hdr', hdr_dtype),
                            ('frame', 'O')]))
                        buff['hdr'] = hdr
                        offs += hdr_dtype.itemsize

                    buff['frame'][0] = np.frombuffer(ccsds_data,
                                                     count=nbytes // 2,
                                                     offset=offs, dtype='>u2')
                    buff_sci += (buff.copy(),)
                    offs += nbytes
                elif 0x320 <= ap_id(hdr) < 0x335:           # other valid APIDs
                    buff = np.frombuffer(ccsds_data, count=1, offset=offs,
                                         dtype=dtype_tmtc(hdr))
                    buff_hk += (buff,)
                    offs += dtype_tmtc(hdr).itemsize
                else:
                    offs += hdr_dtype.itemsize + hdr['length'] - 5
        ccsds_sci += buff_sci
        ccsds_hk += buff_hk
        del ccsds_data

    if verbose:
        print(f'[INFO]: number of Science packages {len(ccsds_sci)}')
        print(f'[INFO]: number of Engineering packages {len(ccsds_hk)}')

    return ccsds_sci, ccsds_hk


def fix_sub_sec(tai_sec, sub_sec) -> tuple:
    """
    In ICU S/W version 0x123 a bug was introduced which corrupted the
    read of the sub-sec parameter. Therefore, all ambient measurements
    performed between 2020-12-02T09:34 and 2021-01-04T16:23 have to be
    adjusted.
    """
    us100 = np.round(10000 * sub_sec.astype(float) / 65536)
    buff = us100 + tai_sec - 10000
    us100 = buff.astype('u8') % 10000
    sub_sec = ((us100 << 16) // 10000).astype('u2')

    return tai_sec, sub_sec


def which_epoch(timestamp: int) -> datetime:
    """
    Determine year of epoch, 1970 (UTC) or 1958 (TAI).
    """
    if timestamp < 1956528000:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    return (datetime(1958, 1, 1, tzinfo=timezone.utc)
            - timedelta(seconds=get_leap_seconds(timestamp)))


def img_sec_of_day(img_sec, img_subsec, img_hk) -> tuple[datetime, float | Any]:
    """
    Convert Image CCSDS timestamp to seconds after midnight

    Parameters
    ----------
    img_sec : numpy array (dtype='u4')
        Seconds since 1970-01-01 (or 1958-01-01)
    img_subsec : numpy array (dtype='u2')
        Sub-seconds as (1 / 2**16) seconds
    img_hk :  numpy array

    Returns
    -------
    tuple
        reference day: float, sec_of_day: numpy.ndarray
    """
    # determine for the first timestamp the offset with last midnight [seconds]
    epoch = which_epoch(img_sec[0])
    tstamp0 = epoch + timedelta(seconds=int(img_sec[0]))
    ref_day = datetime(year=tstamp0.year,
                       month=tstamp0.month,
                       day=tstamp0.day, tzinfo=timezone.utc)
    # seconds since midnight
    offs_sec = (ref_day - epoch).total_seconds()

    # Determine offset wrt start-of-integration (IMRO + 1)
    # Where the default is defined as IMRO:
    #  [full-frame] COADDD + 2  (no typo, this is valid for the later MPS's)
    #  [binned] 2 * COADD + 1   (always valid)
    offs_msec = 0
    if img_hk['ICUSWVER'][0] > 0x123:
        mps = TMscience(img_hk)
        if np.bincount(mps.binning_table).argmax() == 0:
            imro = mps.get('REG_NCOADDFRAMES') + 2
        else:
            imro = 2 * mps.get('REG_NCOADDFRAMES') + 1
        offs_msec = mps.get('FTI') * (imro + 1) / 10

    # return seconds since midnight
    return ref_day, img_sec - offs_sec + img_subsec / 65536 - offs_msec / 1000


def hk_sec_of_day(ccsds_sec, ccsds_subsec, ref_day=None) -> np.ndarray:
    """
    Convert CCSDS timestamp to seconds after midnight

    Parameters
    ----------
    ccsds_sec : numpy array (dtype='u4')
        Seconds since 1970-01-01 (or 1958-01-01)
    ccsds_subsec : numpy array (dtype='u2')
        Sub-seconds as (1 / 2**16) seconds
    ref_day : datetime.datetime

    Returns
    -------
    numpy.ndarray with sec_of_day
    """
    # determine for the first timestamp the offset with last midnight [seconds]
    epoch = which_epoch(ccsds_sec[0])
    if ref_day is None:
        tstamp0 = epoch + timedelta(seconds=int(ccsds_sec[0]))
        ref_day = datetime(year=tstamp0.year,
                           month=tstamp0.month,
                           day=tstamp0.day, tzinfo=timezone.utc)
    offs_sec = (ref_day - epoch).total_seconds()

    # return seconds since midnight
    return ccsds_sec - offs_sec + ccsds_subsec / 65536
