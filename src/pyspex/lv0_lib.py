#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Contains a collection of routines to access SPEXone Level-0 data.

Routines to read and write SPEXone CCSDS data:

   `dtype_tmtc`, `read_lv0_data`, `dump_hkt`, `dump_science`

Handy routines to convert CCSDS parameters:

   `ap_id`, `grouping_flag`, `packet_length`, `sequence`,
   `hk_sec_of_day`, `img_sec_of_day`
"""
from __future__ import annotations

__all__ = ['ap_id', 'dtype_tmtc', 'dump_hkt', 'dump_science',
           'grouping_flag', 'packet_length', 'read_lv0_data', 'sequence']

import logging
from pathlib import Path

import numpy as np

from .lib.tmtc_def import tmtc_dtype

# - local functions --------------------------------
module_logger = logging.getLogger('pyspex.lv0_lib')


def _cfe_header_(flname: Path) -> np.ndarray:
    """Read cFE file header (only for file_format='dsb')."""
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
    module_logger.debug('content of cFE header "%s"', cfe_hdr)
    return cfe_hdr


def _dtype_packet_(file_format: str) -> np.dtype | None:
    """Return definition of the CCSDS packet headers (primary and secondary).

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


def _fix_hk24_(sci_hk: np.ndarray):
    """Correct 32-bit values in the Science HK.

    Which originate from 24-bit values in the detector register parameters.

    In addition::

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


# - helper functions to read Level-0 data ----------
def ap_id(hdr: np.ndarray) -> int:
    """Return Telemetry APID, the range 0x320 to 0x351 is available to SPEXone.

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
    """Return grouping flag.

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
    """Return sequence counter, rollover to zero at 0x3FFF.

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
    """Return size of secondary header + user data - 1 in bytes.

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


def dtype_tmtc(hdr: np.ndarray) -> np.dtype:
    """Return definition of a CCSDS TmTc package (given APID).

    Parameters
    ----------
    hdr :  np.ndarray
        Structured numpy array with contents of the level 0 CCSDS header

    Returns
    -------
    np.dtype
        numpy dtype of CCSDS TmTC package or None if APID is not implemented
    """
    return {0x320: np.dtype([('hdr', hdr.dtype),           # NomHk
                             ('hk', tmtc_dtype(0x320))]),
            0x322: np.dtype([('hdr', hdr.dtype),           # DemHk
                             ('hk', tmtc_dtype(0x322))]),
            0x331: np.dtype([('hdr', hdr.dtype),           # TcAccept
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2')]),
            0x332: np.dtype([('hdr', hdr.dtype),           # TcReject
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2'),
                             ('TcRejectCode', '>u2'),
                             ('RejectParameter1', '>u2'),
                             ('RejectParameter2', '>u2')]),
            0x333: np.dtype([('hdr', hdr.dtype),           # TcExecute
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2')]),
            0x334: np.dtype([('hdr', hdr.dtype),           # TcFail
                             ('TcPacketId', '>u2'),
                             ('TcSeqControl', '>u2'),
                             ('TcFailCode', '>u2'),
                             ('FailParameter1', '>u2'),
                             ('FailParameter2', '>u2')]),
            0x335: np.dtype([('hdr', hdr.dtype),           # EventRp
                             ('Event_ID', 'u1'),
                             ('Event_Sev', 'u1'),
                             ('Word1', '>u2'),
                             ('Word2', '>u2'),
                             ('Word3', '>u2'),
                             ('Word4', '>u2'),
                             ('Word5', '>u2'),
                             ('Word6', '>u2'),
                             ('Word7', '>u2'),
                             ('Word8', '>u2')]),
            0x340: np.dtype([('hdr', hdr.dtype),           # MemDump
                             ('Image_ID', 'u1'),
                             ('_FillerByte', 'u1'),
                             ('Address32', '>u4'),
                             ('Length', '>u4'),
                             ('Data', 'u1', (2036,))]),
            0x341: np.dtype([('hdr', hdr.dtype),           # MemCheckRp
                             ('Image_ID', 'u1'),
                             ('_FillerByte', 'u1'),
                             ('Address32', '>u4'),
                             ('Length', '>u4'),
                             ('CheckSum', '>u4')]),
            0x33d: np.dtype([('HTR_1_IsEna', 'u1'),        # ThemTableRp
                             ('HTR_1_AtcCorMan', 'u1'),
                             ('HTR_1_THMCH', 'u1'),
                             ('_FillerByte1', 'u1'),
                             ('HTR_1_ManOutput', '>u2'),
                             ('HTR_1_ATC_SP', '>u4'),
                             ('HTR_1_ATC_P', '>u4'),
                             ('HTR_1_ATC_I', '>u4'),
                             ('HTR_1_ATC_I_INIT', '>u4'),
                             ('HTR_2_IsEna', 'u1'),
                             ('HTR_2_AtcCorMan', 'u1'),
                             ('HTR_2_THMCH', 'u1'),
                             ('_FillerByte2', 'u1'),
                             ('HTR_2_ManOutput', '>u2'),
                             ('HTR_2_ATC_SP', '>u4'),
                             ('HTR_2_ATC_P', '>u4'),
                             ('HTR_2_ATC_I', '>u4'),
                             ('HTR_2_ATC_I_INIT', '>u4'),
                             ('HTR_3_IsEna', 'u1'),
                             ('HTR_3_AtcCorMan', 'u1'),
                             ('HTR_3_THMCH', 'u1'),
                             ('_FillerByte3', 'u1'),
                             ('HTR_3_ManOutput', '>u2'),
                             ('HTR_3_ATC_SP', '>u4'),
                             ('HTR_3_ATC_P', '>u4'),
                             ('HTR_3_ATC_I', '>u4'),
                             ('HTR_3_ATC_I_INIT', '>u4'),
                             ('HTR_4_IsEna', 'u1'),
                             ('HTR_4_AtcCorMan', 'u1'),
                             ('HTR_4_THMCH', 'u1'),
                             ('_FillerByte4', 'u1'),
                             ('HTR_4_ManOutput', '>u2'),
                             ('HTR_4_ATC_SP', '>u4'),
                             ('HTR_4_ATC_P', '>u4'),
                             ('HTR_4_ATC_I', '>u4'),
                             ('HTR_4_ATC_I_INIT', '>u4')])
            }.get(ap_id(hdr), None)


# - main function ----------------------------------
def read_lv0_data(file_list: list[Path, ...],
                  file_format: str, *,
                  debug: bool = False) -> tuple[tuple, tuple]:
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
    hdr_dtype = _dtype_packet_(file_format)
    scihk_dtype = tmtc_dtype(0x350)
    icutm_dtype = np.dtype([('tai_sec', '>u4'),
                            ('sub_sec', '>u2')])

    # read level 0 headers and CCSDS data of Science and TmTc data
    ccsds_sci = ()
    ccsds_hk = ()
    for flname in file_list:
        offs = 0
        if file_format == 'dsb':
            cfe_hdr = _cfe_header_(flname)
            offs += cfe_hdr['FileHeaderLength']

        buff_sci = ()          # Use chunking to speed-up memory allocation
        buff_hk = ()
        with open(flname, 'rb') as fp:
            module_logger.info('processing file "%s"', flname)

            # read CCSDS header and user data
            ccsds_data = fp.read()
            while offs < len(ccsds_data):
                try:
                    hdr = np.frombuffer(ccsds_data, count=1, offset=offs,
                                        dtype=hdr_dtype)[0]
                except ValueError as exc:
                    module_logger.warning('header read error with "%s".', exc)
                    break

                if debug:
                    print(ap_id(hdr), grouping_flag(hdr),
                          hdr_dtype.itemsize, hdr['length'], offs)
                    if hdr['length'] == 0 or not 0x320 <= ap_id(hdr) < 0x351:
                        raise ValueError('corrupted CCSDS header detected')
                    offs += hdr_dtype.itemsize + hdr['length'] - 5
                    continue

                # copy the full CCSDS package
                if ap_id(hdr) == 0x350:                   # Science APID
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
                elif 0x320 <= ap_id(hdr) <= 0x335:           # other valid APIDs
                    buff = np.frombuffer(ccsds_data, count=1, offset=offs,
                                         dtype=dtype_tmtc(hdr))
                    buff_hk += (buff,)
                    offs += dtype_tmtc(hdr).itemsize
                else:
                    if hdr['length'] == 0:
                        raise ValueError('corrupted CCSDS header detected')
                    offs += hdr_dtype.itemsize + hdr['length'] - 5
        ccsds_sci += buff_sci
        ccsds_hk += buff_hk
        del ccsds_data

    module_logger.info('number of Science packages %d', len(ccsds_sci))
    module_logger.info('number of Engineering packages %d', len(ccsds_hk))

    return ccsds_sci, ccsds_hk


def dump_hkt(flname: str, ccsds_hk: tuple[np.ndarray, ...]):
    """Dump header info of the SPEXone housekeeping telemetry packets."""
    with Path(flname).open('w', encoding='ascii') as fp:
        fp.write('APID Grouping Counter Length     TAI_SEC    SUB_SEC'
                 ' ICUSWVER MPS_ID TcSeqControl TcErrorCode\n')
        for buf in ccsds_hk:
            hdr = buf['hdr'][0]
            msg = (f"{ap_id(hdr):4x} {grouping_flag(hdr):8d}"
                   f" {sequence(hdr):7d} {packet_length(hdr):6d}"
                   f" {hdr['tai_sec']:11d} {hdr['sub_sec']:10d}")

            _hk = buf['hk'][0] if 'hk' in buf else None
            msg += {0x320: f" {_hk['ICUSWVER']:8x} {_hk['MPS_ID']:6d}",
                    0x331: f" {-1:8x} {-1:6d} {buf['TcSeqControl'][0]:12d}",
                    0x332: (f" {-1:8x} {-1:6d} {buf['TcSeqControl'][0]:12d}"
                            f" {bin(buf['TcRejectCode'][0])}"
                            f" {buf['RejectParameter1'][0]}"
                            f" {buf['RejectParameter2'][0]}"),
                    0x333: f" {-1:8x} {-1:6d} {buf['TcSeqControl'][0]:12d}",
                    0x334: (f" {-1:8x} {-1:6d} {buf['TcSeqControl'][0]:12d}"
                            f" {bin(buf['TcFailCode'][0])}"
                            f" {buf['FailParameter1'][0]}"
                            f" {buf['FailParameter2'][0]}"),
                    0x335: (f" {-1:8x} {-1:6d} {buf['Event_ID'][0]:d}"
                            f" {buf['Event_Sev'][0]}")}.get(ap_id(hdr), '')
            fp.write(msg + '\n')


def dump_science(flname: str, ccsds_sci: tuple[np.ndarray, ...]):
    """Dump telemetry header info (Science)."""
    with Path(flname).open('w', encoding='ascii') as fp:
        fp.write('APID Grouping Counter Length'
                 ' ICUSWVER MPS_ID  IMRLEN     ICU_SEC ICU_SUBSEC\n')
        for segment in ccsds_sci:
            hdr = segment['hdr'][0]
            if grouping_flag(hdr) == 1:
                nom_hk = segment['hk']
                icu_tm = segment['icu_tm']
                fp.write(f"{ap_id(hdr):4x} {grouping_flag(hdr):8d}"
                         f" {sequence(hdr):7d} {packet_length(hdr):6d}"
                         f" {nom_hk['ICUSWVER'][0]:8x}"
                         f" {nom_hk['MPS_ID'][0]:6d}"
                         f" {nom_hk['IMRLEN'][0]:7d}"
                         f" {icu_tm['tai_sec'][0]:11d}"
                         f" {icu_tm['sub_sec'][0]:10d}\n")
            else:
                fp.write(f'{ap_id(hdr):4x} {grouping_flag(hdr):8d}'
                         f' {sequence(hdr):7d} {packet_length(hdr):6d}\n')
