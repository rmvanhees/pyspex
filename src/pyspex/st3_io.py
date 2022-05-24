"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Classes to read SPEXone Level-0 packages are ScienceCCSDS and TmTcCCSDS for
resp. Science data and telemetry data. Both classes can be used to read data
in pure CCSDS format and ST3 format. The latter wil generated after integration
of SPEXone on the PACE platform.

Both classes have a much higher performance than obsolete class CCSDSio.

ToDo
----
- Remove the 16-bit ITOS header from the ST3-format hdr_dtype definition
- Add data sanity checks

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np

from pyspex.lib.tmtc_def import tmtc_dtype

# - global parameters ------------------------------
FULLFRAME_BYTES = 2 * 2048 * 2048


# - local functions --------------------------------
def hdr_dtype(st3_format=True) -> np.dtype:
    """
    Returns numpy dtype for packet header of data in CCSDS or ST3 format
    """
    if st3_format:
        return np.dtype([('itos_hdr', '>u2', (8,)),
                         ('spacewire', '>u2'),
                         ('type', '>u2'),
                         ('sequence', '>u2'),
                         ('length', '>u2'),
                         ('tai_sec', '>u4'),
                         ('sub_sec', '>u2')])

    return np.dtype([('type', '>u2'),
                     ('sequence', '>u2'),
                     ('length', '>u2'),
                     ('tai_sec', '>u4'),
                     ('sub_sec', '>u2')])


def ap_id(hdr: np.ndarray) -> int:
    """
    Returns Telemetry APID, the range 0x320 to 0x351 is available to SPEXone

    The following values are recognized:
        0x350 : Science
        0x320 : NomHk
        0x322 : DemHk
        0x331 : TcAccept
        0x332 : TcReject
        0x333 : TcExecute
        0x334 : TcFail
    """
    return hdr['type'] & 0x7FF


def grouping_flag(hdr: np.ndarray) -> int:
    """
    Returns grouping flag

    The 2-byte flag is encoded as follows:
        00 continuation packet-data segment
        01 first packet-data segment
        10 last packet-data segment
        11 packet-data unsegmented
    """
    return (hdr['sequence'] >> 14) & 0x3


def sequence(hdr: np.ndarray) -> int:
    """
    Returns sequence counter, rollover to zero at 0x3FFF
    """
    return hdr['sequence'] & 0x3FFF


def packet_length(hdr: np.ndarray) -> int:
    """
    Returns size of secondary header + user data - 1 in bytes

    Notes
    -----
    We always read the primary header and secondary header at once.

    Value range: 7 - 16375
    """
    return hdr['length']


def split_ccsds(ccsds_data: bytes) -> tuple:
    """
    Split Science and Telemetry data and store each in tuples (ST3 only)
    """
    _hdr_dtype = hdr_dtype(True)
    _hdr_bytes = _hdr_dtype.itemsize - 5

    offs = 0
    ccsds_sci = ()
    ccsds_hk = ()
    while offs < len(ccsds_data):
        hdr = np.frombuffer(ccsds_data, count=1, offset=offs,
                            dtype=_hdr_dtype)[0]
        nbytes = hdr['length'] + _hdr_bytes
        if ap_id(hdr) == 0x350:
            ccsds_sci += (ccsds_data[offs:offs+nbytes],)
        else:
            ccsds_hk += (ccsds_data[offs:offs+nbytes],)
        offs += nbytes

    return (ccsds_sci, ccsds_hk)


def select_science(ccsds_data: bytes) -> tuple:
    """
    Select Science CCSDS packages
    """
    _hdr_dtype = hdr_dtype(False)
    _hdr_bytes = _hdr_dtype.itemsize - 5

    offs = 0
    ccsds_sci = ()
    while offs < len(ccsds_data):
        hdr = np.frombuffer(ccsds_data, count=1, offset=offs,
                            dtype=_hdr_dtype)[0]
        nbytes = hdr['length'] + _hdr_bytes
        if ap_id(hdr) == 0x350:
            ccsds_sci += (ccsds_data[offs:offs+nbytes],)
        offs += nbytes

    return ccsds_sci


def select_hk(ccsds_data: bytes) -> tuple:
    """
    Select telemetry CCSDS packages
    """
    _hdr_dtype = hdr_dtype(False)
    _hdr_bytes = _hdr_dtype.itemsize - 5

    offs = 0
    ccsds_hk = ()
    while offs < len(ccsds_data):
        hdr = np.frombuffer(ccsds_data, count=1, offset=offs,
                            dtype=_hdr_dtype)[0]
        nbytes = hdr['length'] + _hdr_bytes
        if ap_id(hdr) != 0x350:
            ccsds_hk += (ccsds_data[offs:offs+nbytes],)
        offs += nbytes

    return ccsds_hk


# - class definitions ------------------------------
class ScienceCCSDS():
    """
    Read SPEXone Science data
    """
    APID = 0x350

    def __init__(self, ccsds_data: tuple, verbose=False):
        """
        Parameters
        ----------
        ccsds_data :  tuple
           SPEXone science data packages in CCSDS or ST3 format
        verbose :  bool, optional
           Be verbose
        """
        self.verbose = verbose
        self.ccsds_data = ccsds_data
        self._offs = 0

        # check if data-format is CCSDS or ST3
        if self.is_ccsds:
            st3_format = False
            self.hdr_dtype = hdr_dtype(st3_format)
        elif self.is_st3:
            st3_format = True
            self.hdr_dtype = hdr_dtype(st3_format)
        else:
            raise ValueError('format error')
        self.icutm_dtype = np.dtype([('tai_sec', '>u4'),
                                     ('sub_sec', '>u2')])
        if verbose:
            print('[INFO]: Science data is in'
                  f' {"ST3" if st3_format else "CCSDS"} format')

        # start processing at the first package with grouping flag in (1, 3)
        indx = self.find_first_segment()
        if indx > 0:
            self.ccsds_data = self.ccsds_data[indx:]

    @property
    def scihk_dtype(self) -> np.dtype:
        """
        Returns numpy dtype for Science housekeeping data
        """
        return tmtc_dtype(self.APID)

    @property
    def is_ccsds(self) -> bool:
        """
        Returns True if data is in CCSDS format
        """
        if self.ccsds_data:
            hdr = np.frombuffer(self.ccsds_data[0], count=1, offset=0,
                                dtype=hdr_dtype(st3_format=False))
            if (hdr['type'] >> 11) == 1 and ap_id(hdr) == self.APID:
                return True

        return False

    @property
    def is_st3(self) -> bool:
        """
        Returns True if data is in ST3 format
        """
        if self.ccsds_data:
            hdr = np.frombuffer(self.ccsds_data[0], count=1, offset=0,
                                dtype=hdr_dtype(st3_format=True))
            if (hdr['type'] >> 11) == 1 and ap_id(hdr) == self.APID:
                return True

        return False

    def find_first_segment(self) -> int:
        """
        Find byte offset to first segment with grouping flag equal 1 or 3
        """
        indx = 0
        for segment in self.ccsds_data:
            hdr = np.frombuffer(segment, count=1, offset=0,
                                dtype=self.hdr_dtype)[0]
            if ap_id(hdr) == self.APID and grouping_flag(hdr) in (1, 3):
                break
            indx += 1

        if self.verbose:
            print(f'[INFO]: found first valid segment at {indx}')
        return indx

    # def check(self) -> None:
    #    """
    #    Perform sanity checks, reading packet-headers only
    #    """
    #    return

    @staticmethod
    def _fix_hk24_(sci_hk):
        """
        Correct 32-bit integers in the Science HK which originate from
        24-bit integers in the detector register values

        In addition:
         - copy the first 4 bytes of DET_CHENA to DET_ILVDS
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

    def dump(self, flname: str) -> None:
        """
        Dump header information of all data segments

        Parameters
        ----------
        flname :  str
          Name of the ASCII file to be overwritten with a data dump
        """
        # pylint: disable=consider-using-with
        fp = open(flname, 'w', encoding='ascii')
        fp.write('APID Grouping Counter Length'
                 ' ICUSWVER MPS_ID  IMRLEN     ICU_SEC ICU_SUBSEC\n')

        # process all data
        for segment in self.ccsds_data:
            try:
                hdr = np.frombuffer(segment, count=1, offset=0,
                                    dtype=self.hdr_dtype)[0]
                offs = self.hdr_dtype.itemsize
                if grouping_flag(hdr) in (1, 3):
                    sci_hk = np.frombuffer(segment, count=1, offset=offs,
                                           dtype=self.scihk_dtype)[0]
                    offs += self.scihk_dtype.itemsize
                    icu_tm = np.frombuffer(segment, count=1, offset=offs,
                                           dtype=self.icutm_dtype)[0]
                    offs += self.icutm_dtype.itemsize
                    fp.write(f"{ap_id(hdr):4x} {grouping_flag(hdr):8d}"
                             f" {sequence(hdr):7d} {packet_length(hdr):6d}"
                             f" {sci_hk['ICUSWVER']:8x} {sci_hk['MPS_ID']:6d}"
                             f" {sci_hk['IMRLEN']:7d} {icu_tm['tai_sec']:11d}"
                             f" {icu_tm['sub_sec']:10d}\n")
                else:
                    fp.write(f'{ap_id(hdr):4x} {grouping_flag(hdr):8d}'
                             f' {sequence(hdr):7d} {packet_length(hdr):6d}\n')
            except EOFError:
                pass
            except Exception as exc:
                raise RuntimeError('an exception occured during read') from exc

        fp.close()

    def read(self, select='all') -> tuple:
        """
        Read SPEXone Science data into a tuple of numpy arrays
        """
        diff_bytes = self.icutm_dtype.itemsize - 1
        diff_bytes1 = \
            2 * self.icutm_dtype.itemsize + self.scihk_dtype.itemsize - 1

        buff = np.empty(1, dtype=np.dtype([('hdr', self.hdr_dtype),
                                           ('hk', self.scihk_dtype),
                                           ('icu_tm', self.icutm_dtype),
                                           ('frame', 'O')]))
        # process all data
        frame = ()
        packets = ()
        for segment in self.ccsds_data:
            try:
                hdr = np.frombuffer(segment, count=1, offset=0,
                                    dtype=self.hdr_dtype)[0]
                offs = self.hdr_dtype.itemsize
                if grouping_flag(hdr) == 1:
                    buff['hdr'] = hdr
                    buff['hk'] = self._fix_hk24_(
                        np.frombuffer(segment, count=1, offset=offs,
                                      dtype=self.scihk_dtype)[0])
                    offs += self.scihk_dtype.itemsize
                    buff['icu_tm'] = np.frombuffer(segment, count=1,
                                                   offset=offs,
                                                   dtype=self.icutm_dtype)[0]
                    offs += self.icutm_dtype.itemsize
                    frame = ()
                    nbytes = hdr['length'] - diff_bytes1
                else:
                    nbytes = hdr['length'] - diff_bytes
                frame += (np.frombuffer(segment, dtype='>u2',
                                        count=nbytes // 2, offset=offs),)
                if grouping_flag(hdr) == 2:
                    buff['frame'][0] = np.concatenate(frame)
                    if select == 'all':
                        packets += (buff.copy(),)
                    elif (select == 'binned'
                          and buff['hk']['IMRLEN'][0] < FULLFRAME_BYTES):
                        packets += (buff.copy(),)
                    elif (select == 'fullFrame'
                          and buff['hk']['IMRLEN'][0] == FULLFRAME_BYTES):
                        packets += (buff.copy(),)
            except EOFError:
                pass
            except Exception as exc:
                raise RuntimeError('an exception occured') from exc

        return packets


# --------------------------------------------------
class TmTcCCSDS():
    """
    Read SPEXone telemetry data
    """
    VALID_APID = [0x320, 0x322, 0x331, 0x332, 0x333, 0x334, 0x335]

    def __init__(self, ccsds_data: tuple, verbose=False):
        """
        Parameters
        ----------
        ccsds_data :  tuple
           SPEXone telemetry data packages in CCSDS or ST3 format
        verbose :  bool, optional
           Be verbose
        """
        self.verbose = verbose
        self.ccsds_data = ccsds_data
        self._offs = 0

        # check if data-format is CCSDS or ST3
        if self.is_ccsds:
            st3_format = False
            self.hdr_dtype = hdr_dtype(st3_format)
        elif self.is_st3:
            st3_format = True
            self.hdr_dtype = hdr_dtype(st3_format)
        else:
            raise ValueError('format error')
        if verbose:
            print('[INFO]: TmTc data is in'
                  f' {"ST3" if st3_format else "CCSDS"} format')

    @property
    def is_ccsds(self) -> bool:
        """
        Returns True if data is in CCSDS format
        """
        hdr = np.frombuffer(self.ccsds_data[0], count=1, offset=0,
                            dtype=hdr_dtype(st3_format=False))
        if (hdr['type'] >> 11) == 1 and ap_id(hdr) in self.VALID_APID:
            return True

        return False

    @property
    def is_st3(self) -> bool:
        """
        Returns True if data is in ST3 format
        """
        hdr = np.frombuffer(self.ccsds_data[0], count=1, offset=0,
                            dtype=hdr_dtype(st3_format=True))
        if (hdr['type'] >> 11) == 1 and ap_id(hdr) in self.VALID_APID:
            return True

        return False

    @property
    def dtype_dict(self) -> dict:
        """
        Return dictionary with numpy dtype definitions for telemetry packages
        """
        return {0x320: np.dtype([('hdr', self.hdr_dtype),
                                 ('hk', tmtc_dtype(0x320))]),
                0x322: np.dtype([('hdr', self.hdr_dtype),
                                 ('hk', tmtc_dtype(0x322))]),
                0x331: np.dtype([('hdr', self.hdr_dtype),
                                 ('TcPacketId', '>u2'),
                                 ('TcSeqControl', '>u2')]),
                0x332: np.dtype([('hdr', self.hdr_dtype),
                                 ('TcPacketId', '>u2'),
                                 ('TcSeqControl', '>u2'),
                                 ('TcErrorCode', '>u2'),
                                 ('RejectParameter1', '>u2'),
                                 ('RejectParameter2', '>u2')]),
                0x333: np.dtype([('hdr', self.hdr_dtype),
                                 ('TcPacketId', '>u2'),
                                 ('TcSeqControl', '>u2')]),
                0x334: np.dtype([('hdr', self.hdr_dtype),
                                 ('TcPacketId', '>u2'),
                                 ('TcSeqControl', '>u2'),
                                 ('TcErrorCode', '>u2'),
                                 ('FailParameter1', '>u2'),
                                 ('FailParameter2', '>u2')])}

    def dump(self, flname: str) -> None:
        """
        Dump header information of all data segments

        Parameters
        ----------
        flname :  str
          Name of the ASCII file to be overwritten with a data dump
        """
        # pylint: disable=consider-using-with
        fp = open(flname, 'w', encoding='ascii')
        fp.write('APID Grouping Counter Length     TAI_SEC    SUB_SEC'
                 ' ICUSWVER MPS_ID TcSeqControl TcErrorCode\n')

        for segment in self.ccsds_data:
            try:
                hdr = np.frombuffer(segment, count=1, offset=0,
                                    dtype=self.hdr_dtype)[0]
                offs = self.hdr_dtype.itemsize
                if ap_id(hdr) == 0x320:
                    mon_hk = np.frombuffer(segment, count=1, offset=offs,
                                           dtype=tmtc_dtype(0x320))[0]
                    fp.write(f"{ap_id(hdr):4x} {grouping_flag(hdr):8d}"
                             f" {sequence(hdr):7d} {packet_length(hdr):6d}"
                             f" {hdr['tai_sec']:11d} {hdr['sub_sec']:10d}"
                             f" {mon_hk['ICUSWVER']:8x}"
                             f" {mon_hk['MPS_ID']:6d}\n")
                elif ap_id(hdr) in (0x332, 0x334):
                    tc_dtype = self.dtype_dict[ap_id(hdr)]
                    tc_val = np.frombuffer(segment, count=1, offset=0,
                                           dtype=tc_dtype)[0]
                    fp.write(f"{ap_id(hdr):4x} {grouping_flag(hdr):8d}"
                             f" {sequence(hdr):7d} {packet_length(hdr):6d}"
                             f" {hdr['tai_sec']:11d} {hdr['sub_sec']:10d}"
                             f" {-1:8x} {-1:6d} {tc_val['TcSeqControl']:12d}"
                             f" {bin(tc_val['TcErrorCode'])}\n")
                else:
                    fp.write(f"{ap_id(hdr):4x} {grouping_flag(hdr):8d}"
                             f" {sequence(hdr):7d} {packet_length(hdr):6d}"
                             f" {hdr['tai_sec']:11d} {hdr['sub_sec']:10d}\n")
            except Exception as exc:
                raise RuntimeError('an exception occured during read') from exc

        fp.close()

    def read(self):
        """
        Read SPEXone telemetry data into a tuple of numpy arrays
        """
        # reset private class attribute
        packets = ()
        for segment in self.ccsds_data:
            try:
                hdr = np.frombuffer(segment, count=1, offset=0,
                                    dtype=self.hdr_dtype)[0]
                tm_dtype = self.dtype_dict.get(ap_id(hdr), None)
                if tm_dtype is not None:
                    packets += (np.frombuffer(segment, count=1, offset=0,
                                              dtype=tm_dtype),)
            except Exception as exc:
                raise RuntimeError('an exception occured during read') from exc

        return packets
