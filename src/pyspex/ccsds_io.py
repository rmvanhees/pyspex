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
Contains the class `CCSDSio` to read SPEXone telemetry packets.
"""
from __future__ import annotations
__all__ = ['CCSDSio']

from pathlib import Path

import numpy as np

from .lib.tmtc_def import tmtc_dtype

# - global parameters ------------------------------
# Define parameters of Primary header
#  - Packet type     (3 bits): Version No.
#                              Indicates this is a CCSDS version 1 packet
#                     (1 bit): Type indicator
#                              Indicates this is a telemetery packet
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

HDR_DTYPE = np.dtype([
    ('type', '>u2'),
    ('sequence', '>u2'),
    ('length', '>u2'),
    ('tai_sec', '>u4'),
    ('sub_sec', '>u2')
])

TIME_DTYPE = np.dtype([
    ('tai_sec', '>u4'),
    ('sub_sec', '>u2')
])

SCIHK_DTYPE = tmtc_dtype(0x350)

# - Error messages ------------------------
MSG_SKIP_FRAME = "[WARNING]: rejected a frame because it's incomplete"
MSG_INVALID_APID = \
    '[WARNING]: found one or more telemetry packages with an invalid APID'
MSG_CORRUPT_APID = 'corrupted segements - detected APID 1 after <> 2'
MSG_CORRUPT_FRAME = 'corrupted segements - previous frame not closed'


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
    def __init__(self, file_list: list) -> None:
        """Initialize access to a SPEXone Level-0 product (CCSDS format).
        """
        # initialize class attributes
        self.file_list = iter(file_list)
        self.found_invalid_apid = False
        self.__hdr = None
        self.fp = None

        if file_list:
            self.open_next_file()

    def __repr__(self) -> str:
        return (f'{self.version_no:03d} 0x{self.ap_id:x}'
                f' {self.grouping_flag} {self.sequence_count:5d}'
                f' {self.packet_length:5d} {self.fp.tell():9d}')

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __enter__(self):
        """Method called to initiate the context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Method called when exiting the context manager.
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self) -> None:
        """Close resources.
        """
        if self.fp is not None:
            if self.found_invalid_apid:
                print(MSG_INVALID_APID)
            self.found_invalid_apid = False
            self.fp.close()

    # ---------- define some class properties ----------
    @property
    def version_no(self) -> int | None:
        """Returns CCSDS version number.
        """
        if self.__hdr is None:
            return None

        return (self.__hdr['type'] >> 13) & 0x7

    @property
    def type_indicator(self) -> int | None:
        """Returns type of telemetry packet.
        """
        if self.__hdr is None:
            return None

        return (self.__hdr['type'] >> 12) & 0x1

    @property
    def secnd_hdr_flag(self) -> bool | None:
        """Returns flag indicating presence of a secondary header.
        """
        if self.__hdr is None:
            return None

        return (self.__hdr['type'] >> 11) & 0x1

    @property
    def ap_id(self) -> int | None:
        """Returns SPEXone ApID.
        """
        if self.__hdr is None:
            return None

        return self.__hdr['type'] & 0x7FF

    @property
    def grouping_flag(self) -> int | None:
        """Returns grouping flag

        The meaning of the grouping flag values are::

          00 continuation packet-data segment
          01 first packet-data segment
          10 last packet-data segment
          11 packet-data unsegmented

        """
        if self.__hdr is None:
            return None

        return (self.__hdr['sequence'] >> 14) & 0x3

    @property
    def sequence_count(self) -> int | None:
        """Returns sequence counter, rollover to zero at 0x3FFF.
        """
        if self.__hdr is None:
            return None

        return self.__hdr['sequence'] & 0x3FFF

    @property
    def packet_length(self) -> int | None:
        """Returns size of packet data in bytes.

        Value equals secondary header + user data (always odd)
        """
        if self.__hdr is None:
            return None

        return self.__hdr['length']

    # ---------- define empty telemetry packet ----------
    def open_next_file(self) -> None:
        """Open next file from file_list.
        """
        flname = next(self.file_list)
        if not Path(flname).is_file():
            raise FileNotFoundError(f'{flname} does not exist')

        self.close()
        # pylint: disable=consider-using-with
        self.fp = open(flname, 'rb')

    @staticmethod
    def fix_dem_hk24(dem_hk):
        """Correct 32-bit integers in the DemHK which originate from
        24-bit integers in the detector register values.

        Parameters
        ----------
        dem_hk : numpy.ndarray
           SPEXone DEM housekeeping packages

        Returns
        -------
        numpy.ndarray
           SPEXone DEM housekeeping packages
        """
        for key in ['DET_EXPTIME', 'DET_EXPSTEP',
                    'DET_KP1', 'DET_KP2', 'DET_EXPTIME2', 'DET_EXPSTEP2']:
            dem_hk[key] = dem_hk[key] >> 8

        return dem_hk

    @staticmethod
    def fix_sci_hk24(sci_hk):
        """Correct 32-bit integers in the Science HK which originate from
        24-bit integers in the detector register values.

        In addition:
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
        if np.all(sci_hk['ICUSWVER'] < 0x129):
            key = 'REG_BINNING_TABLE_START'
            sci_hk[key] = np.ndarray(shape=sci_hk.shape,
                                     dtype='<u4',
                                     buffer=sci_hk[key])

        sci_hk['DET_ILVDS'] = sci_hk['DET_CHENA'] & 0xf

        for key in ['TS1_DEM_N_T', 'TS2_HOUSING_N_T', 'TS3_RADIATOR_N_T',
                    'TS4_DEM_R_T', 'TS5_HOUSING_R_T', 'TS6_RADIATOR_R_T',
                    'LED1_ANODE_V', 'LED1_CATH_V', 'LED1_I',
                    'LED2_ANODE_V', 'LED2_CATH_V', 'LED2_I',
                    'ADC1_VCC', 'ADC1_REF', 'ADC1_T',
                    'ADC2_VCC', 'ADC2_REF', 'ADC2_T',
                    'DET_EXPTIME', 'DET_EXPSTEP', 'DET_KP1',
                    'DET_KP2', 'DET_EXPTIME2', 'DET_EXPSTEP2',
                    'DET_CHENA']:
            sci_hk[key] = sci_hk[key] >> 8

        return sci_hk

    def __rd_science(self, hdr) -> np.ndarray:
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
        packet = np.empty(1, dtype=np.dtype([
            ('packet_header', HDR_DTYPE),
            ('science_hk', SCIHK_DTYPE),
            ('icu_time', TIME_DTYPE),
            ('image_data', 'O')]))
        packet['packet_header'] = hdr

        # first segement or unsegmented data packet provides Science_HK
        if self.grouping_flag in (1, 3):
            packet['science_hk'] = self.fix_sci_hk24(
                np.fromfile(self.fp, count=1, dtype=SCIHK_DTYPE))
            num_bytes -= SCIHK_DTYPE.itemsize
            packet['icu_time'] = np.fromfile(self.fp, count=1,
                                             dtype=TIME_DTYPE)
            num_bytes -= TIME_DTYPE.itemsize

        # read detector image data
        packet['image_data'][0] = np.fromfile(self.fp, dtype='>u2',
                                              count=num_bytes // 2)
        return packet

    def __rd_nomhk(self, hdr) -> np.ndarray:
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
        packet = np.empty(1, dtype=np.dtype([
            ('packet_header', HDR_DTYPE),
            ('nominal_hk', tmtc_dtype(0x320))]))
        packet['packet_header'] = hdr
        packet['nominal_hk'] = np.fromfile(self.fp, count=1,
                                           dtype=tmtc_dtype(0x320))
        return packet

    def __rd_demhk(self, hdr) -> np.ndarray:
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
        packet = np.empty(1, dtype=np.dtype([
            ('packet_header', HDR_DTYPE),
            ('detector_hk', tmtc_dtype(0x322))]))
        packet['packet_header'] = hdr
        packet['detector_hk'] = self.fix_dem_hk24(
            np.fromfile(self.fp, count=1, dtype=tmtc_dtype(0x322)))
        return packet

    def __rd_tc_accept(self, _) -> np.ndarray:
        """Read/dump TcAccept packet.
        """
        self.fp.seek(-1 * HDR_DTYPE.itemsize, 1)
        packet = np.fromfile(self.fp, count=1, dtype=np.dtype([
            ('packet_header', HDR_DTYPE),
            ('TcPacketId', '>u2'),
            ('TcSeqControl', '>u2')]))
        print('[TcAccept]: ', self, packet['TcPacketId'][0],
              packet['TcSeqControl'][0])
        return packet

    def __rd_tc_execute(self, _) -> np.ndarray:
        """Read/dump TcExecute packet.
        """
        self.fp.seek(-1 * HDR_DTYPE.itemsize, 1)
        packet = np.fromfile(self.fp, count=1, dtype=np.dtype([
            ('packet_header', HDR_DTYPE),
            ('TcPacketId', '>u2'),
            ('TcSeqControl', '>u2')]))
        print('[TcExecute]:', self, packet['TcPacketId'][0],
              packet['TcSeqControl'][0])
        return packet

    def __rd_tc_fail(self, _) -> np.ndarray:
        """Read/dump TcFail packet.
        """
        self.fp.seek(-1 * HDR_DTYPE.itemsize, 1)
        packet = np.fromfile(self.fp, count=1, dtype=np.dtype([
            ('packet_header', HDR_DTYPE),
            ('TcPacketId', '>u2'),
            ('TcSeqControl', '>u2'),
            ('TcFailCode', '>u2'),
            ('FailParameter1', '>u2'),
            ('FailParameter2', '>u2')]))
        print('[TcFail]:   ', self, packet['TcPacketId'][0],
              packet['TcSeqControl'][0], bin(packet['TcFailCode'][0]),
              packet['FailParameter1'][0], packet['FailParameter2'][0]
              )
        return packet

    def __rd_tc_reject(self, _) -> np.ndarray:
        """Read/dump TcReject packet.
        """
        self.fp.seek(-1 * HDR_DTYPE.itemsize, 1)
        packet = np.fromfile(self.fp, count=1, dtype=np.dtype([
            ('packet_header', HDR_DTYPE),
            ('TcPacketId', '>u2'),
            ('TcSeqControl', '>u2'),
            ('TcRejectCode', '>u2'),
            ('RejectParameter1', '>u2'),
            ('RejectParameter2', '>u2')]))
        print('[TcReject]: ', self, packet['TcPacketId'][0],
              packet['TcSeqControl'][0], bin(packet['TcRejectCode'][0]),
              packet['RejectParameter1'][0], packet['RejectParameter2'][0])
        return packet

    def __rd_other(self, hdr) -> np.ndarray | None:
        """Read other telemetry packet.
        """
        num_bytes = self.packet_length - TIME_DTYPE.itemsize + 1
        if not 0x320 <= self.ap_id <= 0x350:
            self.found_invalid_apid = True
            self.fp.seek(num_bytes, 1)
            return None

        packet = np.empty(1, dtype=np.dtype([
            ('packet_header', HDR_DTYPE),
            ('raw_data', 'u1', num_bytes)]))
        packet['packet_header'] = hdr
        packet['raw_data'] = np.fromfile(self.fp, count=num_bytes, dtype='u1')
        return packet

    def read_packet(self) -> np.ndarray | None:
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
        rd_func = {0x350: self.__rd_science,
                   0x320: self.__rd_nomhk,
                   0x322: self.__rd_demhk,
                   0x331: self.__rd_tc_accept,
                   0x332: self.__rd_tc_reject,
                   0x333: self.__rd_tc_execute,
                   0x334: self.__rd_tc_fail}.get(self.ap_id, self.__rd_other)
        return rd_func(hdr)

    @staticmethod
    def select_tm(packets_in: tuple, ap_id: int) -> tuple:
        """Select telemetry packages on SPEXone ApID.

        Parameters
        ----------
        packets_in: tuple
           Tuple with mix of SPEXone telemetry packages
        ap_id: int
           SPEXone ApID

        Returns
        -------
        tuple
           selected telemetry packages
        """
        packets = ()
        for packet in packets_in:
            if 'packet_header' not in packet.dtype.names:
                continue

            if (packet['packet_header']['type'] & 0x7FF) == ap_id:
                packets += (packet,)

        return packets

    def science_tm(self, packets_in: tuple) -> tuple:
        """Combine segmented Science telemetry packages.

        Parameters
        ----------
        packets_in: tuple
           Tuple with science or house-keeping telemetry packages

        Returns
        -------
        tuple
           unsegmented Science telemetry packages
        """
        # reject non-Science telemetry packages
        packets = self.select_tm(packets_in, 0x350)
        if not packets:
            return ()

        # check if grouping_flag of first segement equals 1
        #   else reject all segments with grouping_flag != 1
        self.__hdr = packets[0]['packet_header']
        if self.grouping_flag != 1:
            ii = 0
            for packet in packets:
                self.__hdr = packet['packet_header']
                if self.grouping_flag == 1:
                    break
                ii += 1

            print(f'[WARNING]: first frame incomplete - skipped {ii} segments')
            packets = packets[ii:]
            if not packets:
                return ()

        # check if grouping_flag of last segement equals 2
        #   else reject all segments after the last segment
        #   with grouping_flag == 2
        self.__hdr = packets[-1]['packet_header']
        if self.grouping_flag != 2:
            ii = 0
            for packet in packets:
                self.__hdr = packet['packet_header']
                if self.grouping_flag == 2:
                    break
                ii += 1

            print(f'[WARNING]: last frame incomplete - rejected {ii} segments')
            packets = packets[:-ii]
            if not packets:
                return ()

        res = ()
        offs = 0
        prev_grp_flag = 2
        for packet in packets:
            grouping_flag = (packet['packet_header']['sequence'] >> 14) & 0x3
            # print(prev_grp_flag, grouping_flag, len(res), offs,
            #      packet['image_data'].size)
            # handle segmented data
            if grouping_flag == 1:       # first segment
                # group_flag of previous package should be 2
                if prev_grp_flag != 2:
                    if packet['image_data'].size in (3853, 7853):
                        print(MSG_SKIP_FRAME)
                        prev_grp_flag = 2
                        offs = 0
                    else:
                        raise RuntimeError(MSG_CORRUPT_APID)

                img_size = packet['science_hk']['IMRLEN'] // 2
                rec_buff = np.empty(1, dtype=np.dtype([
                    ('packet_header', HDR_DTYPE),
                    ('science_hk', SCIHK_DTYPE),
                    ('icu_time', TIME_DTYPE),
                    ('image_data', 'O')]))[0]
                img_buff = np.empty(img_size, dtype='u2')

                rec_buff['packet_header'] = packet['packet_header']
                rec_buff['science_hk'] = packet['science_hk']
                rec_buff['icu_time'] = packet['icu_time']
                img_buff[offs:offs + packet['image_data'].size] = \
                    packet['image_data']
                offs += packet['image_data'].size
            else:                        # continuation or last segment
                # group_flag of previous package should be 0 or 1
                if prev_grp_flag == 2:
                    raise RuntimeError(MSG_CORRUPT_FRAME)

                img_buff[offs:offs + packet['image_data'].size] = \
                    packet['image_data']
                offs += packet['image_data'].size
                if grouping_flag == 2:
                    if offs == img_size:
                        rec_buff['image_data'] = img_buff
                        res += (rec_buff,)
                    else:
                        print(MSG_SKIP_FRAME)
                    offs = 0

            # keep current group flag for next read
            prev_grp_flag = grouping_flag

        return res
