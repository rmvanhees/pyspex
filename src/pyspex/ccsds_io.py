"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Class to read SPEXone ICU packages (version 2)

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path

import numpy as np

from pyspex.lib.tmtc_def import tmtc_dtype

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
HDR_DTYPE = np.dtype([
    ('type', '>u2'),
    ('sequence', '>u2'),
    ('length', '>u2')
])

# Defines parameters of a CCSDS timestamp
#  - Seconds     (32 bits): seconds (TAI)
#  - Sub-seconds (16 bits): sub-seconds (1/2 ** 16)
TIME_DTYPE = np.dtype([
    ('tai_sec', '>u4'),
    ('sub_sec', '>u2')
])

SCIHK_DTYPE = tmtc_dtype(0x350)

# - local functions --------------------------------


# - class CCSDSio -------------------------
class CCSDSio:
    """
    Read SPEXone telemetry packets.

    Attributes
    ----------
    found_invalid_apid :  bool
    file_list :  iter
    fp :  file pointer

    Methods
    -------
    version_no
       Returns CCSDS version number.
    type_indicator
       Returns type of telemetry packet.
    secnd_hdr_flag
       Returns flag indicating presence of a secondary header.
    ap_id
       Returns SPEXone ApID.
    grouping_flag
       Returns grouping flag.
    sequence_count
       Returns sequence counter, rollover to zero at 0x3FFF.
    packet_length
       Returns size of packet data in bytes.

    close()
       Close resources.
    open_next_file()
       Open next file from file_list.
    fix_sci_hk24(sci_hk)
       Correct 32-bit integers in the Science_HK which originate from
       24-bit integers in the detector register values.
    read_packet()
       Read next telemetry packet.
    nomhk_tm(packets_in)
       Select NomHK telemetry packages
    science_tm(packets_in)
       Combine segmented Science telemetry packages.

    Notes
    -----
    The formats of the PACE telemetry packets are following the standards:
    CCSDS-131.0-B-3, CCSDS-132.0-B-2 and CCSDS-133.0-B-1.

    This module is currenty restriced to telementry packets with APID:
    0x350 (Science), 0x320 (NomHK) and 0x322 (DemHK).

    A telemtry packet consist of a PRIMARY HEADER, an optionaly SECONDARY
    HEADER (consist of a timestamp) and USER DATA with the actual telemetry
    packet data.

    Doc: TMTC handbook (SPX1-TN-005), issue 12, 15-May-2020

    Examples
    --------
    >>> packets = ()
    >>> with CCSDSio(file_list) as ccsds:
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
    >>>     science_tm = ccsds.group_tm(packets)
    >>>     # now you may want to collect the engineering packages
    """
    def __init__(self, file_list: str) -> None:
        """
        Initialize access to a SPEXone Level-0 product (CCSDS format)

        Parameters
        ----------
        file_list: list of strings
           list of file-names, where each file contains parts of a measurement
        """
        # initialize class attributes
        self.__hdr = None
        self.found_invalid_apid = False
        self.file_list = iter(sorted(file_list))
        self.fp = None

        self.open_next_file()

    def __repr__(self) -> str:
        return '{:03d} {} {} 0x{:x} {} {:5d} {:5d} {}'.format(
            self.version_no, self.type_indicator, self.secnd_hdr_flag,
            self.ap_id, self.grouping_flag, self.sequence_count,
            self.packet_length, self.fp.tell())

    def __iter__(self):
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __enter__(self):
        """
        method called to initiate the context manager
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """
        method called when exiting the context manager
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self) -> None:
        """
        Close resources
        """
        if self.fp is not None:
            if self.found_invalid_apid:
                print('[WARNING]: found one or more telemetry packages'
                      ' with an invalid APID')
            self.found_invalid_apid = False
            self.fp.close()

    # ---------- define some class properties ----------
    @property
    def version_no(self) -> int:
        """
        Returns CCSDS version number
        """
        if self.__hdr is None:
            return None

        return (self.__hdr['type'] >> 13) & 0x7

    @property
    def type_indicator(self) -> int:
        """
        Returns type of telemetry packet
        """
        if self.__hdr is None:
            return None

        return (self.__hdr['type'] >> 12) & 0x1

    @property
    def secnd_hdr_flag(self) -> bool:
        """
        Returns flag indicating presence of a secondary header
        """
        if self.__hdr is None:
            return None

        return (self.__hdr['type'] >> 11) & 0x1

    @property
    def ap_id(self) -> int:
        """
        Returns SPEXone ApID
        """
        if self.__hdr is None:
            return None

        return self.__hdr['type'] & 0x7FF

    @property
    def grouping_flag(self) -> int:
        """
        Returns grouping flag

        Possible values:
          00 continuation packet-data segment
          01 first packet-data segment
          10 last packet-data segment
          11 packet-data unsegmented
        """
        if self.__hdr is None:
            return None

        return (self.__hdr['sequence'] >> 14) & 0x3

    @property
    def sequence_count(self) -> int:
        """
        Returns sequence counter, rollover to zero at 0x3FFF
        """
        if self.__hdr is None:
            return None

        return self.__hdr['sequence'] & 0x3FFF

    @property
    def packet_length(self) -> int:
        """
        Returns size of packet data in bytes

        Notes
        -----
        Value equals secondary header + user data (always odd)
        """
        if self.__hdr is None:
            return None

        return self.__hdr['length']

    # ---------- define empty telemetry packet ----------
    def open_next_file(self):
        """
        Open next file from file_list
        """
        flname = next(self.file_list)
        if not Path(flname).is_file():
            raise FileNotFoundError('{} does not exist'.format(flname))

        self.close()
        self.fp = open(flname, 'rb')

    @staticmethod
    def fix_dem_hk24(dem_hk):
        """
        Correct 32-bit integers in the DemHK which originate from
        24-bit integers in the detector register values
        """
        for key in ['DET_EXPTIME', 'DET_EXPSTEP',
                    'DET_KP1', 'DET_KP2', 'DET_EXPTIME2', 'DET_EXPSTEP2']:
            dem_hk[key] = dem_hk[key] >> 8

        return dem_hk

    @staticmethod
    def fix_nom_hk24(nom_hk):
        """
        Correct 32-bit integers in the NomHK which originate from
        24-bit integers in the detector register values
        """
        # for key in []:
        #     nom_hk[key] = nom_hk[key] >> 8

        return nom_hk

    @staticmethod
    def fix_sci_hk24(sci_hk):
        """
        Correct 32-bit integers in the Science HK which originate from
        24-bit integers in the detector register values

        In addition, copy the first 4 bytes of DET_CHENA to DET_ILVDS
        """
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

    def __tm(self, num_data=None):
        """
        Returns empty Science telemetry packet

        Parameters
        ----------
        num_data : int
          Size of the Science data in bytes (Science package, only)

        Returns
        -------
        ndarray
        """
        # NomHK telemetry packet
        if self.ap_id == 0x320:
            return np.zeros(1, dtype=np.dtype([
                ('primary_header', HDR_DTYPE),
                ('secondary_header', TIME_DTYPE),
                ('nominal_hk', tmtc_dtype(0x320))]))

        # DemHK telemetry packet
        if self.ap_id == 0x322:
            return np.zeros(1, dtype=np.dtype([
                ('primary_header', HDR_DTYPE),
                ('secondary_header', TIME_DTYPE),
                ('detector_hk', tmtc_dtype(0x322))]))

        # Science telemetry packet
        if self.ap_id == 0x350:
            return np.zeros(1, dtype=np.dtype([
                ('primary_header', HDR_DTYPE),
                ('secondary_header', TIME_DTYPE),
                ('science_hk', SCIHK_DTYPE),
                ('icu_time', TIME_DTYPE),
                ('image_data', 'u2', (num_data,))]))

        raise KeyError('unknown APID: {:d}'.format(self.ap_id))

    def read_packet(self):
        """
        Read next telemetry packet

        Returns
        -------
        ndarray
        """
        # read primary header
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

        # save primary header as class attribute
        self.__hdr = hdr[0]
        num_bytes = self.packet_length + 1

        time_tm = [None]
        if self.secnd_hdr_flag == 1:
            time_tm = np.fromfile(self.fp, count=1, dtype=TIME_DTYPE)
            num_bytes -= TIME_DTYPE.itemsize

        if self.ap_id == 0x350:             # Science telemetry packet
            # first segement or unsegmented data packet provides Science_HK
            if self.grouping_flag in (1, 3):
                hk_data = np.fromfile(self.fp, count=1, dtype=SCIHK_DTYPE)
                num_bytes -= SCIHK_DTYPE.itemsize
                time_icu = np.fromfile(self.fp, count=1, dtype=TIME_DTYPE)
                num_bytes -= TIME_DTYPE.itemsize

            # read detector image data
            data = np.fromfile(self.fp, dtype='>u2', count=num_bytes // 2)

            # combine telemetry data in a numpy dataset
            packet = self.__tm(data.size)
            packet['primary_header'] = hdr
            packet['secondary_header'] = time_tm
            if self.grouping_flag in (1, 3):
                packet['science_hk'] = self.fix_sci_hk24(hk_data)
                packet['icu_time'] = time_icu
            packet['image_data'] = data
        elif self.ap_id == 0x320:        # NomHK telemetry packet
            packet = self.__tm()
            packet['primary_header'] = hdr
            packet['secondary_header'] = time_tm
            hk_data = np.fromfile(self.fp, count=1, dtype=tmtc_dtype(0x320))
            packet['nominal_hk'] = self.fix_nom_hk24(hk_data)
        elif self.ap_id == 0x322:        # DemHK telemetry packet
            packet = self.__tm()
            packet['primary_header'] = hdr
            packet['secondary_header'] = time_tm
            hk_data = np.fromfile(self.fp, count=1, dtype=tmtc_dtype(0x322))
            packet['detector_hk'] = self.fix_dem_hk24(hk_data)
        else:
            if not 0x320 <= self.ap_id <= 0x350:
                self.found_invalid_apid = True

            packet = hdr
            # move to the next telemetry packet
            self.fp.seek(num_bytes, 1)

        return packet

    def nomhk_tm(self, packets_in: tuple):
        """
        Select NomHK telemetry packages

        Parameters
        ----------
        packets: tuple
           Tuple with science or house-keeping telemetry packages

        Returns
        -------
        Tuple with NomHK telemetry packages (chronological)
        """
        # reject non-NomHK telemetry packages
        packets = ()
        for packet in packets_in:
            if 'primary_header' not in packet.dtype.names:
                continue

            self.__hdr = packet['primary_header']
            if self.ap_id == 0x320:
                packets += (packet,)

        if not packets:
            return ()

        return packets

    def demhk_tm(self, packets_in: tuple):
        """
        Select DemHK telemetry packages

        Parameters
        ----------
        packets: tuple
           Tuple with science or house-keeping telemetry packages

        Returns
        -------
        Tuple with DemHK telemetry packages (chronological)
        """
        # reject non-DemHK telemetry packages
        packets = ()
        for packet in packets_in:
            if 'primary_header' not in packet.dtype.names:
                continue

            self.__hdr = packet['primary_header']
            if self.ap_id == 0x322:
                packets += (packet,)

        if not packets:
            return ()

        return packets

    def science_tm(self, packets_in: tuple):
        """
        Combine segmented Science telemetry packages

        Parameters
        ----------
        packets: tuple
           Tuple with science or house-keeping telemetry packages

        Returns
        -------
        Tuple with unsegmented Science telemetry packages (chronological)
        """
        # reject non-Science telemetry packages and non-segmented packages
        packets = ()
        for packet in packets_in:
            if 'primary_header' not in packet.dtype.names:
                continue

            self.__hdr = packet['primary_header']
            if self.ap_id == 0x350 and self.grouping_flag != 3:
                packets += (packet,)

        if not packets:
            return ()

        # first telemetry package must have grouping flag equals 1
        self.__hdr = packets[0]['primary_header']
        if self.grouping_flag != 1:
            msg = ('[WARNING]: rejected first image because it is incomplete'
                   ' - received only {:d} segments.')
            ii = 0
            for packet in packets:
                self.__hdr = packet['primary_header']
                if self.grouping_flag == 1:
                    break
                ii += 1

            print(msg.format(ii))
            packets = packets[ii:]

        # last telemetry package must have grouping flag equals 2
        self.__hdr = packets[-1]['primary_header']
        if self.grouping_flag != 2:
            msg = ('[WARNING]: rejected last image because it is incomplete'
                   ' - received only {:d} segments.')
            ii = len(packets)
            while ii > 0:
                ii -= 1
                self.__hdr = packets[ii]['primary_header']
                if self.grouping_flag == 2:
                    break

            print(msg.format(len(packets) - (ii + 1)))
            packets = packets[:ii+1]

        res = ()
        prev_grp_flag = 2
        for packet in packets:
            self.__hdr = packet['primary_header']

            # handle segmented data
            if self.grouping_flag == 1:
                # group_flag of previous package should be 2
                if prev_grp_flag != 2:
                    msg = ('corrupted segements - detected APID 1 after <> 2')
                    raise RuntimeError(msg)

                rec_buff = self.__tm(packet['science_hk']['IMRLEN'] // 2)[0]
                rec_buff['primary_header'] = packet['primary_header']
                rec_buff['secondary_header'] = packet['secondary_header']
                rec_buff['science_hk'] = packet['science_hk']
                img_buff = packet['image_data']
            elif self.grouping_flag in (0, 2):
                # group_flag of previous package should be 0 or 1
                if prev_grp_flag not in (0, 1):
                    msg = ('corrupted segements - detected segement of'
                           ' new image, however, previous not closed')
                    raise RuntimeError(msg)

                img_buff = np.concatenate((img_buff, packet['image_data']))
                if self.grouping_flag == 2:
                    if rec_buff['image_data'].size == img_buff.size:
                        rec_buff['image_data'] = img_buff
                        res += (rec_buff,)
                    else:
                        print('[WARNING]: rejected image'
                              ' because it is incomplete')

            # keep current group flag for next read
            prev_grp_flag = self.grouping_flag

        return res
