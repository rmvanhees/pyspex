"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Class to read SPEXone ICU packages

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np

from pyspex.lib.tmtc_def import tmtc_def

# - global parameters ------------------------------


# - local functions --------------------------------


# - class CCSDSio -------------------------
class CCSDSio:
    """
    Defines SPEXone L0 data package

    Doc: TMTC handbook (SPX1-TN-005), issue 12, 2020-05-15
    """
    def __init__(self, tmtc_issue=12, verbose=False):
        """
        Parameters
        ----------
        tmtc_issue: int
           Issue of the TMTC handbook which contains the definition of the
           Science Data Header format. Default: 12
        verbose: bool
           Be verbose. Default: be not verbose
        """
        # initialize class attributes
        self.offset = None
        self.__hdr = None
        self.segmented = True
        self.tmtc_issue = tmtc_issue
        self.verbose = verbose

    @staticmethod
    def __hdr1_def():
        """
        Defines parameters of Primary header
        - Packet type     (3 bits): Version No.
                                    Indicates this is a CCSDS version 1 packet
                           (1 bit): Type indicator
                                    Indicates this is a telemetery packet
                           (1 bit): Secondary flag
                                    Indicate presence of Secondary header
                         (11 bits): ApID
                                    SPEXone ApID [0x320 - 0x351] or 2047

        - Packet Sequence (2 bits): Grouping flag
                                    00 continuation packet-data segment
                                    01 first packet-data segment
                                    10 last packet-data segment
                                    11 packet-data unsegmented
                         (14 bits): Counter per ApID, rollover to 0 at 0x3FFF
        - Packet length  (16 bits): size of packet data in bytes (always odd)
                                    (secondary header + User data) - 1
        """
        return [
            ('type', '>u2'),         # 0x000
            ('sequence', '>u2'),     # 0x002
            ('length', '>u2')        # 0x004
        ]

    @staticmethod
    def __timestamp():
        """
        Defines parameters of a timestamp
        - Seconds     (32 bits): seconds (TAI)
        - Sub-seconds (16 bits): sub-seconds (1/2 ** 16)
        """
        return [
            ('tai_sec', '>u4'),      # 0x006
            ('sub_sec', '>u2')       # 0x00A
        ]

    @property
    def version_no(self):
        """
        Returns CCSDS version number
        """
        if self.__hdr is None:
            return None

        return (self.__hdr['type'] >> 13) & 0x7

    @property
    def type_indicator(self):
        """
        Returns type of telemetry packet
        """
        if self.__hdr is None:
            return None

        return (self.__hdr['type'] >> 12) & 0x1

    @property
    def secnd_hdr_flag(self):
        """
        Returns flag indicating presence of a secondary header
        """
        if self.__hdr is None:
            return None

        return (self.__hdr['type'] >> 11) & 0x1

    @property
    def ap_id(self):
        """
        Returns SPEXone ApID
        """
        if self.__hdr is None:
            return None

        return self.__hdr['type'] & 0x7FF

    @property
    def grouping_flag(self):
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
    def sequence_count(self):
        """
        Returns sequence counter, rollover to zero at 0x3FFF
        """
        if self.__hdr is None:
            return None

        return self.__hdr['sequence'] & 0x3FFF

    @property
    def packet_length(self):
        """
        Returns size of packet data in bytes
          Value equals secondary header + user data (always odd)
        """
        if self.__hdr is None:
            return None

        return self.__hdr['length']

    def __tm(self, num_data):
        """
        Return empty telemetry packet
        """
        return np.zeros(1, dtype=np.dtype([
            ('primary_header', np.dtype(self.__hdr1_def())),
            ('secondary_header', np.dtype(self.__timestamp())),
            ('mps', np.dtype(tmtc_def(0x350))),
            ('icu_time', np.dtype(self.__timestamp())),
            ('image_data', 'u2', (num_data,))]))

    def __rd_tm_packet(self, flname):
        """
        Read next telemetry packet

        Parameters
        ----------
        None

        Returns
        -------
        TM packet: primary & secondary header, MPS and image-data
        """
        hdr1_dtype = np.dtype(self.__hdr1_def())
        hdr2_dtype = np.dtype(self.__timestamp())
        mps_dtype = np.dtype(tmtc_def(0x350))

        # read parts of one telemetry packet data
        with open(flname, 'rb') as fp:
            hdr_one = np.fromfile(fp, dtype=hdr1_dtype,
                                  count=1, offset=self.offset)
            if hdr_one.size == 0:
                return None

            self.__hdr = hdr_one[0]
            if self.verbose:
                print('[DEBUG] ApID: ', self.ap_id,
                      self.secnd_hdr_flag, self.grouping_flag,
                      self.sequence_count, self.packet_length)
            if self.ap_id != 0x350:
                return None

            hdr_two = None
            mps = None
            icu_time = None
            num_bytes = self.packet_length + 1
            if self.secnd_hdr_flag == 1:
                hdr_two = np.fromfile(fp, dtype=hdr2_dtype, count=1)[0]
                num_bytes -= hdr2_dtype.itemsize

            # MPS is provided in first segement or unsegmented data packet
            mps = None
            if self.grouping_flag in (1, 3):
                mps = np.fromfile(fp, dtype=mps_dtype, count=1)[0]
                num_bytes -= mps_dtype.itemsize
                if self.tmtc_issue == 12:
                    icu_time = np.fromfile(fp, dtype=hdr2_dtype, count=1)[0]
                    num_bytes -= hdr2_dtype.itemsize

                # Correct 32-bit integers which originate from 24-bit
                # Necessary due to an allignment problem, in addition,
                # the 4 bytes of DET_CHENA also contain DET_ILVDS
                key_list = ['DET_EXPTIME', 'DET_EXPSTEP', 'DET_KP1',
                            'DET_KP2', 'DET_EXPTIME2', 'DET_EXPSTEP2',
                            'DET_CHENA']
                mps['DET_ILVDS'] = mps['DET_CHENA'] & 0xf
                for key in key_list:
                    mps[key] = mps[key] >> 8

            # remainder is image data
            data = np.fromfile(fp, dtype='>u2', count=num_bytes // 2)
            if self.verbose:
                msg = '[INFO]: {:d} {:d} {:3d} {:d} {:5d} {:d} {:d} {:d}  {:9d}'
                if mps is None:
                    print(msg.format(
                        self.secnd_hdr_flag, self.grouping_flag,
                        self.ap_id, hdr_two['tai_sec'], hdr_two['sub_sec'],
                        fp.tell() - self.offset, data.nbytes,
                        self.packet_length, fp.tell()))
                else:
                    msg += ' {:3d} {:9d}'
                    print(msg.format(
                        self.secnd_hdr_flag, self.grouping_flag,
                        self.ap_id, hdr_two['tai_sec'], hdr_two['sub_sec'],
                        fp.tell() - self.offset, data.nbytes,
                        self.packet_length, fp.tell(),
                        mps['MPS_ID'], mps['IMRLEN']))

        # combine parts to telemetry packet
        tm_packet = self.__tm(data.size)
        tm_packet[0]['primary_header'] = self.__hdr
        if hdr_two is not None:
            tm_packet[0]['secondary_header'] = hdr_two
        if mps is not None:
            tm_packet[0]['mps'] = mps
        if icu_time is not None:
            tm_packet[0]['icu_time'] = icu_time
        tm_packet[0]['image_data'] = data

        # move offset to next telemetry packet
        self.offset += self.__hdr.nbytes + self.packet_length + 1
        return tm_packet[0]

    def read(self, flname):
        """
        Read science or house-keeping telemetry packages

        Parameters
        ----------
        flname: str
           Name of the file with SPEXone ICU packages

        Returns
        -------
        tuple with data packages
        """
        packets = ()
        self.offset = 0

        # We should check that segmented packages consist of a sequence
        # with grouping flags {1, N * 0, 2}. I need to clean-up this code!
        while True:
            try:
                buff = self.__rd_tm_packet(flname)
            except (IOError, EOFError) as msg:
                print('[ERROR]: ', msg)
                return None

            # end of loop
            if buff is None:
                break

            # check if data is segmented
            if not packets:
                self.segmented = (self.grouping_flag != 3)

            if self.segmented == (self.grouping_flag == 3):
                print("[FATAL]: mixing segmented and unsegmented packages")

            packets += (buff,)

        return packets

    def group(self, packets):
        """
        Combine segmented data-packages

        Parameters
        ----------
        packets: tuple
           Tuple with science or house-keeping telemetry packages

        Returns
        -------
        tuple with unsegmented or house-keeping telemetry packages
        """
        if not self.segmented:
            return packets

        if ((packets[0]['primary_header']['sequence'] >> 14) & 0x0003) != 1:
            msg = '[WARNING]: removed {:d} segments of incomplete first image'
            ii = 0
            for buff in packets:
                flag = (buff['primary_header']['sequence'] >> 14) & 0x0003
                if flag == 1:
                    break
                ii += 1

            print(msg.format(ii))
            packets = packets[ii:]

        if ((packets[-1]['primary_header']['sequence'] >> 14) & 0x0003) != 2:
            msg = '[WARNING]: removed {:d} segments of incomplete last image'

            ii = len(packets)
            while ii > 0:
                ii -= 1
                flag = (packets[ii]['primary_header']['sequence'] >> 14) \
                       & 0x0003
                if flag == 2:
                    break

            print(msg.format(len(packets) - (ii + 1)))
            packets = packets[:ii+1]

        res = ()
        prev_grp_flag = 2
        for buff in packets:
            # get grouping flag of current package
            new_grp_flag = (buff['primary_header']['sequence'] >> 14) & 0x0003

            # handle segmented data
            if new_grp_flag == 1:
                # group_flag of previous package should be 2
                if prev_grp_flag != 2:
                    print('[WARNING]: first segment, but previous not closed?')

                data_buffer = buff['image_data']
                buff0 = self.__tm(buff['mps']['IMRLEN'] // 2)[0]
                buff0['primary_header'] = buff['primary_header']
                buff0['secondary_header'] = buff['secondary_header']
                buff0['mps'] = buff['mps']
                res += (buff0,)
            elif new_grp_flag in (0, 2):
                # group_flag of previous package should be 0 or 1
                if prev_grp_flag not in (0, 1):
                    print('[WARNING]: previous packet not closed?')

                data_buffer = np.concatenate((data_buffer, buff['image_data']))
                if new_grp_flag == 2:
                    if res[-1]['image_data'].size == data_buffer.size:
                        res[-1]['image_data'] = data_buffer
                    else:
                        print('[WARNING]: rejected incomplete data-buffer')

            # keep current group flag for next read
            prev_grp_flag = new_grp_flag

        return res
