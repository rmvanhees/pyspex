#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Read Primary and Secondary CCSDS headers and obtain their parameters."""
from __future__ import annotations

__all__ = ['CCSDShdr']

from datetime import datetime, timedelta

import numpy as np


class CCSDShdr:
    """Read CCSDS telemetry packet structure which consists of the
    primary header: version, type, apid, grouping flag, sequence count
    and packet length, and the secondary header: tai_sec and sub_sec.

    Parameters
    ----------
    buffer :  buffer_like
       Array of type (unsigned) byte.
    offs :  int, default=0
       Start reading the buffer from this offset (in bytes)
    """

    def __init__(self, buffer: np.ndarray, offs: int = 0):
        """Initialise the class instance."""
        self.__dtype__ = np.dtype([('type', '>u2'),
                                   ('sequence', '>u2'),
                                   ('length', '>u2'),
                                   ('tai_sec', '>u4'),
                                   ('sub_sec', '>u2')])
        self.__hdr = np.frombuffer(buffer, count=1, offset=offs,
                                   dtype=self.__dtype__)[0]

    @property
    def dtype(self) -> np.dtype:
        """Data-type of the returned array."""
        return self.__dtype__

    @property
    def hdr(self) -> np.ndarray:
        """Structured array holding the CCSDS header."""
        return self.__hdr

    @property
    def version(self):
        """Return zero to indicate that this is a version 1 packet."""
        return (self.hdr['type'] >> 13) & 0x7

    @property
    def type(self):
        """Return zero to indicate that this is a telemetry packet."""
        return (self.hdr['type'] >> 12) & 0x1

    @property
    def apid(self):
        """Return ApID: an identifier for this telemetry packet."""
        return self.hdr['type'] & 0x7FF

    @property
    def grouping_flag(self):
        """Data packages can be segmented.

        Note
        ----
        This flag is encoded as::

         00 : continuation segement
         01 : first segment
         10 : last segment
         11 : unsegmented
        """
        return (self.hdr['sequence'] >> 14) & 0x3

    @property
    def sequence(self):
        """Return a counter which is incremented with each consecutive packet
        of a particular ApID. This value will rollover to 0 after 0x3FF is
        reached.
        """
        return self.hdr['sequence'] & 0x3FFF

    @property
    def packet_length(self):
        """Returns a value equal to the number of bytes in the Secondary
        header plus User Data minus 1.
        """
        return self.hdr['length']

    @property
    def tai_sec(self):
        """Seconds since 1958 (TAI)."""
        return self.hdr['tai_sec']

    @property
    def sub_sec(self):
        """Sub-seconds (1 / 2**16)."""
        return self.hdr['sub_sec']

    def tstamp(self, epoch: datetime) -> datetime:
        """Return time of the telemetry packtet.

        Parameters
        ----------
        epoch :  datetime
           Provide the UTC epoch of the time (thus corrected for leap seconds)
        """
        return (epoch + timedelta(
            seconds=int(self.tai_sec),
            microseconds=100 * int(self.sub_sec / 65536 * 10000)))
