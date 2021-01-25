"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python class to access SPEXone Science telemetry data (ApID 0x350)

Copyright (c) 2020-2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np


# - global parameters -------------------


# - class TMscience -------------------------
class TMscience:
    """
    Access/convert parameters of a SPEXone telemetry Science packet

    Methods
    -------
    get(key)
       Return (raw) SPEXone telemetry Science parameter.
    number_channels
       Return number of LVDS channels used.
    binning_table_id
       Return the binning table identifier (zero for full-frame images).
    lvds_clock
       Returns flag for LVDS clock: False: disabled & True: enabled.
    offset
       Returns digital offset including ADC offset [counts].
    pga_gain
       Returns PGA gain [Volt].
    exp_time
       Returns pixel exposure time [master clock periods].
    fot_time
       Returns frame overhead time [master clock periods].
    rot_time
       Returns image read-out time [master clock periods].
    frame_period
       Returns frame period [master clock periods].
    pll_control
       Returns raw PLL control parameters: pll_range, pll_out_fre, pll_div.
    exp_control
       Returns raw exposure time parameters: inte_sync, exp_dual, exp_ext.
    """
    def __init__(self, tm_science):
        """
        Initialize class TMscience

        Parameters
        tm_science :  ndarray
           SPEXone telemetry Science data
        """
        self.__tm = tm_science

    def get(self, key: str):
        """
        Return (raw) Science telemetry parameter
        """
        return self.__tm[key] if key in self.__tm.dtype.names else None

    @property
    def binning_table_id(self) -> int:
        """
        Return the binning table identifier (zero for full-frame images)

        Notes
        -----
        The CCSDS data may hold data collected with different MPS, but
        only when the size of the detector data does not change. Therefore,
        a mix of Science mode and Full-frame data should not occur.

        v126: Sometimes the MPS information is not updated for the first
              images. We try to fix this and warn the user.
        """
        full_frame = np.unique(self.__tm['REG_FULL_FRAME'])
        if len(full_frame) > 1:
            print('[WARNING]: value of REG_FULL_FRAME not unique')
            print(self.__tm['REG_FULL_FRAME'])
        full_frame = self.__tm['REG_FULL_FRAME'][-1]

        cmv_outputmode = np.unique(self.__tm['REG_CMV_OUTPUTMODE'])
        if len(cmv_outputmode) > 1:
            print('[WARNING]: value of REG_CMV_OUTPUTMODE not unique')
            print(self.__tm['REG_CMV_OUTPUTMODE'])
        cmv_outputmode = self.__tm['REG_CMV_OUTPUTMODE'][-1]

        if full_frame == 1:
            if cmv_outputmode != 3:
                raise KeyError('Diagnostic mode with REG_CMV_OUTPMODE != 3')
            return np.zeros(len(self.__tm), dtype='i1')

        if full_frame == 2:
            if cmv_outputmode != 1:
                raise KeyError('Science mode with REG_CMV_OUTPMODE != 1')
            bin_tbl_start = self.__tm['REG_BINNING_TABLE_START']
            indx0 = (self.__tm['REG_FULL_FRAME'] != 2).nonzero()[0]
            if indx0.size > 0:
                indx2 = (self.__tm['REG_FULL_FRAME'] == 2).nonzero()[0]
                bin_tbl_start[indx0] = bin_tbl_start[indx2[0]]
            res = 1 + (bin_tbl_start - 0x80000000) // 0x400000
            return res & 0xFF

        raise KeyError('REG_FULL_FRAME not equal to 1 or 2')

    @property
    def number_channels(self) -> int:
        """
        Return number of LVDS channels used
        """
        return 2 ** (4 - (self.__tm['DET_OUTMODE'] & 0x3))

    @property
    def lvds_clock(self) -> bool:
        """
        Returns flag for LVDS clock: False: disabled & True: enabled
        """
        return ((self.__tm['DET_PLLENA'] & 0x3) == 0
                and (self.__tm['DET_PLLBYP'] & 0x3) != 0
                and (self.__tm['DET_CHENA'] & 0x40000) != 0)

    @property
    def offset(self) -> int:
        """
        Returns digital offset including ADC offset
        """
        buff = self.__tm['DET_OFFSET'].astype('i4')
        if np.isscalar(buff):
            if buff >= 8192:
                buff -= 16384
        else:
            buff[buff >= 8192] -= 16384

        return buff + 70

    @property
    def pga_gain(self) -> float:
        """
        Returns PGA gain [Volt]
        """
        # need first bit of address 121
        reg_pgagainfactor = self.__tm['DET_BLACKCOL'] & 0x1

        reg_pgagain = self.__tm['DET_PGAGAIN']

        return (1 + 0.2 * reg_pgagain) * 2 ** reg_pgagainfactor

    @property
    def exp_time(self) -> float:
        """
        Returns pixel exposure time [master clock periods]
        """
        return 129 * (0.43 * self.__tm['DET_FOTLEN']
                      + self.__tm['DET_EXPTIME'])

    @property
    def fot_time(self) -> int:
        """
        Returns frame overhead time [master clock periods]
        """
        return 129 * (self.__tm['DET_FOTLEN']
                      + 2 * (16 // self.number_channels))

    @property
    def rot_time(self) -> int:
        """
        Returns image read-out time [master clock periods]
        """
        return 129 * (16 // self.number_channels) * self.__tm['DET_NUMLINES']

    @property
    def frame_period(self) -> float:
        """
        Returns frame period [master clock periods]
        """
        return 2.38e-7 + (self.__tm['REG_NCOADDFRAMES']
                          * (self.exp_time + self.fot_time + self.rot_time))

    @property
    def pll_control(self) -> tuple:
        """
        Returns raw PLL control parameters: pll_range, pll_out_fre, pll_div

        Notes
        -----
        PLL_range:    bits [7], valid values: 0 or 1
        PLL_out_fre:  bits [4:7], valid values:  0, 1, 2 or 5
        PLL_div:      bits [0:3], valid values 9 (10-bit) or 11 (12-bit)

        Other PLL registers are: PLL_enable, PLL_in_fre, PLL_bypass, PLL_load
        """
        pll_div = self.__tm['DET_PLLRATE'] & 0xF             # bit [0:4]
        pll_out_fre = (self.__tm['DET_PLLRATE'] >> 4) & 0x7  # bit [4:7]
        pll_range = (self.__tm['DET_PLLRATE'] >> 7)          # bit [7]

        return (pll_range, pll_out_fre, pll_div)

    @property
    def exp_control(self) -> tuple:
        """
        Returns raw exposure time parameters: inte_sync, exp_dual, exp_ext
        """
        inte_sync = (self.__tm['INTE_SYNC'] >> 2) & 0x1
        exp_dual = (self.__tm['INTE_SYNC'] >> 1) & 0x1
        exp_ext = self.__tm['INTE_SYNC'] & 0x1

        return (inte_sync, exp_dual, exp_ext)
