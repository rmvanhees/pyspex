#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Defines the SPEXone telemetry packets as numpy data-types.

References
----------
* SPX1-TN-005 Telemetry and Telecommand Handbook, issue 14, date 15-Mar-2021
"""
__all__ = ['tmtc_dtype']

import numpy as np


def __tmtc_def(apid) -> list:
    """Returns SPEXone telemetry packet structure as a list of tuples.

    Parameters
    ----------
    apid : int
       SPEXone telemetry APID.
       Implemented APIDs: 0x350 (Science), 0x320 (NomHK) and 0x322 (DemHK).

    Returns
    -------
    list of tuples
       Definition of a numpy structured datatype.
    """
    if apid == 0x350:                           # *** Science TM ***
        return [                                # offs  start in packet
            ('ICUSWVER', '>u2'),                # 0     0x000c
            ('MPS_ID', 'u1'),                   # 2     0x000e
            ('MPS_VER', 'u1'),                  # 3     0x000f
            ('TS1_DEM_N_T', '>u4'),             # 4     0x0010
            ('TS2_HOUSING_N_T', '>u4'),         # 8     0x0014
            ('TS3_RADIATOR_N_T', '>u4'),        # 12    0x0018
            ('TS4_DEM_R_T', '>u4'),             # 16    0x001c
            ('TS5_HOUSING_R_T', '>u4'),         # 20    0x0020
            ('TS6_RADIATOR_R_T', '>u4'),        # 24    0x0024
            ('ICU_5V_T', '>i2'),                # 28    0x0028
            ('ICU_4V_T', '>i2'),                # 30    0x002a
            ('ICU_HG1_T', '>i2'),               # 32    0x002c
            ('ICU_HG2_T', '>i2'),               # 34    0x002e
            ('ICU_MID_T', '>i2'),               # 36    0x0030
            ('ICU_MCU_T', '>i2'),               # 38    0x0032
            ('ICU_DIGV_T', '>i2'),              # 40    0x0034
            ('ICU_4p0V_V', '>u2'),              # 42    0x0036
            ('ICU_3p3V_V', '>u2'),              # 44    0x0038
            ('ICU_1p2V_V', '>u2'),              # 46    0x003a
            ('ICU_4p0V_I', '>u2'),              # 48    0x003c
            ('ICU_3p3V_I', '>u2'),              # 50    0x003e
            ('ICU_1p2V_I', '>u2'),              # 52    0x0040
            ('ICU_5p0V_V', '>u2'),              # 54    0x0042
            ('ICU_5p0V_I', '>u2'),              # 56    0x0044
            ('DEM_V', '>u2'),                   # 58    0x0046
            ('DEM_I', '>u2'),                   # 60    0x0048
            ('LED1_ANODE_V', '>u4'),            # 62    0x004a
            ('LED1_CATH_V', '>u4'),             # 66    0x004e
            ('LED1_I', '>u4'),                  # 70    0x0052
            ('LED2_ANODE_V', '>u4'),            # 74    0x0056
            ('LED2_CATH_V', '>u4'),             # 78    0x005a
            ('LED2_I', '>u4'),                  # 82    0x005e
            ('ADC1_VCC', '>u4'),                # 86    0x0062
            ('ADC1_REF', '>u4'),                # 90    0x0066
            ('ADC1_T', '>u4'),                  # 94    0x006a
            ('ADC2_VCC', '>u4'),                # 98    0x006e
            ('ADC2_REF', '>u4'),                # 102   0x0072
            ('ADC2_T', '>u4'),                  # 106   0x0076
            ('REG_FW_VERSION', 'u1'),           # 110   0x007a
            ('REG_NCOADDFRAMES', 'u1'),         # 111   0x007b
            ('REG_IGEN_SELECT', 'u1'),          # 112   0x007c
            ('REG_FULL_FRAME', 'u1'),           # 113   0x007d
            ('REG_BINNING_TABLE_START', '>u4'),  # 114  0x007e
            ('REG_CMV_OUTPUTMODE', 'u1'),       # 118   0x0082
            ('dummy_01', 'u1'),                 # 119   0x0083
            ('REG_COADD_BUF_START', '>u4'),     # 120   0x0084
            ('REG_COADD_RESA_START', '>u4'),    # 124   0x0088
            ('REG_COADD_RESB_START', '>u4'),    # 128   0x008c
            ('REG_FRAME_BUFA_START', '>u4'),    # 132   0x0090
            ('REG_FRAME_BUFB_START', '>u4'),    # 136   0x0094
            ('REG_LINE_ENABLE_START', '>u4'),   # 140   0x0098
            ('DET_REG000', 'u1'),               # 144   0x009c
            ('dummy_02', 'u1'),                 # 145   0x009d
            ('DET_NUMLINES', '>u2'),            # 146   0x009e
            ('DET_START1', '>u2'),              # 148   0x00a0
            ('DET_START2', '>u2'),              # 150   0x00a2
            ('DET_START3', '>u2'),              # 152   0x00a4
            ('DET_START4', '>u2'),              # 154   0x00a6
            ('DET_START5', '>u2'),              # 156   0x00a8
            ('DET_START6', '>u2'),              # 158   0x00aa
            ('DET_START7', '>u2'),              # 160   0x00ac
            ('DET_START8', '>u2'),              # 152   0x00ae
            ('DET_NUMLINES1', '>u2'),           # 164   0x00b0
            ('DET_NUMLINES2', '>u2'),           # 166   0x00b2
            ('DET_NUMLINES3', '>u2'),           # 168   0x00b4
            ('DET_NUMLINES4', '>u2'),           # 170   0x00b6
            ('DET_NUMLINES5', '>u2'),           # 172   0x00b8
            ('DET_NUMLINES6', '>u2'),           # 174   0x00ba
            ('DET_NUMLINES7', '>u2'),           # 176   0x00bc
            ('DET_NUMLINES8', '>u2'),           # 178   0x00be
            ('DET_SUBS', '>u2'),                # 180   0x00c0
            ('DET_SUBA', '>u2'),                # 182   0x00c2
            ('DET_MONO', 'u1'),                 # 184   0x00c4
            ('DET_IMFLIP', 'u1'),               # 185   0x00c5
            ('DET_EXPCNTR', 'u1'),              # 186   0x00c6
            ('DET_ILVDS', 'u1'),                # 187   0x00c7
            ('DET_EXPTIME', '>u4'),             # 188   0x00c8
            ('DET_EXPSTEP', '>u4'),             # 192   0x00cc
            ('DET_KP1', '>u4'),                 # 196   0x00d0
            ('DET_KP2', '>u4'),                 # 200   0x00d4
            ('DET_NOFSLOPES', 'u1'),            # 204   0x00D8
            ('DET_EXPSEQ', 'u1'),               # 205   0x00d9
            ('DET_EXPTIME2', '>u4'),            # 206   0x00da
            ('DET_EXPSTEP2', '>u4'),            # 210   0x00de
            ('DET_REG062', 'u1'),               # 214   0x00e2
            ('DET_REG063', 'u1'),               # 215   0x00e3
            ('DET_REG064', 'u1'),               # 216   0x00e4
            ('DET_REG065', 'u1'),               # 217   0x00e5
            ('DET_REG066', 'u1'),               # 218   0x00e6
            ('DET_REG067', 'u1'),               # 219   0x00e7
            ('DET_REG068', 'u1'),               # 220   0x00e8
            ('DET_EXP2_SEQ', 'u1'),             # 221   0x00e9
            ('DET_NOFFRAMES', '>u2'),           # 222   0x00ea
            ('DET_OUTMODE', 'u1'),              # 224   0x00ec
            ('DET_FOTLEN', 'u1'),               # 225   0x00ed
            ('DET_ILVDSRCVR', 'u1'),            # 226   0x00ee
            ('DET_REG075', 'u1'),               # 227   0x00ef
            ('DET_REG076', 'u1'),               # 228   0x00f0
            ('DET_CALIB', 'u1'),                # 229   0x00f1
            ('DET_TRAINPTRN', '>u2'),           # 230   0x00f2
            ('DET_CHENA', '>u4'),               # 232   0x00f4
            ('DET_ICOL', 'u1'),                 # 236   0x00F8
            ('DET_ICOLPR', 'u1'),               # 237   0x00f9
            ('DET_IADC', 'u1'),                 # 238   0x00fa
            ('DET_IAMP', 'u1'),                 # 239   0x00fb
            ('DET_VTFL1', 'u1'),                # 240   0x00fc
            ('DET_VTFL2', 'u1'),                # 241   0x00fd
            ('DET_VTFL3', 'u1'),                # 242   0x00fe
            ('DET_VRSTL', 'u1'),                # 243   0x00ff
            ('DET_REG092', 'u1'),               # 244   0x0100
            ('DET_REG093', 'u1'),               # 245   0x0101
            ('DET_VPRECH', 'u1'),               # 246   0x0102
            ('DET_VREF', 'u1'),                 # 247   0x0103
            ('DET_REG096', 'u1'),               # 248   0x0104
            ('DET_REG097', 'u1'),               # 249   0x0105
            ('DET_VRAMP1', 'u1'),               # 250   0x0106
            ('DET_VRAMP2', 'u1'),               # 251   0x0107
            ('DET_OFFSET', '>u2'),              # 252   0x0108
            ('DET_PGAGAIN', 'u1'),              # 254   0x010a
            ('DET_ADCGAIN', 'u1'),              # 255   0x010b
            ('DET_REG104', 'u1'),               # 256   0x010c
            ('DET_REG105', 'u1'),               # 257   0x010d
            ('DET_REG106', 'u1'),               # 258   0x010e
            ('DET_REG107', 'u1'),               # 259   0x010f
            ('DET_TDIG1', 'u1'),                # 260   0x0110
            ('DET_TDIG2', 'u1'),                # 261   0x0111
            ('DET_REG110', 'u1'),               # 262   0x0112
            ('DET_BITMODE', 'u1'),              # 263   0x0113
            ('DET_ADCRES', 'u1'),               # 264   0x0114
            ('DET_PLLENA', 'u1'),               # 265   0x0115
            ('DET_PLLINFRE', 'u1'),             # 266   0x0116
            ('DET_PLLBYP', 'u1'),               # 267   0x0117
            ('DET_PLLRATE', 'u1'),              # 268   0x0118
            ('DET_PLLLOAD', 'u1'),              # 269   0x0119
            ('DET_DETDUM', 'u1'),               # 270   0x011a
            ('DET_REG119', 'u1'),               # 271   0x011b
            ('DET_REG120', 'u1'),               # 272   0x011c
            ('DET_BLACKCOL', 'u1'),             # 273   0x011d
            ('DET_REG122', 'u1'),               # 274   0x011e
            ('DET_VBLACKSUN', 'u1'),            # 275   0x011f
            ('DET_REG124', 'u1'),               # 276   0x0120
            ('DET_REG125', 'u1'),               # 277   0x0121
            ('DET_T', '>u2'),                   # 278   0x0122
            ('FTI', '>u2'),                     # 280   0x0124  (100 usec)
            ('IMDMODE', 'u1'),                  # 282   0x0126
            ('dummy_03', 'u1'),                 # 283   0x0127
            ('IMRLEN', '>u4')                   # 284   0x0128
        ]                                       # 288

    if apid == 0x320:                           # ***** NomHK *****
        return [                                # offs  start in packet
            ('SEQCNT', '>u2'),                  # 0     0x000c
            ('TCPKTID', '>u2'),                 # 2     0x000e
            ('TCPKTSEQCTRL', '>u2'),            # 4     0x0010
            ('TCREJCODE', 'u1'),                # 6     0x0012
            ('TCFAILCODE', 'u1'),               # 7     0x0013
            ('TCREJPKTID', '>u2'),              # 8     0x0014
            ('TCFAILPKTID', '>u2'),             # 10    0x0016
            ('TCACCCNT', '>u2'),                # 12    0x0018
            ('TCREJCNT', '>u2'),                # 14    0x001a
            ('TCEXECCNT', '>u2'),               # 16    0x001c
            ('TCFAILCNT', '>u2'),               # 18    0x001e
            ('ICUSWVER', '>u2'),                # 20    0x0020
            ('SYSSTATE', '>u4'),                # 22    0x0022
            ('ICUMODE', 'u1'),                  # 26    0x0026
            ('EXTPPSSTAT', 'u1'),               # 27    0x0027
            ('TIMEMSGSTAT', 'u1'),              # 28    0x0028
            ('OBTSYNCSTAT', 'u1'),              # 29    0x0029
            ('MPS_ID', 'u1'),                   # 30    0x002a
            ('MPS_VER', 'u1'),                  # 31    0x002b
            ('EVNTCNT_DEBUG', 'u1'),            # 32    0x002c
            ('EVNTCNT_PROG', 'u1'),             # 33    0x002d
            ('EVNTCNT_WARN', 'u1'),             # 34    0x002e
            ('EVNTCNT_ERR', 'u1'),              # 35    0x002f
            ('EVNTCNT_FATAL', 'u1'),            # 36    0x0030
            ('BOOTSTATEPREV', 'u1'),            # 37    0x0031
            ('BOOTCNTGOOD_IM0', '>u4'),         # 38    0x0032
            ('BOOTCNTGOOD_IM1', '>u4'),         # 42    0x0036
            ('BOOTCNTGOOD_IM2', '>u4'),         # 46    0x003a
            ('BOOTCNTGOOD_IM3', '>u4'),         # 50    0x003e
            ('BOOTATTEMPTS_CURRIM', 'u1'),      # 54    0x0042
            ('dummy_01', 'u1'),                 # 55    0x0043
            ('SWIMG_LOADED', 'u1'),             # 56    0x0044
            ('SWIMG_DEFAULT', 'u1'),            # 57    0x0045
            ('SWIMG_NXTBOOT', 'u1'),            # 58    0x0046
            ('WRITEPROT', 'u1'),                # 59    0x0047
            ('BOOTCAUSE', 'u1'),                # 60    0x0048
            ('TCVER_STAT', 'u1'),               # 61    0x0049
            ('SPW_REG_A', '>u4'),               # 62    0x004a
            ('SPW_REG_B', '>u4'),               # 66    0x004e
            ('LAST_CRC', '>u4'),                # 70    0x0052
            ('SCITM_PKTINTVL', '>u2'),          # 74    0x0056
            ('SCITM_BUFFREE', '>u4'),           # 76    0x0058
            ('SWEXECTIMEWC', '>u8'),            # 80    0x005c
            ('ERRCNT_PLACEHOLDER_03', '>u2'),   # 88    0x0064
            # ('FillerByte', 'u1')
            ('TS1_DEM_N_T', '>u4'),             # 90    0x0066
            # ('FillerByte', 'u1')
            ('TS2_HOUSING_N_T', '>u4'),         # 94    0x006a
            # ('FillerByte', 'u1')
            ('TS3_RADIATOR_N_T', '>u4'),        # 98    0x006e
            # ('FillerByte', 'u1')
            ('TS4_DEM_R_T', '>u4'),             # 102   0x0072
            # ('FillerByte', 'u1')
            ('TS5_HOUSING_R_T', '>u4'),         # 106   0x0076
            # ('FillerByte', 'u1')
            ('TS6_RADIATOR_R_T', '>u4'),        # 110   0x007a
            ('ICU_5V_T', '>u2'),                # 114   0x007e
            ('ICU_4V_T', '>u2'),                # 116   0x0080
            ('ICU_HG1_T', '>u2'),               # 118   0x0082
            ('ICU_HG2_T', '>u2'),               # 120   0x0084
            ('ICU_MID_T', '>u2'),               # 122   0x0086
            ('ICU_MCU_T', '>u2'),               # 124   0x0088
            ('ICU_DIGV_T', '>u2'),              # 126   0x008a
            ('ICU_4P0V_V', '>u2'),              # 128   0x008c
            ('ICU_3P3V_V', '>u2'),              # 130   0x008e
            ('ICU_1P2V_V', '>u2'),              # 132   0x0090
            ('ICU_4P0V_I', '>u2'),              # 134   0x0092
            ('ICU_3P3V_I', '>u2'),              # 136   0x0094
            ('ICU_1P2V_I', '>u2'),              # 138   0x0096
            ('DEM_STATUS', 'u1'),               # 140   0x0098
            ('dummy_02', 'u1'),                 # 141   0x0099
            ('ICU_5P0V_V', '>u2'),              # 142   0x009a
            ('ICU_5P0V_I', '>u2'),              # 144   0x009c
            ('DEMSPWSTAT', 'u1'),               # 146   0x009e
            ('DEMRESETCNT', 'u1'),              # 147   0x009f
            ('HTRGRP1_V', '>u2'),               # 148   0x00a0
            ('HTRGRP2_V', '>u2'),               # 150   0x00a2
            ('HTR1_I', '>u2'),                  # 152   0x00a4
            ('HTR2_I', '>u2'),                  # 154   0x00a6
            ('HTR3_I', '>u2'),                  # 156   0x00a8
            ('HTR4_I', '>u2'),                  # 158   0x00aa
            ('HTR1_CALCPVAL', '>f4'),           # 160   0x00ac
            ('HTR2_CALCPVAL', '>f4'),           # 164   0x00b0
            ('HTR3_CALCPVAL', '>f4'),           # 168   0x00b4
            ('HTR4_CALCPVAL', '>f4'),           # 172   0x00b8
            ('HTR1_CALCIVAL', '>f4'),           # 176   0x00bc
            ('HTR2_CALCIVAL', '>f4'),           # 180   0x00c0
            ('HTR3_CALCIVAL', '>f4'),           # 184   0x00c4
            ('HTR4_CALCIVAL', '>f4'),           # 188   0x00c8
            ('HTR1_DUTYCYCL', '>u2'),           # 192   0x00cc
            ('HTR2_DUTYCYCL', '>u2'),           # 194   0x00ce
            ('HTR3_DUTYCYCL', '>u2'),           # 196   0x00d0
            ('HTR4_DUTYCYCL', '>u2'),           # 198   0x00d2
            ('LED1_ENADIS', 'u1'),              # 200   0x00d4
            ('LED2_ENADIS', 'u1'),              # 201   0x00d5
            # ('FillerByte', 'u1')
            ('LED1_ANODE_V', '>u4'),            # 202   0x00d6
            # ('FillerByte', 'u1')
            ('LED1_CATH_V', '>u4'),             # 206   0x00da
            # ('FillerByte', 'u1')
            ('LED1_I', '>u4'),                  # 210   0x00de
            # ('FillerByte', 'u1')
            ('LED2_ANODE_V', '>u4'),            # 214   0x00e2
            # ('FillerByte', 'u1')
            ('LED2_CATH_V', '>u4'),             # 218   0x00e6
            # ('FillerByte', 'u1')
            ('LED2_I', '>u4'),                  # 222   0x00ea
            # ('FillerByte', 'u1')
            ('ADC1_VCC', '>u4'),                # 226   0x00ee
            # ('FillerByte', 'u1')
            ('ADC1_GAIN', '>u4'),               # 230   0x00f2
            # ('FillerByte', 'u1')
            ('ADC1_REF', '>u4'),                # 234   0x00f6
            # ('FillerByte', 'u1')
            ('ADC1_T', '>u4'),                  # 238   0x00fa
            # ('FillerByte', 'u1')
            ('ADC1_OFFSET', '>u4'),             # 242   0x00fe
            # ('FillerByte', 'u1')
            ('ADC2_VCC', '>u4'),                # 246   0x0102
            # ('FillerByte', 'u1')
            ('ADC2_GAIN', '>u4'),               # 250   0x0106
            # ('FillerByte', 'u1')
            ('ADC2_REF', '>u4'),                # 254   0x010a
            # ('FillerByte', 'u1')
            ('ADC2_T', '>u4'),                  # 258   0x010e
            # ('FillerByte', 'u1')
            ('ADC2_OFFSET', '>u4'),             # 262   0x0112
            ('DEM_V', '>u2'),                   # 266   0x0116
            ('DEM_I', '>u2'),                   # 268   0x0118
            ('REG_FW_VERSION', 'u1'),           # 270   0x011a
            ('dummy_03', 'u1'),                 # 271   0x011b
            ('DET_T', '>u2'),                   # 272   0x011c
            ('REG_SPW_ERROR', 'u1'),            # 274   0x011e
            ('REG_CMV_OUTOFSYNC', 'u1'),        # 275   0x011f
            ('REG_OCD_ACTUAL', 'u1'),           # 276   0x0120
            ('REG_OCD_STICKY', 'u1'),           # 277   0x0121
            ('REG_PWR_SENS', 'u1'),             # 278   0x0122
            ('REG_FLASH_STATUS', 'u1'),         # 279   0x0123
            ('REG_FLASH_EDAC_BLOCK', '>u2'),    # 280   0x0124
            ('SW_MAIN_LOOP_COUNT', '>u4')       # 282   0x0126
        ]                                       # 286

    if apid == 0x322:                           # ***** DemHK *****
        return [                                # offs  start in packet
            ('REG_STATUS', 'u1'),               # 0     0x000c
            ('REG_NCOADDFRAMES', 'u1'),         # 1     0x000d
            ('REG_IGEN_SELECT', 'u1'),          # 2     0x000e
            ('REG_FIFO_STATUS', 'u1'),          # 3     0x000f
            ('REG_SPW_TURBO', 'u1'),            # 4     0x0010
            ('REG_IGEN_MODE', 'u1'),            # 5     0x0011
            ('REG_IGEN_VALUE', '>u2'),          # 6     0x0012
            ('REG_FULL_FRAME', 'u1'),           # 8     0x0014
            ('dummy_01', 'u1'),                 # 9     0x0015
            ('REG_FLASH_ERASE', '<u4'),         # 10    0x0016 LE
            ('REG_BINNING_TABLE_START', '>u4'),  # 14   0x001a
            ('REG_CMV_OUTPUTMODE', 'u1'),       # 18    0x001e
            ('REG_DETECT_ENABLE', 'u1'),        # 19    0x001f
            ('REG_POWERUP_DELAY', '<u4'),       # 20    0x0020 LE
            # ('FillerByte', 'u1')
            ('REG_LU_THRESHOLD', '<u2'),        # 24    0x0024 LE
            ('REG_COADD_BUF_START', '<u4'),     # 26    0x0026 LE
            ('REG_COADD_RESA_START', '<u4'),    # 30    0x002a LE
            ('REG_COADD_RESB_START', '<u4'),    # 34    0x002e LE
            ('REG_FRAME_BUFA_START', '<u4'),    # 38    0x0032 LE
            ('REG_FRAME_BUFB_START', '<u4'),    # 42    0x0036 LE
            ('dummy_02', 'u1'),                 # 46    0x003a
            ('REG_FLASH_PAGE_SPR_BYTE', 'u1'),  # 47    0x003b
            ('REG_LINE_ENABLE_START', '<u4'),   # 48    0x003c LE
            ('DET_REG000', 'u1'),               # 52    0x0040
            ('dummy_03', 'u1'),                 # 53    0x0041
            ('DET_NUMLINES', '>u2'),            # 54    0x0042
            ('DET_START1', '>u2'),              # 56    0x0044
            ('DET_START2', '>u2'),              # 58    0x0046
            ('DET_START3', '>u2'),              # 60    0x0048
            ('DET_START4', '>u2'),              # 62    0x004a
            ('DET_START5', '>u2'),              # 64    0x004c
            ('DET_START6', '>u2'),              # 66    0x004e
            ('DET_START7', '>u2'),              # 68    0x0050
            ('DET_START8', '>u2'),              # 70    0x0052
            ('DET_NUMLINES1', '>u2'),           # 72    0x0054
            ('DET_NUMLINES2', '>u2'),           # 74    0x0056
            ('DET_NUMLINES3', '>u2'),           # 76    0x0058
            ('DET_NUMLINES4', '>u2'),           # 78    0x005a
            ('DET_NUMLINES5', '>u2'),           # 80    0x005c
            ('DET_NUMLINES6', '>u2'),           # 82    0x005e
            ('DET_NUMLINES7', '>u2'),           # 84    0x0060
            ('DET_NUMLINES8', '>u2'),           # 86    0x0062
            ('DET_SUBS', '>u2'),                # 88    0x0064
            ('DET_SUBA', '>u2'),                # 90    0x0066
            ('DET_MONO', 'u1'),                 # 92    0x0068
            ('DET_IMFLIP', 'u1'),               # 93    0x0069
            ('DET_EXPCNTR', 'u1'),              # 94    0x006a
            ('dummy_04', 'u1'),                 # 95    0x006b
            ('DET_EXPTIME', '>u4'),             # 96    0x006c
            # ('FillerByte', 'u1')
            ('DET_EXPSTEP', '>u4'),             # 100   0x0070
            # ('FillerByte', 'u1')
            ('DET_KP1', '>u4'),                 # 104   0x0074
            # ('FillerByte', 'u1')
            ('DET_KP2', '>u4'),                 # 108   0x0078
            # ('FillerByte', 'u1')
            ('DET_NOFSLOPES', 'u1'),            # 112   0x007c
            ('DET_EXPSEQ', 'u1'),               # 113   0x007d
            ('DET_EXPTIME2', '>u4'),            # 114   0x007e
            # ('FillerByte', 'u1')
            ('DET_EXPSTEP2', '>u4'),            # 118   0x0082
            # ('FillerByte', 'u1')
            ('DET_REG062', 'u1'),               # 122   0x0086
            ('DET_REG063', 'u1'),               # 123   0x0087
            ('DET_REG064', 'u1'),               # 124   0x0088
            ('DET_REG065', 'u1'),               # 125   0x0089
            ('DET_REG066', 'u1'),               # 126   0x008a
            ('DET_REG067', 'u1'),               # 127   0x008b
            ('DET_REG068', 'u1'),               # 128   0x008c
            ('DET_EXP2_SEQ', 'u1'),             # 129   0x008d
            ('DET_NOFFRAMES', '>u2'),           # 130   0x008e
            ('DET_OUTMODE', 'u1'),              # 132   0x0090
            ('DET_FOTLEN', 'u1'),               # 133   0x0091
            ('DET_ILVDSRCVR', 'u1'),            # 134   0x0092
            ('DET_REG075', 'u1'),               # 135   0x0093
            ('DET_REG076', 'u1'),               # 136   0x0094
            ('DET_CALIB', 'u1'),                # 137   0x0095
            ('DET_TRAINPTRN', '>u2'),           # 138   0x0096
            ('DET_CHENA', '>u4'),               # 140   0x0098
            # ('FillerByte', 'u1')
            ('DET_ILVDS', 'u1'),                # 144   0x009c
            ('DET_ICOL', 'u1'),                 # 145   0x009d
            ('DET_ICOLPR', 'u1'),               # 146   0x009e
            ('DET_IADC', 'u1'),                 # 147   0x009f
            ('DET_IAMP', 'u1'),                 # 148   0x00a0
            ('DET_VTFL1', 'u1'),                # 149   0x00a1
            ('DET_VTFL2', 'u1'),                # 150   0x00a2
            ('DET_VTFL3', 'u1'),                # 151   0x00a3
            ('DET_VRSTL', 'u1'),                # 152   0x00a4
            ('DET_REG092', 'u1'),               # 153   0x00a5
            ('DET_REG093', 'u1'),               # 154   0x00a6
            ('DET_VPRECH', 'u1'),               # 155   0x00a7
            ('DET_VREF', 'u1'),                 # 156   0x00a8
            ('DET_REG096', 'u1'),               # 157   0x00a9
            ('DET_REG097', 'u1'),               # 158   0x00aa
            ('DET_VRAMP1', 'u1'),               # 159   0x00ab
            ('DET_VRAMP2', 'u1'),               # 160   0x00ac
            ('dummy_05', 'u1'),                 # 161   0x00ad
            ('DET_OFFSET', '>u2'),              # 162   0x00ae
            ('DET_PGAGAIN', 'u1'),              # 164   0x00b0
            ('DET_ADCGAIN', 'u1'),              # 165   0x00b1
            ('DET_REG104', 'u1'),               # 166   0x00b2
            ('DET_REG105', 'u1'),               # 167   0x00b3
            ('DET_REG106', 'u1'),               # 168   0x00b4
            ('DET_REG107', 'u1'),               # 169   0x00b5
            ('DET_TDIG1', 'u1'),                # 170   0x00b6
            ('DET_TDIG2', 'u1'),                # 171   0x00b7
            ('DET_REG110', 'u1'),               # 172   0x00b8
            ('DET_BITMODE', 'u1'),              # 173   0x00b9
            ('DET_ADCRES', 'u1'),               # 174   0x00ba
            ('DET_PLLENA', 'u1'),               # 175   0x00bb
            ('DET_PLLINFRE', 'u1'),             # 176   0x00bc
            ('DET_PLLBYP', 'u1'),               # 177   0x00bd
            ('DET_PLLRATE', 'u1'),              # 178   0x00be
            ('DET_PLLLoad', 'u1'),              # 179   0x00bf
            ('DET_DETDum', 'u1'),               # 180   0x00c0
            ('DET_REG119', 'u1'),               # 181   0x00c1
            ('DET_REG120', 'u1'),               # 182   0x00c2
            ('DET_BLACKCOL', 'u1'),             # 183   0x00c3
            ('DET_REG122', 'u1'),               # 184   0x00c4
            ('DET_VBLACKSUN', 'u1'),            # 185   0x00c5
            ('DET_REG124', 'u1'),               # 186   0x00c6
            ('DET_REG125', 'u1')                # 187   0x00c7
        ]                                       # 188

    raise ValueError('Telemetry APID not implemented')


def tmtc_dtype(apid):
    """Obtain SPEXone telemetry packet definition.

    Parameters
    ----------
    apid : int
       SPEXone telemetry APID.
       Implemented APIDs: 0x350 (Science), 0x320 (NomHK) and 0x322 (DemHK).

    Returns
    -------
    numpy.dtype
       Definition of Spexone telemetry packet.

    Examples
    --------

    >>> import numpy as np
    >>> from pyspex.lib.tmtc_def import tmtc_dtype
    >>> mps_dtype = tmtc_dtype(0x350)

    """
    return np.dtype(__tmtc_def(apid))
