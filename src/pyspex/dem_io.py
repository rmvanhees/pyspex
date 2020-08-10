"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation to read SPEXone DEM output

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np

from .lib.tmtc_def import tmtc_def


# - global parameters ------------------------------


# - local functions --------------------------------
def det_dtype():
    """
    Returns numpy dtype with the registers of the SPEXone CMV4000 detector
    """
    return np.dtype([
        ('UNUSED_000', 'u1'),
        ('NUMBER_LINES', 'u1', (2)),
        ('START1', 'u1', (2)),
        ('START2', 'u1', (2)),
        ('START3', 'u1', (2)),
        ('START4', 'u1', (2)),
        ('START5', 'u1', (2)),
        ('START6', 'u1', (2)),
        ('START7', 'u1', (2)),
        ('START8', 'u1', (2)),
        ('NUMBER_LINES1', 'u1', (2)),
        ('NUMBER_LINES2', 'u1', (2)),
        ('NUMBER_LINES3', 'u1', (2)),
        ('NUMBER_LINES4', 'u1', (2)),
        ('NUMBER_LINES5', 'u1', (2)),
        ('NUMBER_LINES6', 'u1', (2)),
        ('NUMBER_LINES7', 'u1', (2)),
        ('NUMBER_LINES8', 'u1', (2)),
        ('SUB_S', 'u1', (2)),
        ('SUB_A', 'u1', (2)),
        ('MONO', 'u1'),                # 1 bits
        ('IMAGE_FLIPPING', 'u1'),      # 2 bits
        ('INTE_SYNC', 'u1'),           # 3 bits: Inte_sync, Exp_dual, Exp_ext
        ('EXP_TIME', 'u1', (3)),
        ('EXP_STEP', 'u1', (3)),
        ('EXP_KP1', 'u1', (3)),
        ('EXP_KP2', 'u1', (3)),
        ('NR_SLOPES', 'u1'),           # 2 bits
        ('EXP_SEQ', 'u1'),
        ('EXP_TIME2', 'u1', (3)),
        ('EXP_STEP2', 'u1', (3)),
        ('UNUSED_062', 'u1'),
        ('UNUSED_063', 'u1'),
        ('UNUSED_064', 'u1'),
        ('UNUSED_065', 'u1'),
        ('UNUSED_066', 'u1'),
        ('UNUSED_067', 'u1'),
        ('UNUSED_068', 'u1'),
        ('EXP2_SEQ', 'u1'),
        ('NUMBER_FRAMES', 'u1', (2)),
        ('OUTPUT_MODE', 'u1'),         # 2 bits
        ('FOT_LENGTH', 'u1'),
        ('I_LVDS_REC', 'u1'),          # 4 bits
        ('UNUSED_075', 'u1'),
        ('UNUSED_076', 'u1'),
        ('COL_CALIB', 'u1'),           # 2 bits: Col_calib, ADC_calib
        ('TRAINING_PATTERN', 'u1', (2)), # 12 bits
        ('CHANNEL_EN', 'u1', (3)),       # 19 bits
        ('I_LVDS', 'u1'),              # 4 bits
        ('I_COL', 'u1'),               # 4 bits
        ('I_COL_PRECH', 'u1'),         # 4 bits
        ('I_ADC', 'u1'),               # 4 bits
        ('I_AMP', 'u1'),               # 4 bits
        ('VTF_L1', 'u1'),              # 7 bits
        ('VLOW2', 'u1'),               # 7 bits
        ('VLOW3', 'u1'),               # 7 bits
        ('VRES_LOW', 'u1'),            # 7 bits
        ('UNUSED_092', 'u1'),
        ('UNUSED_093', 'u1'),
        ('V_PRECH', 'u1'),             # 7 bits
        ('V_REF', 'u1'),               # 7 bits
        ('UNUSED_096', 'u1'),
        ('UNUSED_097', 'u1'),
        ('VRAMP1', 'u1'),              # 7 bits
        ('VRAMP2', 'u1'),              # 7 bits
        ('OFFSET', 'u1', (2)),         # 14 bits
        ('PGA_GAIN', 'u1'),            # 2 bits
        ('ADC_GAIN', 'u1'),
        ('UNUSED_104', 'u1'),
        ('UNUSED_105', 'u1'),
        ('UNUSED_106', 'u1'),
        ('UNUSED_107', 'u1'),
        ('T_DIG1', 'u1'),              # 4 bits
        ('T_DIG2', 'u1'),              # 4 bits
        ('UNUSED_110', 'u1'),
        ('BIT_MODE', 'u1'),            # 1 bits
        ('ADC_RESOLUTION', 'u1'),      # 2 bits
        ('PLL_ENABLE', 'u1'),          # 1 bits
        ('PLL_IN_FRE', 'u1'),          # 2 bits
        ('PLL_BYPASS', 'u1'),          # 1 bits
        ('PLL_RANGE', 'u1'), # 8 bits: PLL_range(1), PLL_out_fre(3), PLL_div(4)
        ('PLL_LOAD', 'u1'),
        ('DUMMY', 'u1'),
        ('UNUSED_119', 'u1'),
        ('UNUSED_120', 'u1'),
        ('BLACK_COL_EN', 'u1'),        # 2 bits: Black_col_en, PGA_gain
        ('UNUSED_122', 'u1'),
        ('V_BLACKSUN', 'u1'),          # 6 bits
        ('UNUSED_124', 'u1'),
        ('UNUSED_125', 'u1'),
        ('TEMP', 'u1', (2))
    ])


# - class DEMio -------------------------
class DEMio:
    """
    This class can be used to read SPEXone DEM output
    """
    def __init__(self):
        self.__hdr = None

    @property
    def hdr(self):
        """
        Return DEM header as numpy compound array
        """
        if self.__hdr is None:
            return None

        return self.__hdr[0]

    def read_hdr(self, hdr_file: str):
        """
        Read DEM header data into MPS record

        Parameters
        ----------
        hdr_file : str
           Name of the (ASCII) header file

        Returns
        -------
        numpy array
        """
        def convert_val(key):
            """
            Convert byte array to integer
            """
            val = 0
            for ii, bval in enumerate(self.__hdr[0][key]):
                val += bval << (ii * 8)

            return val

        self.__hdr = np.zeros((1,), dtype=det_dtype())
        with open(hdr_file, 'r') as fp:
            for line in fp:
                columns = line[:-1].split(',')
                if columns[0] == 'Reg':
                    continue

                # Fix possible errors in Name
                name = columns[2].replace(' [', '[')
                name = name.replace('_ ', '_').replace(' ', '_')

                value = int(columns[-1])
                indx = -1
                if columns[2].endswith(':0]') \
                   or columns[2].endswith('[0]') \
                   or columns[2].endswith('[2]'):
                    name = name.split('[')[0]
                    indx = 0
                elif columns[2].endswith(':8]'):
                    name = name.split('[')[0]
                    indx = 1
                elif columns[2].endswith(':16]'):
                    name = name.split('[')[0]
                    indx = 2
                elif name == 'Unused':
                    if columns[0] == '86':
                        continue
                    name = 'Unused_{:03d}'.format(int(columns[0]))

                key = name.upper()
                if isinstance(self.__hdr[0][key], np.ndarray):
                    self.__hdr[0][key][indx] = value
                else:
                    self.__hdr[0][key] = value

        # convert original detector parameter values to MPS parameters
        convert_det_params = {
            'DET_NUMLINES': convert_val('NUMBER_LINES'),
            'DET_START1': convert_val('START1'),
            'DET_START2': convert_val('START2'),
            'DET_START3': convert_val('START3'),
            'DET_START4': convert_val('START4'),
            'DET_START5': convert_val('START5'),
            'DET_START6': convert_val('START6'),
            'DET_START7': convert_val('START7'),
            'DET_START8': convert_val('START8'),
            'DET_NUMLINES1': convert_val('NUMBER_LINES1'),
            'DET_NUMLINES2': convert_val('NUMBER_LINES2'),
            'DET_NUMLINES3': convert_val('NUMBER_LINES3'),
            'DET_NUMLINES4': convert_val('NUMBER_LINES4'),
            'DET_NUMLINES5': convert_val('NUMBER_LINES5'),
            'DET_NUMLINES6': convert_val('NUMBER_LINES6'),
            'DET_NUMLINES7': convert_val('NUMBER_LINES7'),
            'DET_NUMLINES8': convert_val('NUMBER_LINES8'),
            'DET_SUBS': convert_val('SUB_S'),
            'DET_SUBA': convert_val('SUB_A'),
            'DET_MONO': self.__hdr[0]['MONO'],
            'DET_IMFLIP': self.__hdr[0]['IMAGE_FLIPPING'],
            'DET_EXPCNTR': self.__hdr[0]['INTE_SYNC'],
            'DET_EXPTIME': convert_val('EXP_TIME'),
            'DET_EXPSTEP': convert_val('EXP_STEP'),
            'DET_KP1': convert_val('EXP_KP1'),
            'DET_KP2': convert_val('EXP_KP2'),
            'DET_NOFSLOPES': self.__hdr[0]['NR_SLOPES'],
            'DET_EXPSEQ': self.__hdr[0]['EXP_SEQ'],
            'DET_EXPTIME2': convert_val('EXP_TIME2'),
            'DET_EXPSTEP2': convert_val('EXP_STEP2'),
            'DET_EXP2_SEQ': self.__hdr[0]['EXP2_SEQ'],
            'DET_NOFFRAMES': convert_val('NUMBER_FRAMES'),
            'DET_OUTMODE': self.__hdr[0]['OUTPUT_MODE'],
            'DET_FOTLEN': self.__hdr[0]['FOT_LENGTH'],
            'DET_ILVDSRCVR': self.__hdr[0]['I_LVDS_REC'],
            'DET_CALIB': self.__hdr[0]['COL_CALIB'],
            'DET_TRAINPTRN': convert_val('TRAINING_PATTERN'),
            'DET_CHENA': convert_val('CHANNEL_EN'),
            'DET_ILVDS': self.__hdr[0]['I_LVDS'],
            'DET_ICOL': self.__hdr[0]['I_COL'],
            'DET_ICOLPR': self.__hdr[0]['I_COL_PRECH'],
            'DET_IADC': self.__hdr[0]['I_ADC'],
            'DET_IAMP': self.__hdr[0]['I_AMP'],
            'DET_VTFL1': self.__hdr[0]['VTF_L1'],
            'DET_VTFL2': self.__hdr[0]['VLOW2'],
            'DET_VTFL3': self.__hdr[0]['VLOW3'],
            'DET_VRSTL': self.__hdr[0]['VRES_LOW'],
            'DET_VPRECH': self.__hdr[0]['V_PRECH'],
            'DET_VREF': self.__hdr[0]['V_REF'],
            'DET_VRAMP1': self.__hdr[0]['VRAMP1'],
            'DET_VRAMP2': self.__hdr[0]['VRAMP2'],
            'DET_OFFSET': convert_val('OFFSET'),
            'DET_PGAGAIN': self.__hdr[0]['PGA_GAIN'],
            'DET_ADCGAIN': self.__hdr[0]['ADC_GAIN'],
            'DET_TDIG1': self.__hdr[0]['T_DIG1'],
            'DET_TDIG2': self.__hdr[0]['T_DIG2'],
            'DET_BITMODE': self.__hdr[0]['BIT_MODE'],
            'DET_ADCRES': self.__hdr[0]['ADC_RESOLUTION'],
            'DET_PLLENA': self.__hdr[0]['PLL_ENABLE'],
            'DET_PLLINFRE': self.__hdr[0]['PLL_IN_FRE'],
            'DET_PLLBYP': self.__hdr[0]['PLL_BYPASS'],
            'DET_PLLRATE': self.__hdr[0]['PLL_RANGE'],
            'DET_PLLLOAD': self.__hdr[0]['PLL_LOAD'],
            'DET_DETDUM': self.__hdr[0]['DUMMY'],
            'DET_BLACKCOL': self.__hdr[0]['BLACK_COL_EN'],
            'DET_VBLACKSUN': self.__hdr[0]['V_BLACKSUN'],
            'DET_T': convert_val('TEMP')
        }

        mps = np.zeros((1,), dtype=np.dtype(tmtc_def(0x350)))
        for key in convert_det_params:
            mps[0][key] = convert_det_params[key]

        return mps

    def read_data(self, dat_file: str, numlines=None):
        """
        Returns data of a detector frame (numpy uint16 array)

        Parameters
        ----------
        dat_file : str
           Name of the (binary) data file
        numlines : int
           Provide number of detector rows when no headerfile is present
        """
        if numlines is None:
            if self.hdr is None:
                self.read_hdr(dat_file.replace('b.bin', 'a.txt'))

            # obtain number of rows
            numlines = self.number_lines()

        # Read binary big-endian data
        return np.fromfile(dat_file, dtype='>u2').reshape(numlines, -1)

    def number_lines(self):
        """
        Return number of lines (rows)

        Register address: [1, 2]
        """
        return ((self.hdr['NUMBER_LINES'][1] << 8)
                + self.hdr['NUMBER_LINES'][0])

    def lvds_clock(self):
        """
        Return LVDS clock input (0: disable, 1: enable)

        Register address: 82
        """
        return (self.hdr['CHANNEL_EN'] >> 2) & 0x1

    def pll_control(self):
        """
        Returns PLL control parameters: pll_range, pll_out_fre, pll_div

        PLL_range:    range (0 or 1)
        PLL_out_fre:  output frequency (0, 1, 2 or 5)
        PLL_div:      9 (10 bit) or 11 (12 bit)

        Register address: 116
        """
        pll_div = self.hdr['PLL_RANGE'] & 0xF            # bit [0:4]
        pll_out_fre = (self.hdr['PLL_RANGE'] >> 4) & 0x7 # bit [4:7]
        pll_range = (self.hdr['PLL_RANGE'] >> 7)         # bit [7]

        return (pll_range, pll_out_fre, pll_div)

    def exp_control(self):
        """
        Exposure time control

        Register address: 41
        """
        inte_sync = (self.hdr['INTE_SYNC'] >> 2)  & 0x1
        exp_dual = (self.hdr['INTE_SYNC'] >> 1) & 0x1
        exp_ext = self.hdr['INTE_SYNC'] & 0x1

        return (inte_sync, exp_dual, exp_ext)

    def offset(self):
        """
        Return digital offset including ADC offset

        Register address: [100, 101]
        """
        val = ((self.hdr['OFFSET'][1] << 8)
               + self.hdr['OFFSET'][0])

        return 70 + (val if val < 8192 else val - 16384)

    def pga_gain(self):
        """
        Returns PGA gain (Volt)

        Register address: 102
        """
        reg_pgagain = self.hdr['PGA_GAIN']
        # need first bit of address 121
        reg_pgagainfactor = self.hdr['BLACK_COL_EN'] & 0x1

        return (1 + 0.2 * reg_pgagain) * 2 ** reg_pgagainfactor

    def temp_detector(self):
        """
        equation: ((1184-1066) * 0.3 * 40 / 40Mhz) + offs [K]
        """
        return (self.hdr['TEMP'][1] << 8) + self.hdr['TEMP'][0]

    def t_exp(self, t_mcp=1e-7):
        """
        Return image-pixel exposure time (s)
        """
        # Nominal fot_length = 20, except for very short exposure_time
        reg_fot = self.hdr['FOT_LENGTH']

        reg_exptime = ((self.hdr['EXP_TIME'][2] << 16)
                       + (self.hdr['EXP_TIME'][1] << 8)
                       + self.hdr['EXP_TIME'][0])

        return 129 * t_mcp * (0.43 * reg_fot + reg_exptime)

    def t_fot(self, t_mcp=1e-7, n_ch=2):
        """
        Returns frame overhead time (s)
        """
        # Nominal fot_length = 20, except for very short exposure_time
        reg_fot = self.hdr['FOT_LENGTH']

        return 129 * t_mcp * (reg_fot + 2 * (16  // n_ch))

    def t_rot(self, t_mcp=1e-7, n_ch=2):
        """
        Returns image read-out time (s)
        """
        return 129 * t_mcp * (16 // n_ch) * 2048

    def t_frm(self, n_coad=1):
        """
        Returns frame period (s)
        """
        return n_coad * (self.t_exp() + self.t_fot() + self.t_rot()) + 2.38
