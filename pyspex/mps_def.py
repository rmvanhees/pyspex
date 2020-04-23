"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Definition of the SPEXone Measurement Parameter Settings as stored in the
CCSDS pacckages. The fill methods of MPSdef can be used to write detector
parameters of the DEM measurements to a MPS data structure.

* MPS header
** ICU S/W version
** MPS ID
** MPS version

* MPS ICU measurement control parameters
** fto: Frame trigger timing offset w.r.t. T0 (LSB 0.1 ms)
** fti: Frame trigger timing between frame triggers (LSB 0.1 ms)
** ftc: Frame trigger clearance period (LSB 0.1 ms)
** imro: Image read-out timing offset w.r.t. first trigger pulse
   in number of trigger pulses
** imrsa_A: Start address A of the image read out, if REG_STATUS bit 3 is 0
** imrsa_B: Start address B of the image read out, if REG_STATUS bit 3 is 1
** imrlen: Length of the image to read out in bytes
** pktlen: Maximum length of DEM SpaceWire read packets
** tmro: Telemetry parameter read-out timing offset w.r.t. T0 (LSB 0.1 ms)
** tmri: Telemetry parameter read-out interval (LSB 0.1 ms)
** imdmode: Image data mode of the ICU

* MPS DEM control parameters
** dem_rst: DEM reset register (valid: 0, 1)
** dem_cmv_ctrl: Trigger and detector sync register (valid: 0)
** coadd: Co-adding factor
** dem_igen: Selection of detector data or DEM internal test generator
** frame_mode: Frame mode identifier (valid: 0, 2, 3)
** outpmode: Number of active detector LVDS interfaces (valid: 1, 3)
** bin_tbl: Start address of the non-linear binning table
** coadd_buf: Start address of buffer for storing intermediate results
** coadd_res: Start address of buffer for storing final co-adding result
** frame_bufa: Start address of ping-pong buffer A containing a binned frame
** frame_bufb: Start address of ping-pong buffer B containing a binned frame
** line_ena: Start address of line enable vector

* MPS Detector settings, excluding the unused registers.
** numlin: Number of lines (default 2048)
** str1: Start 1 (default 0)
** str2: Start 2 (default 0)
** str3: Start 3 (default 0)
** str4: Start 4 (default 0)
** str5: Start 5 (default 0)
** str6: Start 6 (default 0)
** str7: Start 7 (default 0)
** str8: Start 8 (default 0)
** numlin1: Number of lines 1 (default 0)
** numlin2: Number of lines 2 (default 0)
** numlin3: Number of lines 3 (default 0)
** numlin4: Number of lines 4 (default 0)
** numlin5: Number of lines 5 (default 0)
** numlin6: Number of lines 6 (default 0)
** numlin7: Number of lines 7 (default 0)
** numlin8: Number of lines 8 (default 0)
** subS: Sub S (default 0)
** subA: Sub A (default 0)
** mono: Mono (default 1)
** imflp: Image flipping (default 0)
** expctrl: Exposure control (default 4)
** exptime: Exposure time (default 2048)
** expstep: Exposure step (default 0)
** expkp1: Exposure kp1 (default 1)
** expkp2: Exposure kp2 (default 1)
** outpmode: Output mode (default 0)
** nrslope: Number of slopes (default 1)
** expseq: Exposure squence (default 1)
** exptime2: Exposure time 2 (default 2048)
** expstep2: Exposure step 2 (default 0)
** expseq2: Exposure squence 2 (default 1)
** numfr: Number of frames (default 1)
** fotlen: FOT length (default 20)
** ilvdsrcvr: I_LVDS record (default 8)
** calib: Column/ADC calib (default 0)
** trainptrn: Training pattern (default 85)
** chena: Channel enable (default 524287)
** ilvds: I_LVDS (default 8)
** icol: I_column (default 4)
** icolpr: I_column_prech (default 1)
** ladc: I_adc (default 14)
** lamp: I_amp (default 12)
** vtfl1: Vtf_L1 (default 64)
** vtfl2: V_low2 (default 96)
** vtfl3: V_low3 (default 96)
** vrstl: V_res_low (default 64)
** vprech: V_prech (default 101)
** vref: V_ref(default 106)
** vramp1: V_ramp1 (default 96)
** vramp2: V_ramp2 (default 96)
** offset: Offset (default 195)
** pgagain: PGA gain (default 1)
** adcgain: ADC gain (default 32)
** tdig1: T_dig1 (default 0)
** tdig2: T_dig2 (default 1)
** bitmode: Bit mode (default 1)
** adcres: ADC resolution (default 0)
** pllena: PLL enable (default 1)
** pllinfre: PLL in fre (default 0)
** pllbyp: PLL bypass (default 0)
** pllrate: PLL rate (default 217)
** pllload: PLL load (default 8)
** detdnum: Detector dummy (default 1)
** blackcol: Black column enable (default 0)
** vblacksun: V_blacksun (default 98)

Copyright (c) 2019 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np

# - global parameters ------------------------------


# - local functions --------------------------------


# - class MPSdef -------------------------
class MPSdef:
    """
    Defines SPEXone MPS table

    Doc: MPS definition (78), v1.0, 2018-10-19 (section 7)
    """
    def __init__(self):
        self.__dtype = np.dtype([
            ('ICUSWVER', '>u2'),            #	0     0x00C
            ('MPS_ID', 'u1'),               #	2     0x00E
            ('MPS_VER', 'u1'),              #	3     0x00F
            ('TS1_DEM_N_T', '>u4'),         #	4     0x010
            ('TS2_HOUSING_N_T', '>u4'),     #	8     0x014
            ('TS3_RADIATOR_N_T', '>u4'),    #	12
            ('TS4_DEM_R_T', '>u4'),	    #	16
            ('TS5_HOUSING_R_T', '>u4'),     #	20
            ('TS6_RADIATOR_R_T', '>u4'),    #	24
            ('ICU_5V_T', '>u2'),	    #	28
            ('ICU_4V_T', '>u2'),	    #	30
            ('ICU_HG1_T', '>u2'),	    #	32
            ('ICU_HG2_T', '>u2'),	    #	34
            ('ICU_MID_T', '>u2'),	    #	36
            ('ICU_MCU_T', '>u2'),	    #	38
            ('ICU_DIGV_T', '>u2'),	    #	40
            ('ICU_4p0V_V', '>u2'),	    #	42
            ('ICU_3p3V_V', '>u2'),	    #	44
            ('ICU_1p2V_V', '>u2'),	    #	46
            ('ICU_4p0V_I', '>u2'),	    #	48
            ('ICU_3p3V_I', '>u2'),	    #	50
            ('ICU_1p2V_I', '>u2'),	    #	52
            ('ICU_5p0V_V', '>u2'),	    #	54
            ('ICU_5p0V_I', '>u2'),	    #	56
            ('DEM_V', '>u2'),	            #	58
            ('DEM_I', '>u2'),	            #	60
            ('LED1_ANODE_V', '>u4'),        #	62
            ('LED1_CATH_V', '>u4'),	    #	66
            ('LED1_I', '>u4'),	            #	70
            ('LED2_ANODE_V', '>u4'),        #	74
            ('LED2_CATH_V', '>u4'),	    #	78
            ('LED2_I', '>u4'),	            #	82
            ('ADC1_VCC', '>u4'),	    #	86
            ('ADC1_REF', '>u4'),	    #	90
            ('ADC1_T', '>u4'),	            #	94
            ('ADC2_VCC', '>u4'),	    #	98
            ('ADC2_REF', '>u4'),	    #	102
            ('ADC2_T', '>u4'),	            #	106
            ('REG_FW_VERSION', 'u1'),           #	110   0x7a
            ('REG_NCOADDFRAMES', 'u1'),         #	111
            ('REG_IGEN_SELECT', 'u1'),          #	112
            ('REG_FULL_FRAME', 'u1'),           #	113
            ('REG_BINNING_TABLE', '>u4'),	#	114
            ('REG_CMV_OUTPUTMODE', 'u1'),	#	118   0x82
            ('dummy_01', 'u1'),	                #	119
            ('REG_COADD_BUF_START', '>u4'),	#	120
            ('REG_COADD_RESA_START', '>u4'),	#	124
            ('REG_COADD_RESB_START', '>u4'),	#	124
            ('REG_FRAME_BUFA_START', '>u4'),	#	128
            ('REG_FRAME_BUFB_START', '>u4'),	#	132
            ('REG_LINE_ENABLE_START', '>u4'),	#	136
            ('DET_REG000', 'u1'),	    #	140
            ('dummy_02', 'u1'), 	    #	141
            ('DET_NUMLINES', '>u2'),        #	142
            ('DET_START1', '>u2'),	    #	144
            ('DET_START2', '>u2'),	    #	146
            ('DET_START3', '>u2'),	    #	148
            ('DET_START4', '>u2'),	    #	150
            ('DET_START5', '>u2'),	    #	152
            ('DET_START6', '>u2'),	    #	154
            ('DET_START7', '>u2'),	    #	156
            ('DET_START8', '>u2'),	    #	158
            ('DET_NUMLINES1', '>u2'),       #	160
            ('DET_NUMLINES2', '>u2'),       #	162
            ('DET_NUMLINES3', '>u2'),       #	164
            ('DET_NUMLINES4', '>u2'),       #	166
            ('DET_NUMLINES5', '>u2'),       #	168
            ('DET_NUMLINES6', '>u2'),       #	170
            ('DET_NUMLINES7', '>u2'),       #	172
            ('DET_NUMLINES8', '>u2'),       #	174
            ('DET_SUBS', '>u2'),	    #	176
            ('DET_SUBA', '>u2'),	    #	178
            ('DET_MONO', 'u1'), 	    #	180
            ('DET_IMFLIP', 'u1'),	    #	181
            ('DET_EXPCNTR', 'u1'),	    #	182
            ('DET_ILVDS', 'u1'), 	    #	183
            ('DET_EXPTIME', '>u4'),	    #	184
            ('DET_EXPSTEP', '>u4'),	    #	188
            ('DET_KP1', '>u4'),	            #	192
            ('DET_KP2', '>u4'),	            #	196
            ('DET_NOFSLOPES', 'u1'),        #	200   0x0D4
            ('DET_EXPSEQ', 'u1'),	    #	201
            ('DET_EXPTIME2', '>u4'),        #	202
            ('DET_EXPSTEP2', '>u4'),        #	206
            ('DET_REG062', 'u1'),	    #	210
            ('DET_REG063', 'u1'),	    #	211
            ('DET_REG064', 'u1'),	    #	212
            ('DET_REG065', 'u1'),	    #	213
            ('DET_REG066', 'u1'),	    #	214
            ('DET_REG067', 'u1'),	    #	215
            ('DET_REG068', 'u1'),	    #	216
            ('DET_EXP2_SEQ', 'u1'),         #	217
            ('DET_NOFFRAMES', '>u2'),       #	218
            ('DET_OUTMODE', 'u1'),	    #	220
            ('DET_FOTLEN', 'u1'),	    #	221
            ('DET_ILVDSRCVR', 'u1'),        #	222
            ('DET_REG075', 'u1'),	    #	224
            ('DET_REG076', 'u1'),	    #	225
            ('DET_CALIB', 'u1'),	    #	223
            ('DET_TRAINPTRN', '>u2'),       #	226
            ('DET_CHENA', '>u4'),	    #	228
            ('DET_ICOL', 'u1'), 	    #	232   0x0F4
            ('DET_ICOLPR', 'u1'),	    #	233
            ('DET_IADC', 'u1'), 	    #	234
            ('DET_IAMP', 'u1'), 	    #	235
            ('DET_VTFL1', 'u1'),	    #	236
            ('DET_VTFL2', 'u1'),	    #	237
            ('DET_VTFL3', 'u1'),	    #	238
            ('DET_VRSTL', 'u1'),	    #	239
            ('DET_REG092', 'u1'),	    #	240
            ('DET_REG093', 'u1'),	    #	241
            ('DET_VPRECH', 'u1'),	    #	242
            ('DET_VREF', 'u1'), 	    #	243
            ('DET_REG096', 'u1'),	    #	244
            ('DET_REG097', 'u1'),	    #	245
            ('DET_VRAMP1', 'u1'),	    #	246
            ('DET_VRAMP2', 'u1'),	    #	247
            ('DET_OFFSET', '>u2'),	    #	248
            ('DET_PGAGAIN', 'u1'),	    #	250
            ('DET_ADCGAIN', 'u1'),	    #	251
            ('DET_REG104', 'u1'),	    #  	252
            ('DET_REG105', 'u1'),	    #	253
            ('DET_REG106', 'u1'),	    #	254
            ('DET_REG107', 'u1'),	    #	255
            ('DET_TDIG1', 'u1'),	    #	256
            ('DET_TDIG2', 'u1'),	    #	257
            ('DET_REG110', 'u1'),	    #	258
            ('DET_BITMODE', 'u1'),	    #	259
            ('DET_ADCRES', 'u1'),	    #	260
            ('DET_PLLENA', 'u1'),	    #	261
            ('DET_PLLINFRE', 'u1'),	    #	262
            ('DET_PLLBYP', 'u1'),	    #	263
            ('DET_PLLRATE', 'u1'),	    #	264
            ('DET_PLLLOAD', 'u1'),	    #	265
            ('DET_DETDUM', 'u1'),	    #	266
            ('DET_REG119', 'u1'),	    #	267
            ('DET_REG120', 'u1'),	    #	268
            ('DET_BLACKCOL', 'u1'),         #	269
            ('DET_REG122', 'u1'),	    #	270
            ('DET_VBLACKSUN', 'u1'),        #	271
            ('DET_REG124', 'u1'),	    #	272
            ('DET_REG125', 'u1'),	    #	273
            ('DET_T', '>u2'),	            #	274
            ('FTI', '>u2'),	            #	276
            ('IMDMODE', 'u1'),	            #   278    0x122
            ('dummy_03', 'u1'), 	    #	279
            ('IMRLEN', '>u4')	            #	280    0x124
        ])

    @property
    def dtype(self):
        """
        Returns numpy data-type for MPS record
        """
        return self.__dtype

    @staticmethod
    def __fill_hdr(mps, hdr):
        """
        Write DEM header parameters to MPS
        """
        for key in hdr:
            mps[0][key] = hdr[key]

    @staticmethod
    def __fill_icu(mps, icu):
        """
        Write original ICU parameters to MPS
        """
        # convert original ICU parameter values to MPS parameters
        convert_icu_params = {
            # '': icu['FTO'],
            'FTI': icu['FTI'],
            # '': icu['FTC'],
            # '': icu['IMRO'],
            # '': icu['IMRSA_A'],
            # '': icu['IMRSA_B'],
            'IMRLEN': icu['IMRLEN'],
            # '': icu['PKTLEN'],
            # '': icu['TMRO'],
            # '': icu['TMRI'],
            'IMDMODE': icu['IMDMODE']
        }

        for key in convert_icu_params:
            mps[0][key] = convert_icu_params[key]

    @staticmethod
    def __fill_dem(mps, dem):
        """
        Write original DEM parameters to MPS
        """
        # convert original DEM parameter values to MPS parameters
        convert_dem_params = {
            # '': dem['DEM_RST'],
            # '': dem['DEM_CMV_CTRL'],
            'REG_NCOADDFRAMES': dem['COADD'],
            'REG_IGEN_SELECT': dem['DEM_IGEN'],
            'REG_FULL_FRAME': dem['FRAME_MODE'],
            'REG_CMV_OUTPUTMODE': dem['OUTPMODE'],
            'REG_BINNING_TABLE_START': dem['BIN_TBL'],
            'REG_COADD_BUF_START': dem['COADD_BUF'],
            'REG_COADD_RESA_START': dem['COADD_RESA'],
            'REG_COADD_RESB_START': dem['COADD_RESB'],
            'REG_FRAME_BUFA_START': dem['FRAME_BUFA'],
            'REG_FRAME_BUFB_START': dem['FRAME_BUFB'],
            'REG_LINE_ENABLE_START': dem['LINE_ENA']
        }

        for key in convert_dem_params:
            mps[0][key] = convert_dem_params[key]

    @staticmethod
    def __fill_det(mps, det):
        """
        Write DEM detector parameters to MPS
        """
        def convert_val(key):
            """
            Convert byte array to integer
            """
            val = 0
            for ii, bval in enumerate(det[key]):
                val += bval << (ii * 8)

            return val

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
            'DET_MONO': det['MONO'],
            'DET_IMFLIP': det['IMAGE_FLIPPING'],
            'DET_EXPCNTR': det['INTE_SYNC'],
            'DET_EXPTIME': convert_val('EXP_TIME'),
            'DET_EXPSTEP': convert_val('EXP_STEP'),
            'DET_KP1': convert_val('EXP_KP1'),
            'DET_KP2': convert_val('EXP_KP2'),
            'DET_NOFSLOPES': det['NR_SLOPES'],
            'DET_EXPSEQ': det['EXP_SEQ'],
            'DET_EXPTIME2': convert_val('EXP_TIME2'),
            'DET_EXPSTEP2': convert_val('EXP_STEP2'),
            'DET_EXP2_SEQ': det['EXP2_SEQ'],
            'DET_NOFFRAMES': convert_val('NUMBER_FRAMES'),
            'DET_OUTMODE': det['OUTPUT_MODE'],
            'DET_FOTLEN': det['FOT_LENGTH'],
            'DET_ILVDSRCVR': det['I_LVDS_REC'],
            'DET_CALIB': det['COL_CALIB'],
            'DET_TRAINPTRN': convert_val('TRAINING_PATTERN'),
            'DET_CHENA': convert_val('CHANNEL_EN'),
            'DET_ILVDS': det['I_LVDS'],
            'DET_ICOL': det['I_COL'],
            'DET_ICOLPR': det['I_COL_PRECH'],
            'DET_IADC': det['I_ADC'],
            'DET_IAMP': det['I_AMP'],
            'DET_VTFL1': det['VTF_L1'],
            'DET_VTFL2': det['VLOW2'],
            'DET_VTFL3': det['VLOW3'],
            'DET_VRSTL': det['VRES_LOW'],
            'DET_VPRECH': det['V_PRECH'],
            'DET_VREF': det['V_REF'],
            'DET_VRAMP1': det['VRAMP1'],
            'DET_VRAMP2': det['VRAMP2'],
            'DET_OFFSET': convert_val('OFFSET'),
            'DET_PGAGAIN': det['PGA_GAIN'],
            'DET_ADCGAIN': det['ADC_GAIN'],
            'DET_TDIG1': det['T_DIG1'],
            'DET_TDIG2': det['T_DIG2'],
            'DET_BITMODE': det['BIT_MODE'],
            'DET_ADCRES': det['ADC_RESOLUTION'],
            'DET_PLLENA': det['PLL_ENABLE'],
            'DET_PLLINFRE': det['PLL_IN_FRE'],
            'DET_PLLBYP': det['PLL_BYPASS'],
            'DET_PLLRATE': det['PLL_RANGE'],
            'DET_PLLLOAD': det['PLL_LOAD'],
            'DET_DETDUM': det['DUMMY'],
            'DET_BLACKCOL': det['BLACK_COL_EN'],
            'DET_VBLACKSUN': det['V_BLACKSUN'],
            'DET_T': convert_val('TEMP')
        }

        for key in convert_det_params:
            mps[0][key] = convert_det_params[key]


    def fill(self, det, dem=None, icu=None, hdr=None):
        """
        Fill MPS compound with information (FillValue = 0)

        Parameters
        ----------
        det : ndarray
              structured array with names, values of all detector registers
        dem : ndarray, optional
              structured array with names, values of all DEM variables
        icu : ndarray, optional
              structured array with names, values of all ICU variables
        hdr : dictionary, optional
              Dictionary with MPS ID and version
        """
        # initialize return value
        mps = np.zeros((1,), dtype=self.dtype)

        if hdr is not None:
            self.__fill_hdr(mps, hdr)

        if icu is not None:
            self.__fill_icu(mps, icu)

        if dem is not None:
            self.__fill_dem(mps, dem)

        self.__fill_det(mps, det)

        return mps
