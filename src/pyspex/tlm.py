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
Contains the class `SPXtlm` to read/access/convert telemetry house-keeping
parameters of SPEXone.
"""
from __future__ import annotations
__all__ = ['SPXtlm']

from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path

import h5py
import numpy as np

from .lib.leap_sec import get_leap_seconds
from .hkt_io import HKTio
from .lv0_io import (grouping_flag, read_lv0_data)

# - global parameters -----------------------
MCP_TO_SEC = 1e-7
# This dictionary is only valid when raw counts are converted to physical units
UNITS_DICT = {'ADC1_GAIN': 'Volt',
              'ADC1_OFFSET': 'Volt',
              'ADC1_REF': 'Volt',
              'ADC1_T': 'degree_Celsius',
              'ADC1_VCC': 'Volt',
              'ADC2_GAIN': 'Volt',
              'ADC2_OFFSET': 'Volt',
              'ADC2_REF': 'Volt',
              'ADC2_T': 'degree_Celsius',
              'ADC2_VCC': 'Volt',
              'DEM_I': 'mA',
              'DEM_T': 'degree_Celsius',
              'DEM_V': 'Volt',
              'HTR1_DutyCycl': '%',
              'HTR1_I': 'mA',
              'HTR2_DutyCycl': '%',
              'HTR2_I': 'mA',
              'HTR3_DutyCycl': '%',
              'HTR3_I': 'mA',
              'HTR4_DutyCycl': '%',
              'HTR4_I': 'mA',
              'HTRGRP1_V': 'Volt',
              'HTRGRP2_V': 'Volt',
              'ICU_1p2V_I': 'mA',
              'ICU_1p2V_V': 'Volt',
              'ICU_3p3V_I': 'mA',
              'ICU_3p3V_V': 'Volt',
              'ICU_4p0V_I': 'mA',
              'ICU_4p0V_V': 'Volt',
              'ICU_4V_T': 'degree_Celsius',
              'ICU_5p0V_I': 'mA',
              'ICU_5p0V_V': 'Volt',
              'ICU_5V_T': 'degree_Celsius',
              'ICU_DIGV_T': 'degree_Celsius',
              'ICU_HG1_T': 'degree_Celsius',
              'ICU_HG2_T': 'degree_Celsius',
              'ICU_MCU_T': 'degree_Celsius',
              'ICU_MID_T': 'degree_Celsius',
              'LED1_ANODE_V': 'Volt',
              'LED1_CATH_V': 'Volt',
              'LED1_I': 'mA',
              'LED2_ANODE_V': 'Volt',
              'LED2_CATH_V': 'Volt',
              'LED2_I': 'mA',
              'TS1_DEM_N_T': 'degree_Celsius',
              'TS2_HOUSING_N_T': 'degree_Celsius',
              'TS3_RADIATOR_N_T': 'degree_Celsius',
              'TS4_DEM_R_T': 'degree_Celsius',
              'TS5_HOUSING_R_T': 'degree_Celsius',
              'TS6_RADIATOR_R_T': 'degree_Celsius'}


# - helper functions ------------------------
def get_epoch(tstamp: int) -> datetime:
    """Return epoch of timestamp.
    """
    if tstamp < 1956528000:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    return (datetime(1958, 1, 1, tzinfo=timezone.utc)
            - timedelta(seconds=get_leap_seconds(tstamp)))


def subsec2musec(sub_sec: int) -> int:
    """Return subsec as microseconds.
    """
    return 100 * int(sub_sec / 65536 * 10000)


def exp_spex_det_t(raw_data: np.ndarray) -> np.ndarray:
    """Convert Detector Temperature Sensor (at DEM) to degrees Celsius.
    """
    res = np.empty(raw_data.size, dtype=float)
    mask = raw_data < 400
    res[mask] = 1.224 * raw_data[mask] - 290.2
    res[~mask] = 0.6426 * raw_data[~mask] - 418.72
    return res


def exp_spex_thermistor(raw_data: np.ndarray) -> np.ndarray:
    """Convert listed Temperature Sensor (at DEM) to degrees Celsius::

       - TS1 DEM Nominal temperature
       - TS2 Housing Nominal Temperature
       - TS3 Radiator Nominal Temperature
       - TS4 DEM Redundant Temperature
       - TS5 Housing Redundant Temperature
       - TS6 Radiator Redundant Temperature*
    """
    coefficients = (21.19, 272589.0, -1.5173e-15, 5.73666e-19, -5.11328e-20)
    return (coefficients[0]
            + coefficients[1] / raw_data
            + coefficients[2] * raw_data ** 4
            + (coefficients[3] + coefficients[4] * np.log(raw_data))
            * raw_data ** 5)


def poly_spex_icuhk_internaltemp(raw_data: np.ndarray) -> np.ndarray:
    """Convert temperature sensors on ICU power supplies to degrees Celsius::

       - ICU V5 supply temperature
       - ICU V4 supply temperature
       - ICU HtrG1 supply temperature
       - ICU HtrG2 supply temperature
       - ICU MidBoard temperature
       - ICU MCU-RAM temperature
       - ICU 1V2, 3V3 supply temperature
    """
    coefficients = (0, 0.0625)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_icuhk_internaltemp2(raw_data: np.ndarray) -> np.ndarray:
    """Convert readings of ICU bus voltages to Volt::

       - ICU 4.0 Volt bus voltage
       - ICU 3.3 Volt bus voltage
       - ICU 1.2 Volt bus voltage
       - DEM power supply
    """
    coefficients = (0, 0.001)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_htr_v(raw_data: np.ndarray) -> np.ndarray:
    """Convert Heater Bus voltages to Volt.
    """
    coefficients = (0, 0.003)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_dutycycle(raw_data: np.ndarray) -> np.ndarray:
    """Convert Heater Controller Duty Cycle output to percent.
    """
    coefficients = (0, 0.1)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_led_anode_v(raw_data: np.ndarray) -> np.ndarray:
    """Convert LED Anode voltage to Volt.
    """
    coefficients = (0, 0.000000623703003)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_led_cath_v(raw_data: np.ndarray) -> np.ndarray:
    """Convert LED Cathode voltage to Volt.
    """
    coefficients = (0, 0.000000415802001953)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_led_i(raw_data: np.ndarray) -> np.ndarray:
    """Convert LED current to milli-Ampere.
    """
    coefficients = (0, 0.0000030307446495961334745762711864407)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_adc_vcc(raw_data: np.ndarray) -> np.ndarray:
    """Convert ADC Analog VCC reading to Volt.
    """
    coefficients = (0, 0.00000127156575520833333)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_adc_gain(raw_data: np.ndarray) -> np.ndarray:
    """Convert ADC Analog VCC reading to Volt.
    """
    coefficients = (0, 0.000000127156575520833333)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_adc_t(raw_data: np.ndarray) -> np.ndarray:
    """Convert ADC1 Temperature reading to degrees Celsius.
    """
    coefficients = (-273.4, 0.0007385466842651367)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_adc_offset(raw_data: np.ndarray) -> np.ndarray:
    """Convert ADC offset (?) to Voltage.
    """
    coefficients = (0, 0.000415802001953)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_dem_i(raw_data: np.ndarray) -> np.ndarray:
    """Convert DEM Supply current to milli-Ampere.
    """
    coefficients = (0, 0.2417)
    return coefficients[0] + coefficients[1] * raw_data


class Enable(Enum):
    """Possible values of the DEM SpaceWire link status.
    """
    INACTIVE = 0
    ACTIVE = 1


class EnaDis(Enum):
    """Possible values of the LED commanded enable state.
    """
    OFF = 0
    ON = 1


class IcuMode(Enum):
    """Possible values of the ICU mode.
    """
    IDLE = 0
    OPER = 1


class IsEna(Enum):
    """Possible values of the Power Sense.
    """
    DISABLED = 0
    ENABLED = 1


class OcdReg(Enum):
    """Possible values of the OCD state.
    """
    PWR_OK = 0
    OC2V1 = 1
    OC3V3 = 2
    OC2V1_AND_3V3 = 3


class TimeMsg(Enum):
    """Received status of external PPS or Time message.
    """
    MISSING = 0
    RECEIVED = 1


class WriteProt(Enum):
    """Status of Flash Write protection (both banks).
    """
    BANK0_DISABLED = 0
    BANK0_ENABLED = 1
    BANK1_DISABLED = 16
    BANK1_ENABLED = 17


def convert_hk(key: str, raw_data: np.ndarray) -> np.nd_array:
    """Convert a DemHK or NomHK parameter to physical units.
    """
    conv_dict = {'DEM_T': exp_spex_det_t,
                 'TS1_DEM_N_T': exp_spex_thermistor,
                 'TS2_HOUSING_N_T': exp_spex_thermistor,
                 'TS3_RADIATOR_N_T': exp_spex_thermistor,
                 'TS4_DEM_R_T': exp_spex_thermistor,
                 'TS5_HOUSING_R_T': exp_spex_thermistor,
                 'TS6_RADIATOR_R_T': exp_spex_thermistor,
                 'ADC1_GAIN': poly_spex_adc_gain,
                 'ADC2_GAIN': poly_spex_adc_gain,
                 'ADC1_OFFSET': poly_spex_adc_offset,
                 'ADC2_OFFSET': poly_spex_adc_offset,
                 'ADC1_T': poly_spex_adc_t,
                 'ADC2_T': poly_spex_adc_t,
                 'ADC1_REF': poly_spex_adc_vcc,
                 'ADC1_VCC': poly_spex_adc_vcc,
                 'ADC2_REF': poly_spex_adc_vcc,
                 'ADC2_VCC': poly_spex_adc_vcc,
                 'DEM_I': poly_spex_dem_i,
                 'HTR1_DutyCycl': poly_spex_dutycycle,
                 'HTR2_DutyCycl': poly_spex_dutycycle,
                 'HTR3_DutyCycl': poly_spex_dutycycle,
                 'HTR4_DutyCycl': poly_spex_dutycycle,
                 'HTRGRP1_V': poly_spex_htr_v,
                 'HTRGRP2_V': poly_spex_htr_v,
                 'ICU_4V_T': poly_spex_icuhk_internaltemp,
                 'ICU_5V_T': poly_spex_icuhk_internaltemp,
                 'ICU_DIGV_T': poly_spex_icuhk_internaltemp,
                 'ICU_HG1_T': poly_spex_icuhk_internaltemp,
                 'ICU_HG2_T': poly_spex_icuhk_internaltemp,
                 'ICU_MCU_T': poly_spex_icuhk_internaltemp,
                 'ICU_MID_T': poly_spex_icuhk_internaltemp,
                 'DEM_V': poly_spex_icuhk_internaltemp2,
                 'ICU_1p2V_V': poly_spex_icuhk_internaltemp2,
                 'ICU_3p3V_V': poly_spex_icuhk_internaltemp2,
                 'ICU_4p0V_V': poly_spex_icuhk_internaltemp2,
                 'ICU_5p0V_V': poly_spex_icuhk_internaltemp2,
                 'LED1_ANODE_V': poly_spex_led_anode_v,
                 'LED2_ANODE_V': poly_spex_led_anode_v,
                 'LED1_CATH_V': poly_spex_led_cath_v,
                 'LED2_CATH_V': poly_spex_led_cath_v,
                 'LED1_I': poly_spex_led_i,
                 'LED2_I': poly_spex_led_i}

    func = conv_dict.get(key, None)
    if func is not None:
        return func(raw_data)

    return raw_data


# - class SPXtlm ----------------------------
class SPXtlm:
    """Access/convert parameters of SPEXone Science telemetry data.
    """
    def __init__(self, verbose: bool = False):
        """Initialize class SPXtlm.
        """
        self.filename = None
        self._verbose = verbose
        self._hdr = ()
        self._tlm = ()
        self.__tm = []

    @property
    def hdr(self):
        """Return CCSDS header data.
        """
        return self._hdr

    @property
    def tlm(self):
        """Return housekeeping packages.
        """
        return self._tlm

    @property
    def tstamp(self):
        """Return timestamps op CCSDS packages.
        """
        return self.__tm

    def hkt_tlm(self, flname: Path,
                instrument: str | None = None, apid: int | None = None):
        """Read telemetry dta from PACE HKT product.

        Parameters
        ----------
        flname :  Path
        instrument :  {'spx', 'sc', 'oci', 'harp'}, optional
        apid :  int, optional
        """
        if instrument is None:
            instrument = 'spx'
        elif instrument not in ['spx', 'sc', 'oci', 'harp']:
            raise KeyError("instrument not in ['spx', 'sc', 'oci', 'harp']")
        if apid is None and instrument == 'spx':
            apid = 0x320

        self.filename = flname
        self.__tm = []
        hkt = HKTio(flname, instrument)
        res = hkt.housekeeping(apid)
        self._hdr = res['hdr']
        self._tlm = res['hk']
        epoch = get_epoch(int(res['hdr']['tai_sec'][0]))
        for sec, subsec in [(x['tai_sec'], x['sub_sec']) for x in res['hdr']]:
            self.__tm.append(epoch + timedelta(
                seconds=int(sec), microseconds=subsec2musec(int(subsec))))

    def lv0_tlm(self, flname: Path, tlm_type: str | None = None):
        """Read telemetry data from SPEXone Level-0 product.

        Parameters
        ----------
        flname :  Path
        tlm_type :  {'NomHK', 'DemHK'}, optional

        Returns
        -------
        np.ndarray
        """
        if tlm_type is None:
            tlm_type = 'DemHK'
        elif tlm_type not in ['NomHK', 'DemHK']:
            raise KeyError("tlm_type not in ['NomHK', 'DemHK']")

        self.filename = flname
        self.__tm = []
        self._hdr = ()
        self._tlm = ()
        res = read_lv0_data([flname], file_format='dsb', verbose=self._verbose)
        ccsds_sci, ccsds_hk = res

        if tlm_type == 'NomHK':
            if not ccsds_hk:
                return
            if self._verbose:
                print('[INFO]: processing NomHK data')
            epoch = get_epoch(int(ccsds_hk[0]['hdr']['tai_sec']))
            self._hdr = np.empty(len(ccsds_hk),
                                 dtype=ccsds_hk[0]['hdr'].dtype)
            self._tlm = np.empty(len(ccsds_hk),
                                 dtype=ccsds_hk[0]['data']['hk'].dtype)
            for ii, segment in enumerate(ccsds_hk):
                self._hdr[ii] = segment['hdr']
                self._tlm[ii] = segment['data']['hk']
                self.__tm.append(epoch + timedelta(
                    seconds=int(segment['hdr']['tai_sec']),
                    microseconds=subsec2musec(int(segment['hdr']['sub_sec']))))
        else:
            if not ccsds_sci:
                return
            if self._verbose:
                print('[INFO]: processing DemHK data')
            epoch = get_epoch(int(ccsds_sci[0]['hdr']['tai_sec']))
            for segment in ccsds_sci:
                hdr = segment['hdr']
                if grouping_flag(hdr) != 1:
                    continue
                self._hdr = np.empty(len(ccsds_sci),
                                     dtype=segment['hdr'].dtype)
                self._tlm = np.empty(len(ccsds_sci),
                                     dtype=segment['data']['hk'].dtype)
                break
            else:
                raise RuntimeError('no valid Science package found')

            ii = 0
            for segment in ccsds_sci:
                hdr = segment['hdr']
                if grouping_flag(hdr) != 1:
                    continue
                data = segment['data'][0]

                self._hdr[ii] = hdr
                self._tlm[ii] = data['hk']
                self.__tm.append(epoch + timedelta(
                    seconds=int(data['icu_tm']['tai_sec']),
                    microseconds=subsec2musec(int(data['icu_tm']['sub_sec']))))
                ii += 1

            self._hdr = self._hdr[:ii]
            self._tlm = self._tlm[:ii]
            for ii, offs_ms in enumerate(self.start_integration):
                self.__tm[ii] -= timedelta(milliseconds=offs_ms)

    def l1a_tlm(self, flname: Path, tlm_type: str | None = None):
        """Read telemetry dta from SPEXone Level-1A product.

        Parameters
        ----------
        flname :  Path
        tlm_type :  str

        Returns
        -------
        np.ndarray
        """
        if tlm_type is None:
            tlm_type = 'DemHK'
        elif tlm_type not in ['NomHK', 'DemHK']:
            raise KeyError("tlm_type not in ['NomHK', 'DemHK']")

        self.filename = flname
        self.__tm = []
        self._hdr = ()
        self._tlm = ()
        with h5py.File(flname) as fid:
            if tlm_type == 'NomHK':
                self._tlm = fid['/engineering_data/NomHK_telemetry'][:]
                dset = fid['/engineering_data/HK_tlm_time']
                ref_date = dset.attrs['units'].decode()[14:] + 'Z'
                epoch = datetime.fromisoformat(ref_date)
                for sec in dset[:]:
                    self.__tm.append(epoch + timedelta(seconds=sec))
            else:
                self._tlm = fid['/science_data/detector_telemetry'][:]
                seconds = fid['/image_attributes/icu_time_sec'][:]
                subsec = fid['/image_attributes/icu_time_subsec'][:]
                epoch = get_epoch(int(seconds[0]))
                for ii, sec in enumerate(seconds):
                    self.__tm.append(epoch + timedelta(
                        seconds=int(sec),
                        milliseconds=-self.start_integration[ii],
                        microseconds=subsec2musec(int(subsec[ii]))))

    @property
    def binning_table(self):
        """Return binning table identifier (zero for full-frame images).

        Notes
        -----
        Requires SPEXone DemHK, will not work with NomHK

        v126: Sometimes the MPS information is not updated for the first \
              images. We try to fix this and warn the user.
        v129: REG_BINNING_TABLE_START is stored in BE instead of LE

        Returns
        -------
        np.ndarray, dtype=int
        """
        if 'REG_FULL_FRAME' not in self._tlm.dtype.names:
            print('[WARNING]: can not determine binning table identifier')
            return np.full(len(self._tlm), -1, dtype='i1')

        full_frame = np.unique(self._tlm['REG_FULL_FRAME'])
        if len(full_frame) > 1:
            print('[WARNING]: value of REG_FULL_FRAME not unique')
            print(self._tlm['REG_FULL_FRAME'])
        full_frame = self._tlm['REG_FULL_FRAME'][-1]

        cmv_outputmode = np.unique(self._tlm['REG_CMV_OUTPUTMODE'])
        if len(cmv_outputmode) > 1:
            print('[WARNING]: value of REG_CMV_OUTPUTMODE not unique')
            print(self._tlm['REG_CMV_OUTPUTMODE'])
        cmv_outputmode = self._tlm['REG_CMV_OUTPUTMODE'][-1]

        if full_frame == 1:
            if cmv_outputmode != 3:
                raise KeyError('Diagnostic mode with REG_CMV_OUTPMODE != 3')
            return np.zeros(len(self._tlm), dtype='i1')

        if full_frame == 2:
            if cmv_outputmode != 1:
                raise KeyError('Science mode with REG_CMV_OUTPUTMODE != 1')
            bin_tbl_start = self._tlm['REG_BINNING_TABLE_START']
            indx0 = (self._tlm['REG_FULL_FRAME'] != 2).nonzero()[0]
            if indx0.size > 0:
                indx2 = (self._tlm['REG_FULL_FRAME'] == 2).nonzero()[0]
                bin_tbl_start[indx0] = bin_tbl_start[indx2[0]]
            res = 1 + (bin_tbl_start - 0x80000000) // 0x400000
            return res & 0xFF

        raise KeyError('REG_FULL_FRAME not equal to 1 or 2')

    @property
    def start_integration(self):
        """Return offset wrt start-of-integration [msec].

        Notes
        -----
        Requires SPEXone DemHK, will not work with NomHK

        Determine offset wrt start-of-integration (IMRO + 1)
        Where the default is defined as IMRO::

        - [full-frame] COADDD + 2  (no typo, this is valid for the later MPS's)
        - [binned] 2 * COADD + 1   (always valid)
        """
        if self._tlm['ICUSWVER'][0] <= 0x123:
            return 0

        if np.bincount(self.binning_table).argmax() == 0:
            imro = self._tlm['REG_NCOADDFRAMES'] + 2
        else:
            imro = 2 * self._tlm['REG_NCOADDFRAMES'] + 1
        return self._tlm['FTI'] * (imro + 1) / 10

    def convert(self, key: str) -> np.ndarray:
        """Convert telemetry parameter to physical units.

        Parameters
        ----------
        key :  str
           Name of telemetry parameter

        Returns
        -------
        np.ndarray
        """
        if key not in self._tlm[0].dtype.names:
            raise KeyError(f'Parameter: {key} not found'
                           f' in {self._tlm[0].dtype.names}')

        raw_data = np.array([tlm[key] for tlm in self._tlm])
        return convert_hk(key, raw_data)


def __test():
    data_dir = Path('/data2/richardh/SPEXone/spx1_lv0/0x12d/2023/05/25')
    flname = data_dir / 'SPX000000880.spx'

    tlm = SPXtlm()
    tlm.lv0_tlm(flname, 'NomHK')
    # tlm.show_tlm()
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('deltaT: ', np.unique(np.diff([tm.timestamp() for tm in tlm.tstamp])))

    tlm.lv0_tlm(flname, 'DemHK')
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('ICU_5p0V_V', tlm.convert('ICU_5p0V_V'),
          UNITS_DICT.get('ICU_5p0V_V', '1'))
    print('REG_CMV_OUTPUTMODE', tlm.convert('REG_CMV_OUTPUTMODE'),
          UNITS_DICT.get('REG_CMV_OUTPUTMODE', '1'))
    print('binning_table: ', tlm.binning_table)
    print('offs_msec: ', tlm.start_integration)
    print('deltaT: ', np.unique(np.diff([tm.timestamp() for tm in tlm.tstamp])))


def __test2():
    data_dir = Path('/data2/richardh/SPEXone/pace_hkt/V1.0/2023/05/25')
    if not data_dir.is_dir():
        data_dir = Path('/nfs/SPEXone/ocal/pace-sds/pace_hkt/V1.0/2023/05/25')
    flname = data_dir / 'PACE.20230525T043614.HKT.nc'
    tlm = SPXtlm()
    tlm.hkt_tlm(flname, instrument='spx', apid=0x320)
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('deltaT: ', np.unique(np.diff([tm.timestamp() for tm in tlm.tstamp])))


def __test3():
    data_dir = Path('/data2/richardh/SPEXone/spx1_l1a/0x12d/2023/05/25')
    flname = data_dir / 'PACE_SPEXONE_OCAL.20230525T030148.L1A.nc'
    tlm = SPXtlm()
    tlm.l1a_tlm(flname, 'NomHK')
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('deltaT: ', np.unique(np.diff([tm.timestamp() for tm in tlm.tstamp])))

    tlm.l1a_tlm(flname, 'DemHK')
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('binning_table: ', tlm.binning_table)
    print('offs_msec: ', tlm.start_integration)
    print('deltaT: ', np.unique(np.diff([tm.timestamp() for tm in tlm.tstamp])))


if __name__ == '__main__':
    __test()
