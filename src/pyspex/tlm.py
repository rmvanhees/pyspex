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

import numpy as np

from .lib.leap_sec import get_leap_seconds
from .hkt_io import HKTio
from .lv0_io import (grouping_flag, read_lv0_data)

# - global parameters -----------------------
MCP_TO_SEC = 1e-7


# - helper functions ------------------------
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
    conv_dict = {'TS1_DEM_N_T': exp_spex_thermistor,
                 'TS2_HOUSING_N_T': exp_spex_thermistor,
                 'TS3_RADIATOR_N_T': exp_spex_thermistor,
                 'TS4_DEM_R_T': exp_spex_thermistor,
                 'TS5_HOUSING_R_T': exp_spex_thermistor,
                 'TS6_RADIATOR_R_T': exp_spex_thermistor,
                 'ICU_5V_T': poly_spex_icuhk_internaltemp,
                 'ICU_4V_T': poly_spex_icuhk_internaltemp,
                 'ICU_HG1_T': poly_spex_icuhk_internaltemp,
                 'ICU_HG2_T': poly_spex_icuhk_internaltemp,
                 'ICU_MID_T': poly_spex_icuhk_internaltemp,
                 'ICU_MCU_T': poly_spex_icuhk_internaltemp,
                 'ICU_DIGV_T': poly_spex_icuhk_internaltemp,
                 'ICU_4p0V_V': poly_spex_icuhk_internaltemp2,
                 'ICU_3p3V_V': poly_spex_icuhk_internaltemp2,
                 'ICU_1p2V_V': poly_spex_icuhk_internaltemp2,
                 'ICU_5p0V_V': poly_spex_icuhk_internaltemp2,
                 'DEM_V': poly_spex_icuhk_internaltemp2,
                 'HTRGRP1_V': poly_spex_htr_v,
                 'HTRGRP2_V': poly_spex_htr_v,
                 'HTR1_DutyCycl': poly_spex_dutycycle,
                 'HTR2_DutyCycl': poly_spex_dutycycle,
                 'HTR3_DutyCycl': poly_spex_dutycycle,
                 'HTR4_DutyCycl': poly_spex_dutycycle,
                 'LED1_ANODE_V': poly_spex_led_anode_v,
                 'LED2_ANODE_V': poly_spex_led_anode_v,
                 'LED1_CATH_V': poly_spex_led_cath_v,
                 'LED2_CATH_V': poly_spex_led_cath_v,
                 'LED1_I': poly_spex_led_i,
                 'LED2_I': poly_spex_led_i,
                 'ADC1_VCC': poly_spex_adc_vcc,
                 'ADC1_REF': poly_spex_adc_vcc,
                 'ADC2_VCC': poly_spex_adc_vcc,
                 'ADC2_REF': poly_spex_adc_vcc,
                 'ADC1_GAIN': poly_spex_adc_gain,
                 'ADC2_GAIN': poly_spex_adc_gain,
                 'ADC1_T': poly_spex_adc_t,
                 'ADC2_T': poly_spex_adc_t,
                 'ADC1_OFFSET': poly_spex_adc_offset,
                 'ADC2_OFFSET': poly_spex_adc_offset,
                 'DEM_I': poly_spex_dem_i,
                 'DEM_T': exp_spex_det_t}

    func = conv_dict.get(key, None)
    if func is not None:
        return func(raw_data)

    return raw_data


# - class SPXtlm ----------------------------
class SPXtlm:
    """Access/convert parameters of SPEXone Science telemetry data.
    """
    def __init__(self):
        """Initialize class SPXtlm.
        """
        self.filename = None
        self._hdr = ()
        self.__tm = ()
        self._tlm = ()

    def show_hdr(self):
        print(self._hdr)

    def show_tm(self):
        print(self.__tm)

    def show_tlm(self):
        print(self._tlm)

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
        if apid is None and instrument == 'spx':
            apid = 0x320
            
        self.filename = flname
        hkt = HKTio(flname, instrument)
        res = hkt.housekeeping(apid)
        self._hdr = res['hdr'].copy()
        self._tlm = res['hk'].copy()
        self.timestamps(res['hdr']['tai_sec'],
                        res['hdr']['sub_sec'])

    def lv0_tlm(self, flname: Path, tlm_type: str | None = None):
        """Read telemetry dta from SPEXone Level-0 product.

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

        self.filename = flname
        res = read_lv0_data([flname], file_format='dsb', verbose=True)
        ccsds_sci, ccsds_hk = res
        print(ccsds_hk[0]['hdr'].dtype)
        print(ccsds_hk[0]['data']['hk'].dtype)
        print(ccsds_sci[0]['hdr'].dtype)
        print(ccsds_sci[0]['data']['hk'].dtype)

        self._hdr = ()
        self._tlm = ()
        self.__tm = ()
        tai_sec = ()
        sub_sec = ()
        if tlm_type == 'NomHK':
            for segment in ccsds_hk:
                self._hdr += (segment['hdr'],)
                self._tlm += (segment['data']['hk'],)
                tai_sec += (segment['hdr']['tai_sec'],)
                sub_sec += (segment['hdr']['sub_sec'],)
        elif tlm_type == 'DemHK':
            for segment in ccsds_sci:
                hdr = segment['hdr']
                if grouping_flag(hdr) != 1:
                    continue
                data = segment['data'][0]
                
                self._hdr += (hdr,)
                self._tlm += (data['hk'],)
                tai_sec += (segment['hdr']['tai_sec'],)
                sub_sec += (segment['hdr']['sub_sec'],)
        else:
            raise KeyError("tlm_type not in ['NomHK', 'DemHK']")
        self.timestamps(tai_sec, sub_sec)
                
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
        self.filename = flname
        self._hdr = ()
        self._tlm = ()
        self.__tm = ()

    def timestamps(self, tai_sec: tuple | np.ndarray,
                   sub_sec: tuple | np.ndarray):
        """Obtain timestamp of telemetry packages.

        Parameters
        ----------
        tai_sec :  tuple[int] | np.ndarray
        sub_sec :  tuple[int] | np.ndarray
        """
        if tai_sec[0] < 1956528000:
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        else:
            epoch = (datetime(1958, 1, 1, tzinfo=timezone.utc)
                     - timedelta(seconds=get_leap_seconds(tai_sec[0])))
        res = []
        for ii in range(len(tai_sec)):
            res.append(epoch + timedelta(
                seconds=int(tai_sec[ii]),
                microseconds=100 * int(sub_sec[ii] / 65536 * 10000)))

        self.__tm = res

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

    def units(self, key: str) -> str:
        """Obtain units of telemetry parameter when converted to physical units.

        Parameters
        ----------
        key :  str
           Name of telemetry parameter

        Returns
        -------
        str
        """
        return {'TS1_DEM_N_T': 'degree_Celsius',
                'TS2_HOUSING_N_T': 'degree_Celsius',
                'TS3_RADIATOR_N_T': 'degree_Celsius',
                'TS4_DEM_R_T': 'degree_Celsius',
                'TS5_HOUSING_R_T': 'degree_Celsius',
                'TS6_RADIATOR_R_T': 'degree_Celsius',
                'ICU_5V_T': 'degree_Celsius',
                'ICU_4V_T': 'degree_Celsius',
                'ICU_HG1_T': 'degree_Celsius',
                'ICU_HG2_T': 'degree_Celsius',
                'ICU_MID_T': 'degree_Celsius',
                'ICU_MCU_T': 'degree_Celsius',
                'ICU_DIGV_T': 'degree_Celsius',
                'ADC1_T': 'degree_Celsius',
                'ADC2_T': 'degree_Celsius',
                'DEM_T': 'degree_Celsius',
                'ICU_4p0V_V': 'Volt',
                'ICU_3p3V_V': 'Volt',
                'ICU_1p2V_V': 'Volt',
                'ICU_5p0V_V': 'Volt',
                'DEM_V': 'Volt',
                'HTRGRP1_V': 'Volt',
                'HTRGRP2_V': 'Volt',
                'LED1_ANODE_V': 'Volt',
                'LED2_ANODE_V': 'Volt',
                'LED1_CATH_V': 'Volt',
                'LED2_CATH_V': 'Volt',
                'ADC1_VCC': 'Volt',
                'ADC1_REF': 'Volt',
                'ADC2_VCC': 'Volt',
                'ADC2_REF': 'Volt',
                'ADC1_GAIN': 'Volt',
                'ADC2_GAIN': 'Volt',
                'ADC1_OFFSET': 'Volt',
                'ADC2_OFFSET': 'Volt',
                'ICU_4p0V_I': 'mA',
                'ICU_3p3V_I': 'mA',
                'ICU_1p2V_I': 'mA',
                'ICU_5p0V_I': 'mA',
                'DEM_I': 'mA',
                'HTR1_I': 'mA',
                'HTR2_I': 'mA',
                'HTR3_I': 'mA',
                'HTR4_I': 'mA',
                'LED1_I': 'mA',
                'LED2_I': 'mA',
                'HTR1_DutyCycl': '%',
                'HTR2_DutyCycl': '%',
                'HTR3_DutyCycl': '%',
                'HTR4_DutyCycl': '%'}.get(key, '1')


def __test():
    data_dir = Path('/data2/richardh/SPEXone/spx1_lv0/0x12d/2023/05/25')
    flname = data_dir / 'SPX000000880.spx'

    tlm = SPXtlm()
    tlm.lv0_tlm(flname, 'NomHK')
    # tlm.show_tlm()
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          tlm.units('TS2_HOUSING_N_T'))
    print(tlm.show_tm())
    tlm.lv0_tlm(flname, 'DemHK')
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          tlm.units('TS2_HOUSING_N_T'))
    print('ICU_5p0V_V', tlm.convert('ICU_5p0V_V'),
          tlm.units('ICU_5p0V_V'))
    print('REG_CMV_OUTPUTMODE', tlm.convert('REG_CMV_OUTPUTMODE'),
          tlm.units('REG_CMV_OUTPUTMODE'))
    print(tlm.show_tm())


def __test2():
    data_dir = Path('/data2/richardh/SPEXone/pace_hkt/V1.0/2023/05/25')
    if not data_dir.is_dir():
        data_dir = Path('/nfs/SPEXone/ocal/pace-sds/pace_hkt/V1.0/2023/05/25')
    flname = data_dir / 'PACE.20230525T043614.HKT.nc'
    tlm = SPXtlm()
    tlm.hkt_tlm(flname, instrument='spx', apid=0x320)
    tlm.show_hdr()
    # tlm.show_tlm()
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          tlm.units('TS2_HOUSING_N_T'))
    # print(tlm.show_tm())


if __name__ == '__main__':
    __test()
