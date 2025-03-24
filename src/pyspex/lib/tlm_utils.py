#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Contains helper functions for the class `SPXtlm`."""

from __future__ import annotations

__all__ = ["CONV_DICT", "HkFlagging", "convert_hk"]

from enum import Enum, IntFlag, auto
from typing import TYPE_CHECKING

import numpy as np
from numpy import ma

if TYPE_CHECKING:
    from collections.abc import Callable


# - helper functions ------------------------
def convert_2_float(raw_data: np.ndarray) -> np.ndarray:
    """Convert integer array to float."""
    return raw_data.astype(float)


def exp_spex_det_t(raw_data: np.ndarray) -> np.ndarray:
    """Convert Detector Temperature Sensor to degree Celsius."""
    res = np.empty(raw_data.size, dtype=float)
    mask = raw_data < 400
    res[mask] = 1.224 * raw_data[mask] - 290.2
    res[~mask] = 0.6426 * raw_data[~mask] - 418.72
    res[res > 50] = np.nan
    return res


def exp_spex_thermistor(raw_data: np.ndarray) -> np.ndarray:
    """Convert readouts of the Temperature Sensors to degree Celsius.

    Notes
    -----
    - TS1 DEM Nominal temperature
    - TS2 Housing Nominal Temperature
    - TS3 Radiator Nominal Temperature
    - TS4 DEM Redundant Temperature
    - TS5 Housing Redundant Temperature
    - TS6 Radiator Redundant Temperature*

    """
    coefficients = (21.19, 272589.0, 1.5173e-15, 5.73666e-19, 5.11328e-20)
    buff = ma.masked_array(raw_data / 256, mask=raw_data == 0)
    buff = (
        coefficients[0]
        + coefficients[1] / buff
        - coefficients[2] * buff**4
        + (coefficients[3] - coefficients[4] * ma.log(buff)) * buff**5
    )
    buff[raw_data == 0] = np.nan
    return buff.data


def poly_spex_icuhk_internaltemp(raw_data: np.ndarray) -> np.ndarray:
    """Convert readouts of temperature sensors on ICU power supplies to degree Celsius.

    Notes
    -----
    - ICU V5 supply temperature
    - ICU V4 supply temperature
    - ICU HtrG1 supply temperature
    - ICU HtrG2 supply temperature
    - ICU MidBoard temperature
    - ICU MCU-RAM temperature
    - ICU 1V2, 3V3 supply temperature

    """
    coefficients = (0.0, 0.0625)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_icuhk_internaltemp2(raw_data: np.ndarray) -> np.ndarray:
    """Convert readings of ICU bus voltages to Voltages.

    Notes
    -----
    - ICU 4.0 Volt bus voltage
    - ICU 3.3 Volt bus voltage
    - ICU 1.2 Volt bus voltage
    - DEM power supply

    """
    coefficients = (0, 0.001)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_htr_v(raw_data: np.ndarray) -> np.ndarray:
    """Convert Heater Bus voltages to Volt."""
    coefficients = (0, 0.01 / 3)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_htr_i(raw_data: np.ndarray) -> np.ndarray:
    """Convert readings of Heater Currents to mA."""
    coefficients = (0, 0.001)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_htr1_p(raw_data: np.ndarray) -> np.ndarray:
    """Convert Heater1 Current to Watt."""
    resistance = 238.3
    return resistance * (raw_data / 1000) ** 2


def poly_spex_htr2_p(raw_data: np.ndarray) -> np.ndarray:
    """Convert Heater2 Current to Watt."""
    resistance = 212.5
    return resistance * (raw_data / 1000) ** 2


def poly_spex_htr3_p(raw_data: np.ndarray) -> np.ndarray:
    """Convert Heater3 Current to Watt."""
    resistance = 237.7
    return resistance * (raw_data / 1000) ** 2


def poly_spex_htr4_p(raw_data: np.ndarray) -> np.ndarray:
    """Convert Heater4 Current to Watt."""
    resistance = 213.2
    return resistance * (raw_data / 1000) ** 2


def poly_spex_dutycycle(raw_data: np.ndarray) -> np.ndarray:
    """Convert Heater Controller Duty Cycle output to percent."""
    coefficients = (0, 0.1)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_led_anode_v(raw_data: np.ndarray) -> np.ndarray:
    """Convert LED Anode voltage to Volt."""
    coefficients = (0, 0.000000623703003)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_led_cath_v(raw_data: np.ndarray) -> np.ndarray:
    """Convert LED Cathode voltage to Volt."""
    coefficients = (0, 0.000000415802001953)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_led_i(raw_data: np.ndarray) -> np.ndarray:
    """Convert LED current to milli-Ampere."""
    coefficients = (0, 0.0000030307446495961334745762711864407)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_adc_vcc(raw_data: np.ndarray) -> np.ndarray:
    """Convert ADC Analog VCC reading to Volt."""
    coefficients = (0, 0.00000127156575520833333)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_adc_gain(raw_data: np.ndarray) -> np.ndarray:
    """Convert ADC Analog VCC reading to Volt."""
    coefficients = (0, 0.000000127156575520833333)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_adc_t(raw_data: np.ndarray) -> np.ndarray:
    """Convert ADC1 Temperature reading to degree Celsius."""
    coefficients = (-273.4, 0.0007385466842651367)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_adc_offset(raw_data: np.ndarray) -> np.ndarray:
    """Convert ADC offset (?) to Voltage."""
    coefficients = (0, 0.000415802001953)
    return coefficients[0] + coefficients[1] * raw_data


def poly_spex_dem_i(raw_data: np.ndarray) -> np.ndarray:
    """Convert DEM Supply current to milli-Ampere."""
    coefficients = (0, 0.2417)
    return coefficients[0] + coefficients[1] * raw_data


class Enable(Enum):
    """Possible values of the DEM SpaceWire link status."""

    INACTIVE = 0
    ACTIVE = 1


class EnaDis(Enum):
    """Possible values of the LED commanded enable state."""

    OFF = 0
    ON = 1


class IcuMode(Enum):
    """Possible values of the ICU mode."""

    IDLE = 0
    OPER = 1


class IsEna(Enum):
    """Possible values of the Power Sense."""

    DISABLED = 0
    ENABLED = 1


class OcdReg(Enum):
    """Possible values of the OCD state."""

    PWR_OK = 0
    OC2V1 = 1
    OC3V3 = 2
    OC2V1_AND_3V3 = 3


class TimeMsg(Enum):
    """Received status of external PPS or Time message."""

    MISSING = 0
    RECEIVED = 1


class WriteProt(Enum):
    """Status of Flash Write protection (both banks)."""

    BANK0_DISABLED = 0
    BANK0_ENABLED = 1
    BANK1_DISABLED = 16
    BANK1_ENABLED = 17


# - exported functions ----------------------
CONV_DICT: dict[
    str, dict[str | None, Callable[..., np.ndarray] | None, tuple[int] | None]
] = {
    "SEQCNT": {"units": None, "func": None, "range": None},
    "TCPKTID": {"units": None, "func": None, "range": None},
    "TCPKTSEQCTRL": {"units": None, "func": None, "range": None},
    "TCREJCODE": {"units": None, "func": None, "range": None},
    "TCFAILCODE": {"units": None, "func": None, "range": None},
    "TCREJPKTID": {"units": None, "func": None, "range": None},
    "TCFAILPKTID": {"units": None, "func": None, "range": None},
    "TCACCCNT": {"units": None, "func": None, "range": None},
    "TCREJCNT": {"units": None, "func": None, "range": None},
    "TCEXECCNT": {"units": None, "func": None, "range": None},
    "TCFAILCNT": {"units": None, "func": None, "range": None},
    "ICUSWVER": {"units": None, "func": None, "range": None},
    "SYSSTATE": {"units": None, "func": None, "range": None},
    "ICUMODE": {"units": None, "func": None, "range": None},
    "EXTPPSSTAT": {"units": None, "func": None, "range": None},
    "TIMEMSGSTAT": {"units": None, "func": None, "range": None},
    "OBTSYNCSTAT": {"units": None, "func": None, "range": None},
    "MPS_ID": {"units": None, "func": None, "range": None},
    "MPS_VER": {"units": None, "func": None, "range": None},
    "EVNTCNT_DEBUG": {"units": None, "func": None, "range": None},
    "EVNTCNT_PROG": {"units": None, "func": None, "range": None},
    "EVNTCNT_WARN": {"units": None, "func": None, "range": None},
    "EVNTCNT_ERR": {"units": None, "func": None, "range": None},
    "EVNTCNT_FATAL": {"units": None, "func": None, "range": None},
    "BOOTSTATEPREV": {"units": None, "func": None, "range": None},
    "BOOTCNTGOOD_IM0": {"units": None, "func": None, "range": None},
    "BOOTCNTGOOD_IM1": {"units": None, "func": None, "range": None},
    "BOOTCNTGOOD_IM2": {"units": None, "func": None, "range": None},
    "BOOTCNTGOOD_IM3": {"units": None, "func": None, "range": None},
    "BOOTATTEMPTS_CURRIM": {"units": None, "func": None, "range": None},
    "MPS_ACT_STATUS": {"units": None, "func": None, "range": None},
    "SWIMG_LOADED": {"units": None, "func": None, "range": None},
    "SWIMG_DEFAULT": {"units": None, "func": None, "range": None},
    "SWIMG_NXTBOOT": {"units": None, "func": None, "range": None},
    "WRITEPROT": {"units": None, "func": None, "range": None},
    "BOOTCAUSE": {"units": None, "func": None, "range": None},
    "TCVER_STAT": {"units": None, "func": None, "range": None},
    "SPW_REG_A": {"units": None, "func": None, "range": None},
    "SPW_REG_B": {"units": None, "func": None, "range": None},
    "LAST_CRC": {"units": None, "func": None, "range": None},
    "SCITM_PKTINTVL": {"units": None, "func": None, "range": None},
    "SCITM_BUFFREE": {"units": None, "func": None, "range": None},
    "SWEXECTIMEWC": {"units": None, "func": None, "range": None},
    "ERRCNT_PLACEHOLDER_03": {"units": None, "func": None, "range": None},
    "TS1_DEM_N_T": {
        "units": "degC",
        "func": exp_spex_thermistor,
        "range": (18.13, 18.53),
    },
    "TS2_HOUSING_N_T": {
        "units": "degC",
        "func": exp_spex_thermistor,
        "range": (19.51, 19.71),
    },
    "TS3_RADIATOR_N_T": {
        "units": "degC",
        "func": exp_spex_thermistor,
        "range": (-1.5, 2.0),
    },
    "TS4_DEM_R_T": {
        "units": "degC",
        "func": exp_spex_thermistor,
        "range": (17.7, 18.1),
    },
    "TS5_HOUSING_R_T": {
        "units": "degC",
        "func": exp_spex_thermistor,
        "range": (19.05, 19.55),
    },
    "TS6_RADIATOR_R_T": {
        "units": "degC",
        "func": exp_spex_thermistor,
        "range": (-1.5, 2.0),
    },
    "ICU_5V_T": {
        "units": "degC",
        "func": poly_spex_icuhk_internaltemp,
        "range": (-20, 60),
    },
    "ICU_4V_T": {
        "units": "degC",
        "func": poly_spex_icuhk_internaltemp,
        "range": (-20, 60),
    },
    "ICU_HG1_T": {
        "units": "degC",
        "func": poly_spex_icuhk_internaltemp,
        "range": (-20, 60),
    },
    "ICU_HG2_T": {
        "units": "degC",
        "func": poly_spex_icuhk_internaltemp,
        "range": (-20, 60),
    },
    "ICU_MID_T": {
        "units": "degC",
        "func": poly_spex_icuhk_internaltemp,
        "range": (-20, 60),
    },
    "ICU_MCU_T": {
        "units": "degC",
        "func": poly_spex_icuhk_internaltemp,
        "range": (-20, 60),
    },
    "ICU_DIGV_T": {
        "units": "degC",
        "func": poly_spex_icuhk_internaltemp,
        "range": (-20, 60),
    },
    "ICU_4P0V_V": {
        "units": "Volt",
        "func": poly_spex_icuhk_internaltemp2,
        "range": (3.86, 5.4),
    },
    "ICU_3P3V_V": {
        "units": "Volt",
        "func": poly_spex_icuhk_internaltemp2,
        "range": (3.1, 3.456),
    },
    "ICU_1P2V_V": {
        "units": "Volt",
        "func": poly_spex_icuhk_internaltemp2,
        "range": (1.21, 1.26),
    },
    "ICU_4P0V_I": {"units": "mA", "func": convert_2_float, "range": (100, 350)},
    "ICU_3P3V_I": {"units": "mA", "func": convert_2_float, "range": (100, 300)},
    "ICU_1P2V_I": {"units": "mA", "func": convert_2_float, "range": (70, 200)},
    "DEM_STATUS": {"units": None, "func": None, "range": None},
    "ICU_5P0V_V": {
        "units": "Volt",
        "func": poly_spex_icuhk_internaltemp2,
        "range": (4.85, 5.25),
    },
    "ICU_5P0V_I": {"units": "mA", "func": convert_2_float, "range": (5, 570)},
    "DEMSPWSTAT": {"units": None, "func": None, "range": None},
    "DEMRESETCNT": {"units": None, "func": None, "range": None},
    "HTRGRP1_V": {"units": "Volt", "func": poly_spex_htr_v, "range": (46, 49)},
    "HTRGRP2_V": {"units": "Volt", "func": poly_spex_htr_v, "range": (46, 49)},
    "HTR1_I": {"units": "mA", "func": poly_spex_htr_i, "range": (0, 250)},
    "HTR2_I": {"units": "mA", "func": poly_spex_htr_i, "range": (0, 250)},
    "HTR3_I": {"units": "mA", "func": poly_spex_htr_i, "range": (0, 250)},
    "HTR4_I": {"units": "mA", "func": poly_spex_htr_i, "range": (0, 250)},
    "HTR1_POWER": {"units": "Watt", "func": poly_spex_htr1_p, "range": (-1, 1)},
    "HTR2_POWER": {"units": "Watt", "func": poly_spex_htr2_p, "range": (0, 10)},
    "HTR3_POWER": {"units": "Watt", "func": poly_spex_htr3_p, "range": (0, 10)},
    "HTR4_POWER": {"units": "Watt", "func": poly_spex_htr4_p, "range": (-1, 1)},
    "HTR1_CALCPVAL": {"units": None, "func": None, "range": None},
    "HTR2_CALCPVAL": {"units": None, "func": None, "range": None},
    "HTR3_CALCPVAL": {"units": None, "func": None, "range": None},
    "HTR4_CALCPVAL": {"units": None, "func": None, "range": None},
    "HTR1_CALCIVAL": {"units": None, "func": None, "range": None},
    "HTR2_CALCIVAL": {"units": None, "func": None, "range": None},
    "HTR3_CALCIVAL": {"units": None, "func": None, "range": None},
    "HTR4_CALCIVAL": {"units": None, "func": None, "range": None},
    "HTR1_DUTYCYCL": {"units": "%", "func": poly_spex_dutycycle, "range": (-1, 1)},
    "HTR2_DUTYCYCL": {"units": "%", "func": poly_spex_dutycycle, "range": (6, 12)},
    "HTR3_DUTYCYCL": {"units": "%", "func": poly_spex_dutycycle, "range": (6, 12)},
    "HTR4_DUTYCYCL": {"units": "%", "func": poly_spex_dutycycle, "range": (-1, 1)},
    "LED1_ENADIS": {"units": None, "func": None, "range": None},
    "LED2_ENADIS": {"units": None, "func": None, "range": None},
    "LED1_ANODE_V": {
        "units": "Volt",
        "func": poly_spex_led_anode_v,
        "range": (2.9, 5.4),
    },
    "LED1_CATH_V": {"units": "Volt", "func": poly_spex_led_cath_v, "range": (0.7, 5.4)},
    "LED1_I": {"units": "mA", "func": poly_spex_led_i, "range": (0, 10.1)},
    "LED2_ANODE_V": {
        "units": "Volt",
        "func": poly_spex_led_anode_v,
        "range": (2.9, 5.4),
    },
    "LED2_CATH_V": {"units": "Volt", "func": poly_spex_led_cath_v, "range": (0.7, 5.4)},
    "LED2_I": {"units": "mA", "func": poly_spex_led_i, "range": (0, 10.1)},
    "ADC1_VCC": {"units": "Volt", "func": poly_spex_adc_vcc, "range": (4.77, 5.25)},
    "ADC1_GAIN": {"units": "Volt", "func": poly_spex_adc_gain, "range": None},
    "ADC1_REF": {"units": "Volt", "func": poly_spex_adc_vcc, "range": None},
    "ADC1_T": {"units": "degC", "func": poly_spex_adc_t, "range": (-20, 60)},
    "ADC1_OFFSET": {"units": "Volt", "func": poly_spex_adc_offset, "range": None},
    "ADC2_VCC": {"units": "Volt", "func": poly_spex_adc_vcc, "range": None},
    "ADC2_GAIN": {"units": "Volt", "func": poly_spex_adc_gain, "range": None},
    "ADC2_REF": {"units": "Volt", "func": poly_spex_adc_vcc, "range": (4.77, 5.25)},
    "ADC2_T": {"units": "degC", "func": poly_spex_adc_t, "range": (-20, 60)},
    "ADC2_OFFSET": {"units": "Volt", "func": poly_spex_adc_offset, "range": None},
    "DEM_V": {
        "units": "Volt",
        "func": poly_spex_icuhk_internaltemp2,
        "range": (4.7, 5.25),
    },
    "DEM_I": {"units": "mA", "func": poly_spex_dem_i, "range": (100, 550)},
    "REG_FW_VERSION": {"units": None, "func": None, "range": None},
    "DET_T": {"units": "degC", "func": exp_spex_det_t, "range": (19, 36)},
    "REG_SPW_ERROR": {"units": None, "func": None, "range": None},
    "REG_CMV_OUTOFSYNC": {"units": None, "func": None, "range": None},
    "REG_OCD_ACTUAL": {"units": None, "func": None, "range": None},
    "REG_OCD_STICKY": {"units": None, "func": None, "range": None},
    "REG_PWR_SENS": {"units": None, "func": None, "range": None},
    "REG_FLASH_STATUS": {"units": None, "func": None, "range": None},
    "REG_FLASH_EDAC_BLOCK": {"units": None, "func": None, "range": None},
    "SW_MAIN_LOOP_COUNT": {"units": None, "func": None, "range": None},
}


def convert_hk(parm: str, raw_data: np.ndarray) -> np.ndarray:
    """Convert a NomHK (or DemHK subset) parameter to physical units."""
    if (res := CONV_DICT.get(parm)) is None:
        raise KeyError(f"Parameter: {parm} not found in CONV_DICT")

    if (func := res["func"]) is not None:
        return func(raw_data)

    return raw_data


class HkFlagging(IntFlag):
    """..."""

    NOMINAL = 0
    CURRENT_TOO_LOW = auto()
    CURRENT_TOO_HIGH = auto()
    TEMP_TOO_LOW = auto()
    TEMP_TOO_HIGH = auto()
    VOLT_TOO_LOW = auto()
    VOLT_TOO_HIGH = auto()
    WATT_TOO_LOW = auto()
    WATT_TOO_HIGH = auto()
    CHANGED = auto()

    @classmethod
    def get_flag(cls: HkFlagging, parm: str, too_low: bool = False) -> HkFlagging:
        """Get value of flag for parameter out-of-range or changed parameter-values."""
        if (res := CONV_DICT.get(parm)) is None:
            raise KeyError(f"Parameter: {parm} not found in CONV_DICT")

        match res["units"]:
            case "mA" | "A":
                value = cls.CURRENT_TOO_LOW if too_low else cls.CURRENT_TOO_HIGH
            case "degC" | "K":
                value = cls.TEMP_TOO_LOW if too_low else cls.TEMP_TOO_HIGH
            case "Volt":
                value = cls.VOLT_TOO_LOW if too_low else cls.VOLT_TOO_HIGH
            case "Watt":
                value = cls.WATT_TOO_LOW if too_low else cls.WATT_TOO_HIGH
            case _:
                value = cls.CHANGED

        return value
