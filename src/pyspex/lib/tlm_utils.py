#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2024 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Contains helper functions for the class `SPXtlm`."""

from __future__ import annotations

__all__ = ["UNITS_DICT", "convert_hk", "HkFlagging"]

from enum import Enum, IntFlag, auto

import numpy as np
from numpy import ma

# This dictionary is only valid when raw counts are converted to physical units
UNITS_DICT = {
    "ADC1_GAIN": "Volt",
    "ADC1_OFFSET": "Volt",
    "ADC1_REF": "Volt",
    "ADC1_T": "degC",
    "ADC1_VCC": "Volt",
    "ADC2_GAIN": "Volt",
    "ADC2_OFFSET": "Volt",
    "ADC2_REF": "Volt",
    "ADC2_T": "degC",
    "ADC2_VCC": "Volt",
    "DEM_I": "mA",
    "DEM_V": "Volt",
    "DET_T": "degC",
    "HTR1_DUTYCYCL": "%",
    "HTR1_I": "mA",
    "HTR2_DUTYCYCL": "%",
    "HTR2_I": "mA",
    "HTR3_DUTYCYCL": "%",
    "HTR3_I": "mA",
    "HTR4_DUTYCYCL": "%",
    "HTR4_I": "mA",
    "HTRGRP1_V": "Volt",
    "HTRGRP2_V": "Volt",
    "ICU_1P2V_I": "mA",
    "ICU_1P2V_V": "Volt",
    "ICU_3P3V_I": "mA",
    "ICU_3P3V_V": "Volt",
    "ICU_4P0V_I": "mA",
    "ICU_4P0V_V": "Volt",
    "ICU_4V_T": "degC",
    "ICU_5P0V_I": "mA",
    "ICU_5P0V_V": "Volt",
    "ICU_5V_T": "degC",
    "ICU_DIGV_T": "degC",
    "ICU_HG1_T": "degC",
    "ICU_HG2_T": "degC",
    "ICU_MCU_T": "degC",
    "ICU_MID_T": "degC",
    "LED1_ANODE_V": "Volt",
    "LED1_CATH_V": "Volt",
    "LED1_I": "mA",
    "LED2_ANODE_V": "Volt",
    "LED2_CATH_V": "Volt",
    "LED2_I": "mA",
    "TS1_DEM_N_T": "degC",
    "TS2_HOUSING_N_T": "degC",
    "TS3_RADIATOR_N_T": "degC",
    "TS4_DEM_R_T": "degC",
    "TS5_HOUSING_R_T": "degC",
    "TS6_RADIATOR_R_T": "degC",
}


RANGE_DICT = {
    "ADC1_T": (-20, 60),
    "ADC1_VCC": (4.77, 5.25),
    "ADC2_T": (-20, 60),
    "DEM_I": (100, 550),
    "DEM_V": (4.7, 5.25),
    "DET_T": (19, 36),
    "HTR1_I": (0, 250),
    "HTR2_I": (0, 250),
    "HTR3_I": (0, 250),
    "HTR4_I": (0, 250),
    "ICU_1P2V_I": (70, 200),
    "ICU_1P2V_V": (1.21, 1.26),
    "ICU_3P3V_I": (100, 300),
    "ICU_3P3V_V": (3.1, 3.456),
    "ICU_4P0V_I": (100, 350),
    "ICU_4P0V_V": (3.86, 5.4),
    "ICU_4V_T": (-20, 60),
    "ICU_5P0V_I": (5, 570),
    "ICU_5P0V_V": (4.85, 5.25),
    "ICU_5V_T": (-20, 60),
    "ICU_DIGV_T": (-20, 60),
    "ICU_HG1_T": (-20, 60),
    "ICU_HG2_T": (-20, 60),
    "ICU_MCU_T": (-20, 60),
    "ICU_MID_T": (-20, 60),
    "LED1_ANODE_V": (2.9, 5.4),
    "LED1_CATH_V": (0.7, 5.4),
    "LED1_I": (0, 10.1),
    "LED2_ANODE_V": (2.9, 5.4),
    "LED2_CATH_V": (0.7, 5.4),
    "LED2_I": (0, 10.1),
    "TS1_DEM_N_T": (18.13, 18.53),
    "TS2_HOUSING_N_T": (19.51, 19.71),
    "TS3_RADIATOR_N_T": (-0.9, 2.1),
    "TS4_DEM_R_T": (17.7, 18.1),
    "TS5_HOUSING_R_T": (19.05, 19.55),
    "TS6_RADIATOR_R_T": (-0.9, 2.1),
}


# - helper functions ------------------------
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
def convert_hk(key: str, raw_data: np.ndarray) -> np.ndarray:
    """Convert a DemHK or NomHK parameter to physical units."""
    conv_dict = {
        "DET_T": exp_spex_det_t,
        "TS1_DEM_N_T": exp_spex_thermistor,
        "TS2_HOUSING_N_T": exp_spex_thermistor,
        "TS3_RADIATOR_N_T": exp_spex_thermistor,
        "TS4_DEM_R_T": exp_spex_thermistor,
        "TS5_HOUSING_R_T": exp_spex_thermistor,
        "TS6_RADIATOR_R_T": exp_spex_thermistor,
        "ADC1_GAIN": poly_spex_adc_gain,
        "ADC2_GAIN": poly_spex_adc_gain,
        "ADC1_OFFSET": poly_spex_adc_offset,
        "ADC2_OFFSET": poly_spex_adc_offset,
        "ADC1_T": poly_spex_adc_t,
        "ADC2_T": poly_spex_adc_t,
        "ADC1_REF": poly_spex_adc_vcc,
        "ADC1_VCC": poly_spex_adc_vcc,
        "ADC2_REF": poly_spex_adc_vcc,
        "ADC2_VCC": poly_spex_adc_vcc,
        "DEM_I": poly_spex_dem_i,
        "HTR1_DUTYCYCL": poly_spex_dutycycle,
        "HTR2_DUTYCYCL": poly_spex_dutycycle,
        "HTR3_DUTYCYCL": poly_spex_dutycycle,
        "HTR4_DUTYCYCL": poly_spex_dutycycle,
        "HTR1_POWER": poly_spex_htr1_p,
        "HTR2_POWER": poly_spex_htr2_p,
        "HTR3_POWER": poly_spex_htr3_p,
        "HTR4_POWER": poly_spex_htr4_p,
        "HTRGRP1_V": poly_spex_htr_v,
        "HTRGRP2_V": poly_spex_htr_v,
        "ICU_4V_T": poly_spex_icuhk_internaltemp,
        "ICU_5V_T": poly_spex_icuhk_internaltemp,
        "ICU_DIGV_T": poly_spex_icuhk_internaltemp,
        "ICU_HG1_T": poly_spex_icuhk_internaltemp,
        "ICU_HG2_T": poly_spex_icuhk_internaltemp,
        "ICU_MCU_T": poly_spex_icuhk_internaltemp,
        "ICU_MID_T": poly_spex_icuhk_internaltemp,
        "DEM_V": poly_spex_icuhk_internaltemp2,
        "ICU_1P2V_V": poly_spex_icuhk_internaltemp2,
        "ICU_3P3V_V": poly_spex_icuhk_internaltemp2,
        "ICU_4P0V_V": poly_spex_icuhk_internaltemp2,
        "ICU_5P0V_V": poly_spex_icuhk_internaltemp2,
        "LED1_ANODE_V": poly_spex_led_anode_v,
        "LED2_ANODE_V": poly_spex_led_anode_v,
        "LED1_CATH_V": poly_spex_led_cath_v,
        "LED2_CATH_V": poly_spex_led_cath_v,
        "LED1_I": poly_spex_led_i,
        "LED2_I": poly_spex_led_i,
    }

    func = conv_dict.get(key)
    if func is not None:
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
    STATUS_CHANGED = auto()

    @classmethod
    def get_flag(cls: HkFlagging, key: str, too_low: bool = False) -> HkFlagging:
        """..."""
        match UNITS_DICT.get(key):
            case "mA" | "A":
                value = cls.CURRENT_TOO_LOW if too_low else cls.CURRENT_TOO_HIGH
            case "degC" | "K":
                value = cls.TEMP_TOO_LOW if too_low else cls.TEMP_TOO_HIGH
            case "Volt":
                value = cls.VOLT_TOO_LOW if too_low else cls.VOLT_TOO_HIGH
            case _:
                value = cls.STATUS_CHANGED

        return value
