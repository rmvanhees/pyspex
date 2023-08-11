#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Defines the Helios spectrum, used at SRON."""
__all__ = ['helios_spectrum']

import numpy as np
import xarray as xr

# - global parameters ------------------------------
HELIOS_ATTRS = {'source': 'Helios',
                'setup.Port': 'A',
                'setup.Lamp': 'HES-150',
                'setup.Output': '100%',
                'setup.Date': '03-Sept-2019',
                'lamp.Port': 'A',
                'lamp.Output': '100%',
                'lamp.Voltage': '19.6177 V',
                'lamp.Current': '6.056 A',
                'lamp.CCT': '2998 K',
                'lamp.Luminance': '14070 Cd/m^2',
                'lamp.Illuminance': '44200 lux',
                'lamp.Luminance_uncertainty': '156700 Cd/m^2',
                'lamp.Luminance_relative_uncertainty': '1.114%',
                'detector.Port': 'Z',
                'detector.Open': '3.711E-4 A',
                'detector.Pinhole': '3.662E-6 A',
                'detector.Filter': '2.899E-5 A',
                'calib.Port': 'Z',
                'calib.Open': '3.793E+7 Cd/(m^2 A)',
                'calib.Filter': '4.855E+8 Cd/(m^2 A)',
                'calib.Pinhole': '3.843E+9 Cd/(m^2 A)'}

HELIOS_SPECTRUM = [
    4.92E+00, 5.12E+00, 5.33E+00, 5.55E+00, 5.76E+00, 5.98E+00,
    6.20E+00, 6.42E+00, 6.64E+00, 6.88E+00, 7.14E+00, 7.39E+00,
    7.64E+00, 7.89E+00, 8.14E+00, 8.39E+00, 8.64E+00, 8.90E+00,
    9.16E+00, 9.44E+00, 9.72E+00, 1.00E+01, 1.03E+01, 1.05E+01,
    1.08E+01, 1.11E+01, 1.14E+01, 1.17E+01, 1.20E+01, 1.24E+01,
    1.27E+01, 1.31E+01, 1.36E+01, 1.40E+01, 1.45E+01, 1.50E+01,
    1.55E+01, 1.60E+01, 1.65E+01, 1.70E+01, 1.75E+01, 1.80E+01,
    1.84E+01, 1.89E+01, 1.94E+01, 1.99E+01, 2.03E+01, 2.09E+01,
    2.14E+01, 2.20E+01, 2.26E+01, 2.32E+01, 2.39E+01, 2.46E+01,
    2.55E+01, 2.63E+01, 2.71E+01, 2.78E+01, 2.86E+01, 2.94E+01,
    3.03E+01, 3.12E+01, 3.21E+01, 3.30E+01, 3.39E+01, 3.47E+01,
    3.56E+01, 3.64E+01, 3.73E+01, 3.81E+01, 3.88E+01, 3.96E+01,
    4.04E+01, 4.11E+01, 4.18E+01, 4.25E+01, 4.32E+01, 4.40E+01,
    4.47E+01, 4.54E+01, 4.62E+01, 4.71E+01, 4.80E+01, 4.88E+01,
    4.95E+01, 5.02E+01, 5.09E+01, 5.16E+01, 5.25E+01, 5.34E+01,
    5.43E+01, 5.51E+01, 5.60E+01, 5.69E+01, 5.77E+01, 5.86E+01,
    5.95E+01, 6.05E+01, 6.14E+01, 6.24E+01, 6.34E+01, 6.44E+01,
    6.54E+01, 6.65E+01, 6.76E+01, 6.87E+01, 6.97E+01, 7.08E+01,
    7.20E+01, 7.31E+01, 7.43E+01, 7.55E+01, 7.66E+01, 7.77E+01,
    7.88E+01, 7.99E+01, 8.10E+01, 8.21E+01, 8.32E+01, 8.44E+01,
    8.55E+01, 8.66E+01, 8.77E+01, 8.89E+01, 9.00E+01, 9.11E+01,
    9.23E+01, 9.35E+01, 9.47E+01, 9.59E+01, 9.71E+01, 9.83E+01,
    9.95E+01, 1.01E+02, 1.02E+02, 1.03E+02, 1.04E+02, 1.06E+02,
    1.07E+02, 1.08E+02, 1.09E+02, 1.10E+02, 1.12E+02, 1.13E+02,
    1.14E+02, 1.15E+02, 1.16E+02, 1.18E+02, 1.19E+02, 1.20E+02,
    1.21E+02, 1.22E+02, 1.24E+02, 1.25E+02, 1.26E+02, 1.27E+02,
    1.29E+02, 1.30E+02, 1.31E+02, 1.32E+02, 1.34E+02, 1.35E+02,
    1.36E+02, 1.37E+02, 1.38E+02, 1.39E+02, 1.41E+02, 1.42E+02,
    1.43E+02, 1.44E+02, 1.45E+02, 1.47E+02, 1.48E+02, 1.49E+02,
    1.50E+02, 1.51E+02, 1.53E+02, 1.54E+02, 1.55E+02, 1.56E+02,
    1.58E+02, 1.59E+02, 1.60E+02, 1.61E+02, 1.62E+02, 1.64E+02,
    1.65E+02, 1.66E+02, 1.67E+02, 1.68E+02, 1.70E+02, 1.71E+02,
    1.72E+02, 1.73E+02, 1.75E+02, 1.76E+02, 1.77E+02, 1.78E+02,
    1.80E+02, 1.81E+02, 1.82E+02, 1.83E+02, 1.84E+02, 1.86E+02,
    1.87E+02, 1.88E+02, 1.90E+02, 1.91E+02, 1.92E+02, 1.93E+02,
    1.95E+02, 1.96E+02, 1.97E+02, 1.98E+02, 1.99E+02, 2.00E+02,
    2.01E+02, 2.03E+02, 2.04E+02, 2.05E+02, 2.06E+02, 2.07E+02,
    2.08E+02, 2.09E+02, 2.11E+02, 2.12E+02, 2.13E+02, 2.14E+02,
    2.15E+02, 2.16E+02, 2.17E+02, 2.19E+02, 2.20E+02, 2.21E+02,
    2.22E+02, 2.23E+02, 2.24E+02, 2.26E+02, 2.27E+02, 2.28E+02,
    2.29E+02, 2.30E+02, 2.31E+02, 2.32E+02, 2.34E+02, 2.35E+02,
    2.36E+02, 2.37E+02, 2.38E+02, 2.39E+02, 2.40E+02, 2.42E+02,
    2.43E+02, 2.44E+02, 2.45E+02, 2.46E+02, 2.47E+02, 2.48E+02,
    2.49E+02, 2.50E+02, 2.51E+02, 2.52E+02, 2.53E+02, 2.54E+02,
    2.55E+02, 2.56E+02, 2.57E+02, 2.58E+02, 2.58E+02, 2.59E+02,
    2.60E+02, 2.61E+02, 2.62E+02, 2.63E+02, 2.64E+02, 2.65E+02,
    2.66E+02, 2.67E+02, 2.69E+02, 2.70E+02, 2.71E+02, 2.72E+02,
    2.73E+02, 2.74E+02, 2.75E+02, 2.76E+02, 2.76E+02, 2.77E+02,
    2.78E+02, 2.79E+02, 2.80E+02, 2.81E+02, 2.81E+02, 2.82E+02,
    2.83E+02, 2.84E+02, 2.85E+02, 2.86E+02, 2.87E+02, 2.88E+02,
    2.88E+02, 2.89E+02, 2.90E+02, 2.91E+02, 2.92E+02, 2.93E+02,
    2.94E+02, 2.94E+02, 2.95E+02, 2.96E+02, 2.97E+02, 2.98E+02,
    2.98E+02, 2.99E+02, 3.00E+02, 3.00E+02, 3.01E+02, 3.02E+02,
    3.02E+02, 3.03E+02, 3.04E+02, 3.05E+02, 3.06E+02, 3.06E+02,
    3.07E+02, 3.08E+02, 3.09E+02, 3.10E+02, 3.10E+02, 3.11E+02,
    3.12E+02, 3.12E+02, 3.13E+02, 3.13E+02, 3.14E+02, 3.15E+02,
    3.15E+02, 3.16E+02, 3.17E+02, 3.17E+02, 3.18E+02, 3.19E+02,
    3.19E+02, 3.20E+02, 3.20E+02, 3.21E+02, 3.22E+02, 3.22E+02,
    3.23E+02, 3.23E+02, 3.24E+02, 3.24E+02, 3.25E+02, 3.26E+02,
    3.26E+02, 3.27E+02, 3.27E+02, 3.28E+02, 3.28E+02, 3.28E+02,
    3.29E+02, 3.29E+02, 3.30E+02, 3.30E+02, 3.30E+02, 3.31E+02,
    3.31E+02, 3.31E+02, 3.32E+02, 3.32E+02, 3.32E+02, 3.32E+02,
    3.33E+02, 3.33E+02, 3.33E+02, 3.33E+02, 3.33E+02, 3.34E+02,
    3.34E+02, 3.34E+02, 3.34E+02, 3.34E+02, 3.34E+02, 3.34E+02,
    3.35E+02, 3.35E+02, 3.35E+02, 3.35E+02, 3.35E+02, 3.35E+02,
    3.35E+02, 3.35E+02, 3.35E+02, 3.35E+02, 3.34E+02, 3.34E+02,
    3.34E+02, 3.34E+02, 3.34E+02, 3.34E+02, 3.34E+02, 3.34E+02,
    3.34E+02, 3.33E+02, 3.33E+02, 3.33E+02, 3.33E+02, 3.32E+02,
    3.32E+02, 3.32E+02, 3.31E+02, 3.31E+02, 3.30E+02, 3.30E+02,
    3.29E+02, 3.29E+02, 3.28E+02, 3.28E+02, 3.27E+02, 3.27E+02,
    3.26E+02, 3.26E+02, 3.25E+02, 3.24E+02, 3.24E+02, 3.23E+02,
    3.22E+02, 3.22E+02, 3.21E+02, 3.20E+02, 3.19E+02, 3.19E+02,
    3.18E+02, 3.17E+02, 3.16E+02, 3.16E+02, 3.15E+02, 3.14E+02,
    3.13E+02, 3.12E+02, 3.11E+02, 3.11E+02, 3.10E+02, 3.09E+02,
    3.08E+02, 3.07E+02, 3.06E+02, 3.05E+02, 3.04E+02, 3.03E+02,
    3.02E+02, 3.01E+02, 3.01E+02, 3.00E+02, 2.99E+02, 2.98E+02,
    2.97E+02, 2.96E+02, 2.95E+02, 2.94E+02, 2.94E+02, 2.93E+02,
    2.92E+02, 2.91E+02, 2.90E+02, 2.89E+02, 2.89E+02, 2.88E+02,
    2.87E+02, 2.86E+02, 2.86E+02, 2.85E+02, 2.85E+02, 2.84E+02,
    2.84E+02, 2.83E+02, 2.82E+02, 2.82E+02, 2.82E+02, 2.81E+02,
    2.81E+02, 2.80E+02, 2.80E+02, 2.79E+02, 2.79E+02, 2.79E+02,
    2.78E+02, 2.78E+02, 2.78E+02, 2.77E+02, 2.77E+02, 2.77E+02,
    2.77E+02, 2.77E+02, 2.77E+02, 2.77E+02, 2.77E+02, 2.77E+02,
    2.77E+02, 2.77E+02, 2.77E+02, 2.77E+02, 2.77E+02, 2.77E+02,
    2.77E+02, 2.77E+02, 2.77E+02, 2.77E+02, 2.78E+02, 2.78E+02,
    2.78E+02, 2.79E+02, 2.79E+02, 2.80E+02, 2.80E+02, 2.81E+02,
    2.82E+02, 2.82E+02, 2.83E+02, 2.84E+02, 2.84E+02, 2.85E+02,
    2.85E+02, 2.86E+02, 2.86E+02, 2.87E+02, 2.87E+02, 2.88E+02,
    2.89E+02, 2.89E+02, 2.90E+02, 2.91E+02, 2.92E+02, 2.92E+02,
    2.93E+02, 2.94E+02, 2.95E+02, 2.96E+02, 2.96E+02, 2.97E+02,
    2.98E+02, 2.99E+02, 3.00E+02, 3.00E+02, 3.01E+02, 3.02E+02,
    3.03E+02, 3.04E+02, 3.05E+02, 3.06E+02, 3.07E+02, 3.08E+02,
    3.09E+02, 3.10E+02, 3.11E+02, 3.11E+02, 3.12E+02, 3.13E+02,
    3.14E+02, 3.15E+02, 3.16E+02, 3.17E+02, 3.18E+02, 3.19E+02,
    3.20E+02, 3.21E+02, 3.22E+02, 3.23E+02, 3.24E+02, 3.25E+02,
    3.26E+02, 3.27E+02, 3.28E+02, 3.29E+02, 3.30E+02, 3.30E+02,
    3.31E+02, 3.32E+02, 3.33E+02, 3.34E+02, 3.34E+02, 3.35E+02,
    3.36E+02, 3.36E+02, 3.37E+02, 3.38E+02, 3.38E+02, 3.39E+02,
    3.40E+02, 3.41E+02, 3.41E+02, 3.42E+02, 3.43E+02, 3.43E+02,
    3.44E+02, 3.45E+02, 3.46E+02, 3.47E+02, 3.47E+02, 3.48E+02,
    3.49E+02, 3.49E+02, 3.50E+02, 3.50E+02, 3.51E+02, 3.52E+02,
    3.52E+02, 3.53E+02, 3.53E+02, 3.54E+02, 3.54E+02, 3.55E+02,
    3.55E+02, 3.56E+02, 3.56E+02, 3.57E+02, 3.57E+02, 3.58E+02,
    3.58E+02, 3.59E+02, 3.59E+02, 3.59E+02, 3.60E+02, 3.60E+02,
    3.61E+02, 3.61E+02, 3.62E+02, 3.62E+02, 3.62E+02, 3.63E+02,
    3.63E+02, 3.64E+02, 3.63E+02, 3.64E+02, 3.64E+02, 3.64E+02,
    3.64E+02, 3.64E+02, 3.63E+02, 3.64E+02, 3.64E+02, 3.64E+02,
    3.64E+02, 3.64E+02, 3.64E+02, 3.65E+02, 3.65E+02, 3.66E+02,
    3.66E+02, 3.67E+02, 3.67E+02, 3.67E+02, 3.68E+02, 3.68E+02,
    3.68E+02, 3.69E+02, 3.69E+02, 3.69E+02, 3.70E+02, 3.70E+02,
    3.70E+02, 3.71E+02, 3.71E+02, 3.71E+02, 3.72E+02, 3.72E+02,
    3.72E+02, 3.73E+02, 3.73E+02, 3.73E+02, 3.74E+02, 3.74E+02,
    3.74E+02, 3.74E+02, 3.74E+02, 3.74E+02, 3.75E+02, 3.75E+02,
    3.75E+02, 3.75E+02, 3.75E+02, 3.75E+02, 3.76E+02, 3.76E+02,
    3.76E+02, 3.76E+02, 3.76E+02, 3.76E+02, 3.77E+02, 3.77E+02,
    3.77E+02, 3.77E+02, 3.77E+02, 3.77E+02, 3.77E+02, 3.77E+02,
    3.77E+02, 3.77E+02, 3.77E+02, 3.77E+02, 3.77E+02, 3.76E+02,
    3.76E+02, 3.75E+02, 3.74E+02, 3.73E+02, 3.72E+02, 3.71E+02,
    3.70E+02, 3.70E+02, 3.69E+02, 3.69E+02, 3.69E+02, 3.69E+02,
    3.69E+02, 3.69E+02, 3.70E+02, 3.70E+02, 3.70E+02, 3.70E+02,
    3.70E+02, 3.70E+02, 3.70E+02, 3.70E+02, 3.70E+02, 3.70E+02,
    3.70E+02, 3.70E+02, 3.69E+02, 3.69E+02, 3.69E+02, 3.69E+02,
    3.69E+02, 3.69E+02, 3.69E+02, 3.69E+02, 3.69E+02, 3.69E+02,
    3.69E+02, 3.69E+02, 3.69E+02, 3.69E+02, 3.69E+02, 3.69E+02,
    3.68E+02, 3.68E+02, 3.68E+02, 3.67E+02, 3.67E+02, 3.66E+02,
    3.66E+02, 3.66E+02, 3.65E+02, 3.65E+02, 3.65E+02, 3.65E+02,
    3.64E+02, 3.64E+02, 3.63E+02, 3.63E+02, 3.63E+02, 3.62E+02,
    3.61E+02, 3.61E+02, 3.60E+02, 3.59E+02, 3.58E+02, 3.57E+02,
    3.56E+02, 3.56E+02, 3.55E+02, 3.55E+02, 3.54E+02, 3.54E+02,
    3.54E+02, 3.53E+02, 3.53E+02, 3.53E+02, 3.52E+02, 3.52E+02,
    3.52E+02, 3.51E+02, 3.50E+02, 3.49E+02, 3.49E+02, 3.48E+02,
    3.47E+02, 3.46E+02, 3.45E+02, 3.44E+02, 3.43E+02, 3.42E+02,
    3.41E+02, 3.40E+02, 3.39E+02, 3.38E+02, 3.37E+02, 3.36E+02,
    3.35E+02, 3.34E+02, 3.34E+02, 3.33E+02, 3.32E+02, 3.32E+02,
    3.31E+02, 3.31E+02, 3.31E+02, 3.30E+02, 3.30E+02, 3.29E+02,
    3.29E+02, 3.28E+02, 3.27E+02, 3.27E+02, 3.26E+02, 3.25E+02,
    3.24E+02, 3.24E+02, 3.23E+02, 3.22E+02, 3.22E+02, 3.21E+02,
    3.21E+02, 3.20E+02, 3.20E+02, 3.19E+02, 3.19E+02, 3.18E+02,
    3.18E+02, 3.17E+02, 3.16E+02, 3.15E+02, 3.15E+02, 3.14E+02,
    3.13E+02, 3.12E+02, 3.11E+02, 3.10E+02, 3.09E+02, 3.08E+02,
    3.07E+02, 3.06E+02, 3.05E+02, 3.04E+02, 3.04E+02, 3.03E+02,
    3.03E+02, 3.02E+02, 3.02E+02, 3.01E+02, 3.01E+02, 3.01E+02,
    3.00E+02, 3.00E+02, 3.00E+02, 3.00E+02, 2.99E+02, 2.99E+02,
    2.99E+02, 2.99E+02, 2.99E+02, 2.99E+02, 2.98E+02, 2.98E+02,
    2.98E+02, 2.97E+02, 2.97E+02, 2.96E+02, 2.96E+02, 2.96E+02,
    2.95E+02, 2.95E+02, 2.95E+02, 2.95E+02, 2.95E+02, 2.95E+02,
    2.96E+02, 2.96E+02, 2.96E+02, 2.96E+02, 2.96E+02, 2.96E+02,
    2.96E+02, 2.96E+02, 2.96E+02, 2.96E+02, 2.95E+02, 2.95E+02,
    2.95E+02, 2.95E+02, 2.95E+02, 2.94E+02, 2.94E+02, 2.94E+02,
    2.93E+02, 2.93E+02, 2.92E+02, 2.92E+02, 2.92E+02, 2.91E+02,
    2.91E+02, 2.91E+02, 2.91E+02, 2.90E+02, 2.90E+02, 2.90E+02,
    2.90E+02, 2.90E+02, 2.90E+02, 2.89E+02, 2.89E+02, 2.89E+02,
    2.89E+02, 2.89E+02, 2.88E+02, 2.88E+02, 2.88E+02, 2.88E+02,
    2.87E+02, 2.87E+02, 2.87E+02, 2.86E+02, 2.86E+02, 2.85E+02,
    2.85E+02, 2.84E+02, 2.84E+02, 2.83E+02, 2.83E+02, 2.83E+02,
    2.82E+02, 2.82E+02, 2.82E+02, 2.82E+02, 2.82E+02, 2.82E+02,
    2.82E+02, 2.82E+02, 2.82E+02, 2.82E+02, 2.81E+02, 2.81E+02,
    2.81E+02, 2.80E+02, 2.79E+02, 2.79E+02, 2.78E+02, 2.77E+02,
    2.77E+02, 2.76E+02, 2.75E+02, 2.75E+02, 2.75E+02, 2.74E+02,
    2.74E+02, 2.74E+02, 2.74E+02, 2.73E+02, 2.73E+02, 2.73E+02,
    2.72E+02, 2.72E+02, 2.71E+02, 2.71E+02, 2.70E+02, 2.69E+02,
    2.69E+02, 2.68E+02, 2.67E+02, 2.67E+02, 2.66E+02, 2.65E+02,
    2.64E+02, 2.63E+02, 2.63E+02, 2.62E+02, 2.61E+02, 2.60E+02,
    2.59E+02, 2.58E+02, 2.57E+02, 2.57E+02, 2.56E+02, 2.55E+02,
    2.55E+02, 2.54E+02, 2.53E+02, 2.53E+02, 2.52E+02, 2.51E+02,
    2.50E+02, 2.49E+02, 2.48E+02, 2.47E+02, 2.46E+02, 2.45E+02,
    2.43E+02, 2.42E+02, 2.40E+02, 2.39E+02, 2.37E+02, 2.35E+02,
    2.34E+02, 2.31E+02, 2.29E+02, 2.27E+02, 2.25E+02, 2.23E+02,
    2.21E+02, 2.20E+02, 2.19E+02, 2.18E+02, 2.17E+02, 2.16E+02,
    2.16E+02, 2.16E+02, 2.16E+02, 2.16E+02, 2.16E+02, 2.15E+02,
    2.15E+02, 2.14E+02, 2.13E+02, 2.13E+02, 2.12E+02, 2.11E+02,
    2.09E+02, 2.08E+02, 2.07E+02, 2.06E+02, 2.05E+02, 2.03E+02,
    2.02E+02, 2.01E+02, 2.00E+02, 1.98E+02, 1.97E+02, 1.95E+02,
    1.94E+02, 1.92E+02, 1.90E+02, 1.89E+02, 1.87E+02, 1.85E+02,
    1.83E+02, 1.82E+02, 1.80E+02, 1.78E+02, 1.76E+02, 1.75E+02,
    1.73E+02, 1.71E+02, 1.69E+02, 1.67E+02, 1.66E+02, 1.64E+02,
    1.62E+02, 1.60E+02, 1.59E+02, 1.57E+02, 1.55E+02, 1.54E+02,
    1.53E+02, 1.52E+02, 1.51E+02, 1.50E+02, 1.49E+02, 1.48E+02,
    1.47E+02, 1.46E+02, 1.45E+02, 1.44E+02, 1.43E+02, 1.42E+02,
    1.41E+02, 1.40E+02, 1.39E+02, 1.39E+02, 1.38E+02, 1.37E+02,
    1.36E+02, 1.36E+02, 1.35E+02, 1.35E+02, 1.34E+02, 1.34E+02,
    1.33E+02, 1.33E+02, 1.33E+02, 1.32E+02, 1.32E+02, 1.31E+02,
    1.31E+02, 1.31E+02, 1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02,
    1.29E+02, 1.29E+02, 1.29E+02, 1.29E+02, 1.29E+02, 1.29E+02,
    1.29E+02, 1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02,
    1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02,
    1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02,
    1.27E+02, 1.27E+02, 1.27E+02, 1.27E+02, 1.27E+02, 1.27E+02,
    1.27E+02, 1.27E+02, 1.27E+02, 1.27E+02, 1.27E+02, 1.27E+02,
    1.28E+02, 1.28E+02, 1.29E+02, 1.29E+02, 1.29E+02, 1.30E+02,
    1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02,
    1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02,
    1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02, 1.31E+02, 1.31E+02,
    1.31E+02, 1.31E+02, 1.31E+02, 1.31E+02, 1.31E+02, 1.31E+02,
    1.32E+02, 1.32E+02, 1.32E+02, 1.32E+02, 1.33E+02, 1.33E+02,
    1.33E+02, 1.33E+02, 1.33E+02, 1.33E+02, 1.33E+02, 1.33E+02,
    1.33E+02, 1.33E+02, 1.33E+02, 1.33E+02, 1.33E+02, 1.34E+02,
    1.34E+02, 1.34E+02, 1.34E+02, 1.34E+02, 1.34E+02, 1.34E+02,
    1.34E+02, 1.35E+02, 1.35E+02, 1.35E+02, 1.35E+02, 1.35E+02,
    1.35E+02, 1.35E+02, 1.35E+02, 1.35E+02, 1.35E+02, 1.35E+02,
    1.35E+02, 1.35E+02, 1.34E+02, 1.34E+02, 1.34E+02, 1.34E+02,
    1.34E+02, 1.34E+02, 1.34E+02, 1.34E+02, 1.34E+02, 1.34E+02,
    1.34E+02, 1.34E+02, 1.34E+02, 1.34E+02, 1.34E+02, 1.33E+02,
    1.33E+02, 1.33E+02, 1.33E+02, 1.33E+02, 1.32E+02, 1.32E+02,
    1.32E+02, 1.32E+02, 1.31E+02, 1.31E+02, 1.31E+02, 1.30E+02,
    1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02,
    1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02, 1.30E+02,
    1.30E+02, 1.29E+02, 1.29E+02, 1.29E+02, 1.29E+02, 1.28E+02,
    1.28E+02, 1.28E+02, 1.27E+02, 1.27E+02, 1.27E+02, 1.26E+02,
    1.26E+02, 1.26E+02, 1.25E+02, 1.25E+02, 1.25E+02, 1.25E+02,
    1.25E+02, 1.24E+02, 1.24E+02, 1.24E+02, 1.24E+02, 1.24E+02,
    1.25E+02, 1.25E+02, 1.25E+02, 1.25E+02, 1.25E+02, 1.25E+02,
    1.26E+02, 1.26E+02, 1.26E+02, 1.27E+02, 1.27E+02, 1.27E+02,
    1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02, 1.28E+02,
    1.28E+02, 1.27E+02, 1.27E+02, 1.27E+02, 1.27E+02, 1.27E+02,
    1.26E+02, 1.26E+02, 1.26E+02, 1.26E+02, 1.26E+02, 1.26E+02,
    1.26E+02, 1.26E+02, 1.26E+02, 1.26E+02, 1.26E+02, 1.26E+02,
    1.26E+02, 1.26E+02, 1.26E+02, 1.26E+02, 1.25E+02, 1.25E+02,
    1.25E+02, 1.25E+02, 1.25E+02, 1.24E+02, 1.24E+02, 1.24E+02,
    1.23E+02, 1.23E+02, 1.22E+02, 1.22E+02, 1.21E+02, 1.21E+02,
    1.20E+02, 1.20E+02, 1.19E+02, 1.18E+02, 1.18E+02, 1.17E+02,
    1.17E+02, 1.16E+02, 1.16E+02, 1.15E+02, 1.15E+02, 1.14E+02,
    1.14E+02, 1.13E+02, 1.13E+02, 1.12E+02, 1.12E+02, 1.12E+02,
    1.11E+02, 1.11E+02, 1.10E+02, 1.10E+02, 1.09E+02, 1.09E+02,
    1.08E+02, 1.08E+02, 1.07E+02, 1.07E+02, 1.06E+02, 1.05E+02,
    1.05E+02, 1.04E+02, 1.03E+02, 1.03E+02, 1.02E+02, 1.01E+02,
    1.01E+02, 9.99E+01, 9.93E+01, 9.87E+01, 9.81E+01, 9.76E+01,
    9.71E+01, 9.67E+01, 9.64E+01, 9.61E+01, 9.59E+01, 9.57E+01,
    9.57E+01, 9.56E+01, 9.57E+01, 9.57E+01, 9.57E+01, 9.58E+01,
    9.58E+01, 9.58E+01, 9.58E+01, 9.58E+01, 9.58E+01, 9.58E+01,
    9.58E+01, 9.58E+01, 9.59E+01, 9.59E+01, 9.59E+01, 9.60E+01,
    9.60E+01, 9.60E+01, 9.60E+01, 9.60E+01, 9.59E+01, 9.58E+01,
    9.57E+01, 9.56E+01, 9.55E+01, 9.54E+01, 9.53E+01, 9.52E+01,
    9.50E+01, 9.48E+01, 9.46E+01, 9.43E+01, 9.40E+01, 9.37E+01,
    9.34E+01, 9.30E+01, 9.27E+01, 9.25E+01, 9.22E+01, 9.21E+01,
    9.19E+01, 9.18E+01, 9.17E+01, 9.17E+01, 9.16E+01, 9.16E+01,
    9.16E+01, 9.16E+01, 9.16E+01, 9.16E+01, 9.16E+01, 9.15E+01,
    9.15E+01, 9.15E+01, 9.14E+01, 9.14E+01, 9.14E+01, 9.13E+01,
    9.13E+01, 9.13E+01, 9.13E+01, 9.13E+01, 9.13E+01, 9.13E+01,
    9.14E+01, 9.15E+01, 9.15E+01, 9.16E+01, 9.16E+01, 9.16E+01,
    9.16E+01, 9.15E+01, 9.14E+01, 9.12E+01, 9.11E+01, 9.09E+01,
    9.07E+01, 9.05E+01, 9.03E+01, 9.02E+01, 9.00E+01, 8.99E+01,
    8.98E+01, 8.97E+01, 8.96E+01, 8.96E+01, 8.95E+01, 8.94E+01,
    8.93E+01, 8.91E+01, 8.89E+01, 8.87E+01, 8.85E+01, 8.82E+01,
    8.80E+01, 8.77E+01, 8.74E+01, 8.71E+01, 8.67E+01, 8.64E+01,
    8.61E+01, 8.58E+01, 8.55E+01, 8.52E+01, 8.49E+01, 8.46E+01,
    8.43E+01, 8.41E+01, 8.38E+01, 8.36E+01, 8.34E+01, 8.31E+01,
    8.29E+01, 8.27E+01, 8.25E+01, 8.23E+01, 8.21E+01, 8.19E+01,
    8.18E+01, 8.16E+01, 8.15E+01, 8.13E+01, 8.12E+01, 8.10E+01,
    8.09E+01, 8.07E+01, 8.05E+01, 8.03E+01, 8.01E+01, 8.00E+01,
    7.98E+01, 7.96E+01, 7.94E+01, 7.93E+01, 7.91E+01, 7.89E+01,
    7.88E+01, 7.86E+01, 7.83E+01, 7.81E+01, 7.78E+01, 7.73E+01,
    7.68E+01, 7.61E+01, 7.53E+01, 7.44E+01, 7.35E+01, 7.26E+01,
    7.17E+01, 7.07E+01, 6.98E+01, 6.90E+01, 6.81E+01, 6.73E+01,
    6.65E+01, 6.57E+01, 6.49E+01, 6.41E+01, 6.33E+01, 6.25E+01,
    6.14E+01, 6.02E+01, 5.90E+01, 5.77E+01, 5.64E+01, 5.51E+01,
    5.38E+01, 5.24E+01, 5.11E+01, 4.97E+01, 4.84E+01, 4.71E+01,
    4.61E+01, 4.52E+01, 4.42E+01, 4.33E+01, 4.24E+01, 4.16E+01,
    4.08E+01, 4.01E+01, 3.94E+01, 3.88E+01, 3.82E+01, 3.77E+01,
    3.71E+01, 3.67E+01, 3.62E+01, 3.59E+01, 3.55E+01, 3.51E+01,
    3.48E+01, 3.44E+01, 3.39E+01, 3.35E+01, 3.31E+01, 3.27E+01,
    3.23E+01, 3.20E+01, 3.17E+01, 3.14E+01, 3.12E+01, 3.11E+01,
    3.09E+01, 3.08E+01, 3.07E+01, 3.07E+01, 3.07E+01, 3.07E+01,
    3.07E+01, 3.07E+01, 3.07E+01, 3.06E+01, 3.06E+01, 3.05E+01,
    3.04E+01, 3.02E+01, 3.01E+01, 3.01E+01, 3.00E+01, 3.01E+01,
    3.02E+01, 3.02E+01, 3.02E+01, 3.02E+01, 3.01E+01, 3.01E+01,
    3.01E+01, 3.01E+01, 3.01E+01, 3.02E+01, 3.02E+01, 3.03E+01,
    3.04E+01, 3.05E+01, 3.06E+01, 3.06E+01, 3.07E+01, 3.08E+01,
    3.09E+01, 3.10E+01, 3.11E+01, 3.12E+01, 3.14E+01, 3.16E+01,
    3.18E+01, 3.20E+01, 3.22E+01, 3.25E+01, 3.27E+01, 3.29E+01,
    3.30E+01, 3.31E+01, 3.32E+01, 3.32E+01, 3.33E+01, 3.33E+01,
    3.33E+01, 3.34E+01, 3.34E+01, 3.34E+01, 3.35E+01, 3.36E+01,
    3.37E+01, 3.38E+01, 3.40E+01, 3.41E+01, 3.43E+01, 3.44E+01,
    3.46E+01, 3.48E+01, 3.50E+01, 3.51E+01, 3.53E+01, 3.54E+01,
    3.56E+01, 3.58E+01, 3.60E+01, 3.61E+01, 3.63E+01, 3.65E+01,
    3.66E+01, 3.67E+01, 3.67E+01, 3.68E+01, 3.69E+01, 3.69E+01,
    3.69E+01, 3.70E+01, 3.70E+01, 3.71E+01, 3.71E+01, 3.70E+01,
    3.68E+01, 3.66E+01, 3.66E+01, 3.66E+01, 3.66E+01, 3.67E+01,
    3.67E+01, 3.67E+01, 3.66E+01, 3.66E+01, 3.67E+01, 3.68E+01,
    3.69E+01, 3.68E+01, 3.68E+01, 3.68E+01, 3.68E+01, 3.67E+01,
    3.67E+01, 3.67E+01, 3.67E+01, 3.67E+01, 3.66E+01, 3.66E+01,
    3.67E+01, 3.68E+01, 3.69E+01, 3.70E+01, 3.70E+01, 3.69E+01,
    3.69E+01, 3.68E+01, 3.66E+01, 3.65E+01, 3.63E+01, 3.61E+01,
    3.60E+01, 3.59E+01, 3.59E+01, 3.57E+01, 3.56E+01, 3.54E+01,
    3.53E+01, 3.52E+01, 3.51E+01, 3.53E+01, 3.54E+01, 3.54E+01,
    3.53E+01, 3.51E+01, 3.50E+01, 3.48E+01, 3.46E+01, 3.44E+01,
    3.42E+01, 3.41E+01, 3.39E+01, 3.37E+01, 3.36E+01, 3.36E+01,
    3.35E+01, 3.34E+01, 3.33E+01, 3.32E+01, 3.31E+01, 3.31E+01,
    3.31E+01, 3.31E+01, 3.31E+01, 3.31E+01, 3.31E+01, 3.31E+01,
    3.31E+01, 3.31E+01, 3.32E+01, 3.32E+01, 3.32E+01, 3.33E+01,
    3.33E+01, 3.33E+01, 3.34E+01, 3.34E+01, 3.34E+01, 3.33E+01,
    3.33E+01, 3.33E+01, 3.33E+01, 3.33E+01, 3.32E+01, 3.32E+01,
    3.33E+01, 3.33E+01, 3.33E+01, 3.34E+01, 3.34E+01, 3.34E+01,
    3.34E+01, 3.34E+01, 3.33E+01, 3.33E+01, 3.32E+01, 3.31E+01,
    3.31E+01, 3.30E+01, 3.30E+01, 3.30E+01, 3.30E+01, 3.31E+01,
    3.30E+01, 3.29E+01, 3.29E+01, 3.28E+01, 3.27E+01, 3.26E+01,
    3.25E+01, 3.24E+01, 3.23E+01, 3.23E+01, 3.23E+01, 3.24E+01,
    3.25E+01, 3.26E+01, 3.27E+01, 3.28E+01, 3.28E+01, 3.29E+01,
    3.30E+01, 3.31E+01, 3.31E+01, 3.31E+01, 3.31E+01, 3.30E+01,
    3.31E+01, 3.31E+01, 3.30E+01, 3.29E+01, 3.29E+01, 3.30E+01,
    3.31E+01, 3.33E+01, 3.34E+01, 3.36E+01, 3.38E+01, 3.39E+01,
    3.41E+01, 3.42E+01, 3.43E+01, 3.43E+01, 3.44E+01, 3.44E+01,
    3.43E+01, 3.43E+01, 3.43E+01, 3.42E+01, 3.42E+01, 3.41E+01,
    3.41E+01, 3.40E+01, 3.40E+01, 3.39E+01, 3.37E+01, 3.35E+01,
    3.34E+01, 3.33E+01, 3.31E+01, 3.30E+01, 3.29E+01, 3.29E+01,
    3.29E+01, 3.29E+01, 3.30E+01, 3.30E+01, 3.31E+01, 3.32E+01,
    3.33E+01, 3.34E+01, 3.35E+01, 3.36E+01, 3.36E+01, 3.37E+01,
    3.38E+01, 3.38E+01, 3.39E+01, 3.40E+01, 3.42E+01, 3.42E+01,
    3.43E+01, 3.44E+01, 3.44E+01, 3.43E+01, 3.42E+01, 3.41E+01,
    3.39E+01, 3.37E+01, 3.35E+01, 3.33E+01, 3.32E+01, 3.30E+01,
    3.30E+01, 3.29E+01, 3.29E+01, 3.29E+01, 3.29E+01, 3.30E+01,
    3.30E+01, 3.30E+01, 3.30E+01, 3.30E+01, 3.30E+01, 3.30E+01,
    3.30E+01, 3.29E+01, 3.29E+01, 3.30E+01, 3.30E+01, 3.30E+01,
    3.30E+01, 3.29E+01, 3.28E+01, 3.27E+01, 3.27E+01, 3.26E+01,
    3.26E+01, 3.27E+01, 3.28E+01, 3.28E+01, 3.29E+01, 3.29E+01,
    3.28E+01, 3.27E+01, 3.25E+01, 3.23E+01, 3.21E+01, 3.19E+01,
    3.17E+01, 3.15E+01, 3.13E+01, 3.12E+01, 3.11E+01, 3.10E+01,
    3.09E+01, 3.08E+01, 3.07E+01, 3.07E+01, 3.06E+01, 3.05E+01,
    3.05E+01, 3.04E+01, 3.03E+01, 3.01E+01, 3.00E+01, 2.98E+01,
    2.97E+01, 2.95E+01, 2.94E+01, 2.92E+01, 2.90E+01, 2.89E+01,
    2.88E+01, 2.87E+01, 2.86E+01, 2.85E+01, 2.84E+01, 2.84E+01,
    2.83E+01, 2.82E+01, 2.80E+01, 2.78E+01, 2.76E+01, 2.74E+01,
    2.71E+01, 2.67E+01, 2.64E+01, 2.61E+01, 2.57E+01, 2.53E+01,
    2.50E+01, 2.46E+01, 2.43E+01, 2.40E+01, 2.37E+01, 2.33E+01,
    2.30E+01, 2.27E+01, 2.24E+01, 2.21E+01, 2.19E+01, 2.16E+01,
    2.13E+01, 2.11E+01, 2.09E+01, 2.07E+01, 2.05E+01, 2.03E+01,
    2.01E+01, 2.00E+01, 1.99E+01, 1.97E+01, 1.96E+01, 1.95E+01,
    1.94E+01, 1.94E+01, 1.93E+01, 1.92E+01, 1.91E+01, 1.91E+01,
    1.91E+01, 1.90E+01, 1.90E+01, 1.90E+01, 1.91E+01, 1.91E+01,
    1.91E+01, 1.92E+01, 1.93E+01, 1.93E+01, 1.94E+01, 1.94E+01,
    1.94E+01, 1.94E+01, 1.94E+01, 1.93E+01, 1.93E+01, 1.92E+01,
    1.91E+01, 1.91E+01, 1.90E+01, 1.90E+01, 1.90E+01, 1.89E+01,
    1.90E+01, 1.90E+01, 1.91E+01, 1.91E+01, 1.92E+01, 1.93E+01,
    1.93E+01, 1.93E+01, 1.93E+01, 1.92E+01, 1.92E+01, 1.91E+01,
    1.90E+01, 1.89E+01, 1.88E+01, 1.87E+01, 1.86E+01, 1.85E+01,
    1.84E+01, 1.83E+01, 1.82E+01, 1.81E+01, 1.81E+01, 1.80E+01,
    1.79E+01, 1.78E+01, 1.77E+01, 1.76E+01, 1.76E+01, 1.75E+01,
    1.74E+01, 1.74E+01, 1.73E+01, 1.73E+01, 1.72E+01, 1.72E+01,
    1.71E+01, 1.71E+01, 1.71E+01, 1.71E+01, 1.71E+01, 1.71E+01,
    1.72E+01, 1.72E+01, 1.72E+01, 1.73E+01, 1.73E+01, 1.73E+01,
    1.73E+01, 1.73E+01, 1.72E+01, 1.72E+01, 1.72E+01, 1.71E+01,
    1.71E+01, 1.70E+01, 1.69E+01, 1.69E+01, 1.68E+01]


# - local functions ----------------------------
def helios_spectrum() -> xr.Dataset:
    """Define Helios spectrum."""
    # Maybe we should also check the light-level value as specified
    # in the name of the L0 file. The light level is coded as:
    # L1: 100%, L2: 50%, L3: 30%, L4: 15%, L5: 7%, L6: 3%
    wavelength = np.linspace(350, 2400, 2051, dtype='f4')
    xar_wv = xr.DataArray(wavelength,
                          coords={'wavelength': wavelength},
                          attrs={'longname': 'wavelength grid',
                                 'units': 'nm',
                                 'comment': 'wavelength annotation'})

    xar_sign = xr.DataArray(1e-3 * np.array(HELIOS_SPECTRUM, dtype='f4'),
                            coords={'wavelength': wavelength},
                            attrs={'longname': 'Helios radiance spectrum',
                                   'units': 'W/(m^2.sr.nm)'})

    return xr.Dataset({'wavelength': xar_wv, 'spectral_radiance': xar_sign},
                      attrs=HELIOS_ATTRS)


def __test(l1a_file: str) -> None:
    """Small function to test this module."""
    # Create a netCDF4 file containing the Helios reference spectrum
    xds = helios_spectrum()
    xds.to_netcdf(l1a_file, mode='w', format='NETCDF4',
                  group='/gse_data/ReferenceSpectrum')


# --------------------------------------------------
if __name__ == '__main__':
    print('---------- SHOW DATASET ----------')
    print(helios_spectrum())
    print('---------- WRITE DATASET ----------')
    __test('test_netcdf.nc')
