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
Defines the spectral dependent DolP of the Moxtek polarizer.
"""
__all__ = ['gsfc_polarizer']

import numpy as np
import xarray as xr

# - global parameters ------------------------------
DOLP_ATTRS = {'instrument': 'Moxtek polarizer characterisation',
              'comment': 'neglect when attribute DolP=0'}

WAVE = [360, 380, 410, 440, 460, 488,
        508.5, 532, 580, 770, 870]
DOLP = [.9805, .9833, .9875, .9913, .9932, .9948,
        .9962, .9972, .9972, .9983, .9972]
RATIO = [101.6, 118.9, 159., 229.5, 292.6, 383.6,
         530.6, 713.3, 710.1, 1193., 723.]


# - local functions ----------------------------
def gsfc_polarizer() -> xr.Dataset:
    """
    Define table for spectral dependent DolP
    """
    xar_wv = xr.DataArray(np.array(WAVE, dtype='f4'),
                          coords={'wavelength': WAVE},
                          attrs={'longname': 'wavelength grid',
                                 'units': 'nm',
                                 'comment': 'wavelength annotation'})
    xar_dolp = xr.DataArray(np.array(DOLP, dtype='f4'),
                            coords={'wavelength': WAVE},
                            attrs={'longname': 'DoLP',
                                   'units': '1'})
    xar_ratio = xr.DataArray(np.array(RATIO, dtype='f4'),
                             coords={'wavelength': WAVE},
                             attrs={'longname': 'Extinction ratio',
                                    'units': '1'})

    return xr.Dataset({'wavelength': xar_wv, 'DoLP': xar_dolp,
                       'Extinction_Ratio': xar_ratio}, attrs=DOLP_ATTRS)


def __test(l1a_file: str) -> None:
    """Small function to test this module.
    """
    # Create a netCDF4 file containing the spectralDoLP
    xds = gsfc_polarizer()
    xds.to_netcdf(l1a_file, mode='w', format='NETCDF4',
                  group='/gse_data/Polarizer')


# --------------------------------------------------
if __name__ == '__main__':
    print('---------- SHOW DATASET ----------')
    print(gsfc_polarizer())
    print('---------- WRITE DATASET ----------')
    __test('test_netcdf.nc')
