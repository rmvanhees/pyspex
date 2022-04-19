"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Define correction for non-ideal GSFC polarizer

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np
import xarray as xr

# - global parameters ------------------------------
DOLP_ATTRS = {}

DOLP_CORR = np.ones(450, dtype='f4')

# - local functions ----------------------------
def gsfc_polarizer() -> xr.Dataset:
    """
    Define correction for non-ideal GSFC polarizer
    """
    wavelength = np.linspace(350, 800, 451, dtype='f4')
    xar_wv = xr.DataArray(wavelength,
                          coords={'wavelength': wavelength},
                          attrs={'longname': 'wavelength grid',
                                 'units': 'nm',
                                 'comment': 'wavelength annotation'})

    xar_corr = xr.DataArray(DOLP_CORR,
                            coords={'wavelength': wavelength},
                            attrs={'longname': 'DoLP',
                                   'units': '1'})

    return xr.Dataset({'wavelength': xar_wv, 'DoLP': xar_corr},
                      attrs=DOLP_ATTRS)


def test(l1a_file: str) -> None:
    """
    Create a netCDF4 file containing the spectralDoLP
    """
    xds = gsfc_polarizer()
    xds.to_netcdf(l1a_file, mode='w', format='NETCDF4',
                  group='/gse_data/Polarizer')


# --------------------------------------------------
if __name__ == '__main__':
    print('---------- SHOW DATASET ----------')
    print(gsfc_polarizer())
    print('---------- WRITE DATASET ----------')
    test('test_netcdf.nc')
