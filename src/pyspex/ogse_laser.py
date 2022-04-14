"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Add the wavelength of the tunable Paladin OPO laser of the NASA GSFC
Calibration facility as used for the ISRF and stray-light measurements to
the SPEXone level-1A products.

Copyright (c) 2021-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
import xarray as xr


# - local functions ----------------------------
def read_gse_excel(target_cwl: str, gse_dir: Path) -> xr.Dataset:
    """
    Return GSE info on central wavelength, line-width and laser radiance

    Parameters
    ----------
    target_cwl: str
       Target laser wavelength as provided in the filename
    gse_dir: Path
       Path to the folder with the Excel file
    """
    wbook = load_workbook(gse_dir / 'SPEXOne_ALL_360-840nm.xlsx')
    wsheet = wbook.active
    actual_cwl = np.array([cel.value for cel in wsheet['E'][1: -1]])
    print(actual_cwl)
    indx = np.argmin(np.abs(actual_cwl - float(target_cwl[:-2])))
    print(indx, actual_cwl[indx], target_cwl, float(target_cwl[:-2]))
    if abs(float(target_cwl[:-2]) - actual_cwl[indx]) > 2:
        print('*** WARNING: no GSE information found')
        wbook.close()
        return None

    indx += 1
    xds = xr.Dataset({
        'wv': xr.DataArray([wsheet['E'][indx].value],
                           coords={'elmnt': np.atleast_1d(1)},
                           attrs={'long_name': 'central wavelength',
                                  'units': 'nm'}),
        'wv_std': xr.DataArray([wsheet['F'][indx].value],
                               coords={'elmnt': np.atleast_1d(1)},
                               attrs={'long_name': 'central wavelength [std]',
                                      'units': 'nm'}),
        'lw': xr.DataArray([wsheet['G'][indx].value],
                           coords={'elmnt': np.atleast_1d(1)},
                           attrs={'long_name': 'line width',
                                  'units': 'nm'}),
        'signal': xr.DataArray([wsheet['I'][indx].value],
                               coords={'elmnt': np.atleast_1d(1)},
                               attrs={'long_name': 'radiance',
                                      'units': 'W/(m^2.sr)'})})
    wbook.close()
    return xds


def test_netcdf(l1a_file: str) -> None:
    """
    Create a netCDF4 file with the Helios data in it
    """
    # xds = read_gse_excel('465.4nm', Path('/data/richardh/SPEXone/GSFC'))
    xds = read_gse_excel('360nm', Path('/data/richardh/SPEXone/GSFC'))
    xds = read_gse_excel('840nm', Path('/data/richardh/SPEXone/GSFC'))
    xds.to_netcdf(l1a_file, mode='w', format='NETCDF4',
                  group='/gse_data/OPO_laser')


# --------------------------------------------------
if __name__ == '__main__':
    test_netcdf('test_netcdf.nc')
