"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Add the wavelength of the tunable Paladin OPO laser of the NASA GSFC
Calibration facility as used for the ISRF and stray-light measurements to
the SPEXone level-1A products.

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path

import numpy as np
from openpyxl import load_workbook
import xarray as xr


# - local functions ----------------------------
def read_gse_excel(gse_dir: Path, target_cwl: str) -> xr.Dataset:
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
    # print(actual_cwl)
    indx = np.argmin(np.abs(actual_cwl - float(target_cwl[:-2])))
    print(f'{target_cwl:>8s}: {actual_cwl[indx]:.3f} at index={indx:d}')
    if abs(float(target_cwl[:-2]) - actual_cwl[indx]) > 2:
        print('*** WARNING: no GSE information found')
        wbook.close()
        return None

    indx += 1
    wavelength = np.array([wsheet['E'][indx].value])
    xar_wv = xr.DataArray(wavelength,
                          coords={'wavelength':wavelength },
                          attrs={'long_name': 'central wavelength',
                                 'units': 'nm'})
    xar_std = xr.DataArray([wsheet['F'][indx].value],
                           coords={'wavelength': wavelength},
                           attrs={'long_name': 'central wavelength [std]',
                                  'units': 'nm'})
    xar_lw = xr.DataArray([wsheet['G'][indx].value],
                          coords={'wavelength': wavelength},
                          attrs={'long_name': 'line width',
                                 'units': 'nm'})
    xar_sign = xr.DataArray([wsheet['I'][indx].value],
                            coords={'wavelength': wavelength},
                            attrs={'long_name': 'radiance',
                                   'units': 'W/(m^2.sr)'})
    wbook.close()
    return xr.Dataset({'wavelength': xar_wv,
                       'wv_std': xar_std,
                       'linewidth': xar_lw,
                       'signal': xar_sign})


def add_ogse_laser(l1a_file: str, target_cwl: str, gse_dir: Path) -> None:
    """
    Add netCDF4 group OPO_laser to gse_data in an L1A product
    """
    # ToDo if target_cw is None: find intended laser wavelength in filename
    if gse_dir is None:
        gse_dir = Path('Logs')
    xds = read_gse_excel(gse_dir, target_cwl)
    xds.to_netcdf(l1a_file, mode='r+', format='NETCDF4',
                  group='/gse_data/OPO_laser')


def test_netcdf(l1a_file: str) -> None:
    """
    Create a netCDF4 file with the Helios data in it
    """
    gse_dir = Path('/data/richardh/SPEXone/GSFC')
    xds = read_gse_excel(gse_dir, '465.4nm')
    xds = read_gse_excel(gse_dir, '360nm', )
    xds = read_gse_excel(gse_dir, '840nm')
    xds.to_netcdf(l1a_file, mode='w', format='NETCDF4',
                  group='/gse_data/OPO_laser')


# --------------------------------------------------
if __name__ == '__main__':
    test_netcdf('test_netcdf.nc')
