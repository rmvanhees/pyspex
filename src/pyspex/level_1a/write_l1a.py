#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Write Level-0 data to new Level-1A product.
"""
from datetime import datetime
from dataclasses import dataclass

import xarray as xr

from pyspex import version
from pyspex.hkt_io import HKTio
from pyspex.lv0_io import (coverage_time,
                           select_lv0_data,
                           write_lv0_data)


def get_l1a_name(datatype: str, config: dataclass,
                 sensing_start: datetime) -> str:
    """
    Generate name of Level-1A product based on filename conventions described
    below

    Parameters
    ----------
    datatype :  {'OCAL', 'DARK', 'CAL', 'SCIENCE'}
    config :  dataclass
    sensing_start :  datetime

    Returns
    -------
    str
        Name of Level-1A product

    Notes
    -----

    === Inflight ===
    L1A file name format, following the NASA ... naming convention:
       PACE_SPEXONE[_TTT].YYYYMMDDTHHMMSS.L1A[.Vnn].nc
    where
       TTT is an optional data type (e.g., for the calibration data files)
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       Vnn file-version number (ommited when nn=1)
    for example (file-version=1):
       [Science Product] PACE_SPEXONE.20230115T123456.L1A.nc
       [Calibration Product] PACE_SPEXONE_CAL.20230115T123456.L1A.nc
       [Dark science Product] PACE_SPEXONE_DARK.20230115T123456.L1A.nc

    === OCAL ===
    L1A file name format:
       SPX1_OCAL_<msm_id>_L1A_YYYYMMDDTHHMMSS_yyyymmddThhmmss_vvvv.nc
    where
       msm_id is the measurement identifier
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       yyyymmddThhmmss is the creation time (UTC) of the product
       vvvv is the version number of the product starting at 0001
    """
    if config.outfile:
        return config.outfile

    if config.file_format != 'raw':
        # inflight product name
        prod_type = {'DARK': '_DARK',
                     'CAL': '_CAL',
                     'SCIENCE': ''}.get(datatype.upper(), '')
        prod_ver = '' if config.file_version == 1\
            else f'.V{config.file_version:02d}'

        return (f'PACE_SPEXONE{prod_type}'
                f'.{sensing_start.strftime("%Y%m%dT%H%M%S"):15s}.L1A'
                f'.{prod_ver}.nc')

    # OCAL product name
    # determine measurement identifier
    msm_id = config.file_list[0].stem
    try:
        new_date = datetime.strptime(
            msm_id[-22:], '%y-%j-%H:%M:%S.%f').strftime('%Y%m%dT%H%M%S.%f')
    except ValueError:
        pass
    else:
        msm_id = msm_id[:-22] + new_date

    return (f'SPX1_OCAL_{msm_id}_L1A'
            f'_{sensing_start.strftime("%Y%m%dT%H%M%S"):15s}'
            f'_{version.get(githash=True)}.nc')


def read_hkt_nav(file_list: list) -> xr.Dataset:
    """
    Read multiple HKT products and collect data in a Python dictionary
    """
    dim_dict = {'att_': 'att_time',
                'orb_': 'orb_time',
                'tilt': 'tilt_time'}

    res = {}
    for name in file_list:
        hkt = HKTio(name)
        nav = hkt.navigation()
        if not res:
            res = nav.copy()
        else:
            for key1, value in nav.items():
                hdim = dim_dict.get(key1, None)
                res[key1] = xr.concat((res[key1], value), dim=hdim)

    return xr.merge((res['att_'], res['orb_'], res['tilt']),
                    combine_attrs='drop_conflicts')


def write_lv0_nav(l1a_file: str, xds_nav: xr.Dataset):
    """
    Add PACE navigation data to existing Level-1A product
    """
    xds_nav.to_netcdf(l1a_file, group='navigation_data', mode='a')


# --------------------------------------------------
def write_l1a(config, science_in, nomhk_in):
    """Write Level-1A product.
    """
    if config.eclipse is None:
        # this are "OCAL data" try to write all data to one L1A product.
        science, nomhk = select_lv0_data('OCAL', science_in, nomhk_in,
                                         config.verbose)
        # write L1A product
        prod_name = get_l1a_name('OCAL', config, coverage_time(science)[0])
        write_lv0_data(config.outdir / prod_name, config.l0_list,
                       config.l0_format, science, nomhk)
        return

    if not config.eclipse:
        # this are "Science data": binned data in "Science mode".
        science, nomhk = select_lv0_data('Science', science_in, nomhk_in,
                                         config.verbose)
        # write L1A product
        prod_name = get_l1a_name('Science', config, coverage_time(science)[0])
        try:
            write_lv0_data(config.outdir / prod_name, config.l0_list,
                           config.l0_format, science, nomhk)
        except (PermissionError, RuntimeError) as exc:
            raise RuntimeError from exc

        # add PACE navigation information from HKT products
        if config.pace_hkt:
            hkt_nav = read_hkt_nav(config.pace_hkt)
            # select HKT data collocated with Science data
            # - issue a warning if selection is empty
            write_lv0_nav(config.outdir / prod_name, hkt_nav)
        return

    # this can be "Dark data": binned data using "Science mode" MPSes
    # and/or "Calibration data”: full frame data in "Diagonstic mode".
    for dtype in ['DARK', 'CAL']:
        science, nomhk = select_lv0_data(dtype, science_in, nomhk_in,
                                         config.verbose)
        if science is None:
            continue

        # write L1A product
        prod_name = get_l1a_name(dtype, config, coverage_time(science)[0])
        try:
            write_lv0_data(config.outdir / prod_name, config.l0_list,
                           config.l0_format, science, nomhk)
        except (PermissionError, RuntimeError) as exc:
            raise RuntimeError from exc

        # add PACE navigation information from HKT products
        if config.pace_hkt:
            hkt_nav = read_hkt_nav(config.pace_hkt)
            # select HKT data collocated with Science data
            # - issue a warning if selection is empty
            write_lv0_nav(config.outdir / prod_name, hkt_nav)
