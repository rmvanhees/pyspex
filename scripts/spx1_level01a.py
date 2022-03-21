#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python script to store SPEXone Level-0 data in a new Level-1A product.

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#from pyspex import spx_product
from pyspex.lv1_io import L1Aio
from pyspex.st3_io import (ap_id, split_ccsds, select_science, select_hk,
                           ScienceCCSDS, TmTcCCSDS)

# - global parameters ------------------------------
EPOCH = datetime(1958, 1, 1, tzinfo=timezone.utc)
LEAP_SECONDS = 0  # only in-flight the CCSDS packages have TAI timestamps

# - local functions --------------------------------
def __rd_l0_data(args) -> tuple:
    """
    Read level 0 data and return Science and telemetry data
    """
    # concatenate CCSDS Science data of the input files
    if args.msmt_id.name.endswith('.ST3'):
        with open(args.msmt_id, 'rb') as fp:
            ccsds_data, ccsds_hk = split_ccsds(fp.read())
    else:
        data_dir = args.msmt_id.parent
        sci_files = sorted(data_dir.glob(args.msmt_id.name + '.[0-9]')) \
            + sorted(data_dir.glob(args.msmt_id.name + '.?[0-9]'))
        ccsds_data = ()
        for flname in sci_files:
            with open(flname, 'rb') as fp:
                ccsds_data += select_science(fp.read())

        hk_files = sorted(data_dir.glob(args.msmt_id.name + '_hk.[0-9]'))
        print(hk_files)
        ccsds_hk = ()
        for flname in hk_files:
            with open(flname, 'rb') as fp:
                ccsds_hk += select_hk(fp.read())

    # perform data dump and exit
    if args.dump:
        # dump header information of the Science packages
        ScienceCCSDS(ccsds_data, args.verbose).dump(
            args.datapath / (args.msmt_id.name + '.dump'))
        # dump header information of the Science packages
        TmTcCCSDS(ccsds_hk, args.verbose).dump(
            args.datapath / (args.msmt_id.name + '_hk.dump'))
        return None

    # read Science packages and collect all detector read-outs
    sci_list = ScienceCCSDS(ccsds_data, args.verbose).read(args.select)
    if not sci_list:
        print('[WARNING]: no science data found or selected')
        return None

    science = np.concatenate(sci_list)
    mps_list = np.unique(science['hk']['MPS_ID']).tolist()
    if args.verbose:
        print(f'[INFO]: list of unique MPS {mps_list}')

    tmtc_list = TmTcCCSDS(ccsds_hk, args.verbose).read()
    nomhk = np.concatenate(
        [(x,) for x in tmtc_list if ap_id(x['hdr']) == 0x320
         and x['hk']['MPS_ID'] in mps_list])

    return (science, nomhk)


def get_science_timestamps(science):
    """
    Return timestamps of the Science packets
    """
    if science['hk']['ICUSWVER'][0] > 0x123:
        img_sec = science['icu_tm']['tai_sec']
        img_subsec = science['icu_tm']['sub_sec']
        return (img_sec, img_subsec)

    img_sec = science['hdr']['tai_sec']
    img_subsec = science['hdr']['sub_sec']
    if science['hk']['ICUSWVER'][0] > 0x123:
        # fix bug in sub-seconds
        us100 = np.round(10000 * img_subsec.astype(float) / 65536)
        buff = us100 + img_sec - 10000
        us100 = buff.astype('u8') % 10000
        img_subsec = ((us100 << 16) // 10000).astype('u2')
    return (img_sec, img_subsec)


def get_nomhk_timestamps(nomhk):
    """
    Return timestamps of the telemetry packets
    """
    nomhk_sec = nomhk['hdr']['tai_sec']
    nomhk_subsec = nomhk['hdr']['sub_sec']
    if nomhk['hk']['ICUSWVER'][0] > 0x123:
        # fix bug in sub-seconds
        us100 = np.round(10000 * nomhk_subsec.astype(float) / 65536)
        buff = us100 + nomhk_sec - 10000
        us100 = buff.astype('u8') % 10000
        nomhk_subsec = ((us100 << 16) // 10000).astype('u2')
    return (nomhk_sec, nomhk_subsec)


def get_l1a_name() -> str:
    """
    Generate Level-1A products name, using the following filename conventions

    Inflight
    --------
    L1A file name format, following the NASA ... naming convention:
       PACE_SPEXone[_TTT].YYYYMMDDTHHMMSS.L1A.Vnn.nc
    where
       TTT is an optional data type (e.g., for the calibration data files)
       YYYYMMDDTHHMMSS is time stamp of the first image in the file
       nn file-version number
    for example
    [Science Product] PACE_SPEXone.20230115T123456.L1A.V01.nc
    [Calibration Product] PACE_SPEXone_CAL.20230115T123456.L1A.V01.nc
    [Monitoring Products] PACE_SPEXone_DARK.20230115T123456.L1A.V01.nc

    OCAL
    ----
    see the SPEXone L1AC IODS
    """
    return 'SPX1_L1A_PRODUCT.nc'

# - main function ----------------------------------
def main():
    """
    main function
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='store Level-0 data in a new SPEXone Level-1A product')
    parser.add_argument('--dump', action='store_true',
                        help=('dump CCSDS packet headers in ASCII'))
    parser.add_argument('--verbose', action='store_true', help='be verbose')
    parser.add_argument('--select', default='all',
                        choices=['all', 'binned', 'fullFrame'],
                        help=('read "all" data (default) or select'
                              ' "binned" / "full-frame" detector-readouts'))
    parser.add_argument('--datapath', type=Path, default=Path('.'),
                        help='directory to store the Level-1A product')
    # parser.add_argument('--st3_nav', default=None, type=str,
    #                    help='name of ST3 file with navigation data')
    # Note that science packages and telementry packages are combined in one
    # ST3 product (in chronological order), but seperated in CCSDS products.
    parser.add_argument('msmt_id', type=Path,
                        help=('name of the SPEXone level 0 in ST3 format,'
                              ' or name of the measurement without extension'
                              '(full path). The names of the telemetry data'
                              ' are expected to be: msmt_id+"_hk.?" (CCSDS)'))
    args = parser.parse_args()
    if args.verbose:
        print(args)

    # read level 0 data
    if (res := __rd_l0_data(args)) is None:
        return

    science, nomhk = res
    if args.verbose:
        print(f'[INFO]: number of Science packages = {science.size}')
        print(f'[INFO]: number of telemetry packages = {nomhk.size}')
    dims = {'number_of_images': science.size,
            'samples_per_image': science['hk']['IMRLEN'].max() // 2,
            'hk_packets': nomhk.size,
            'SC_records': None}
    print(dims)

    # generate name of the level-1A product
    prod_name = 

    # Generate and fill L1A product
    with L1Aio(args.datapath / prod_name, dims=dims) as l1a:
        # write image data, detector telemetry and image attributes
        img_data = np.empty((science.size, dims['samples_per_image']),
                            dtype=float)
        for ii, data in enumerate(science['frame']):
            img_data[ii, :data.size] = data
        l1a.fill_science(img_data, science['hk'],
                         np.bitwise_and(science['hdr']['sequence'], 0x3fff))
        del img_data
        img_sec, img_subsec = get_science_timestamps(science)
        l1a.fill_time(img_sec, img_subsec, group='image_attributes')

        # write engineering data
        if nomhk.size > 0:
            l1a.fill_nomhk(nomhk['hk'])
            nomhk_sec, nomhk_subsec = get_nomhk_timestamps(nomhk)
            l1a.fill_time(nomhk_sec, nomhk_subsec, group='engineering_data')

        # if demhk.size > 0:
        #    l1a.fill_demhk(demhk['hk'])

        # write global attributes
        l1a.fill_global_attrs(inflight=args.msmt_id.name.endswith('.ST3'))
        l1a.set_attr('input_files', str(args.msmt_id))


# --------------------------------------------------
if __name__ == '__main__':
    main()
