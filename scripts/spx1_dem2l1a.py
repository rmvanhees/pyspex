#!/usr/bin/env python3
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
Python implementation to convert SPEXone DEM measurements to L1A format.
The DEM measurements are performed at SRON between 17-10-2019 and 13-12-2019.

References
----------
* SRON-SPEX-TN-2020-001_0_5_SPEXone_Detector_Characterization.pdf
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from pyspex import spx_product
from pyspex.lib.tmtc_def import tmtc_dtype
from pyspex.dem_io import DEMio
from pyspex.lv0_io import img_sec_of_day
from pyspex.lv1_io import L1Aio
from pyspex.lv1_gse import LV1gse

# - global parameters ------------------------------
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """
    main program to illustate the creation of a L1A calibration product
    """
    parser = argparse.ArgumentParser(
        description=('create SPEXone L1A product from DEM measurement(s)'))
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('--reference', default=None,
                        help='reference detector data (for non-linearity)')
    parser.add_argument('--output', default=None,
                        help='define output directory, default=CWD')
    parser.add_argument('--dem_id', choices=('D35', 'D39'), default=None,
                        help=('provide DEM ID'
                              ' or ID will be extracted from path'))
    parser.add_argument('file_list', nargs='+',
                        help=("provide path to _b.bin file of one or more"
                              " SPEXone DEM characterization measurements"))
    args = parser.parse_args()
    if args.verbose:
        print(args)

    # sort file on timestamp in filename (problem: midnight...)
    buff = [name for name in args.file_list if name.endswith('.bin')]
    args.file_list = sorted(buff, key=lambda name: name.split('_')[-2])

    # obtain DEM_ID (can be specified on command-line)
    # and list of measurements used to generate the L1A product
    list_dem_id = []
    for file in args.file_list:
        parts = Path(file).parts
        if len(parts) > 2:
            list_dem_id.append(parts[-2].split('_')[1])

    if args.dem_id is None:
        dem_id_list = []
        for file in args.file_list:
            parts = Path(file).parts
            if len(parts) == 1:
                continue
            for dem_id in ('D35', 'D39'):
                if parts[-2].find(dem_id) != -1:
                    dem_id_list.append(dem_id)

        if not dem_id_list:
            raise KeyError(
                'Can not determine DEM_ID, please specify --dem_id')
        dem_id_list = set(dem_id_list)
        if len(dem_id_list) != 1:
            raise KeyError('DEM_ID of measurements must be unique')

        args.dem_id = dem_id_list.pop()

    # initialze data arrays
    msm_id = None
    tstamp = []
    image_list = []

    # Measurement Parameters Settings
    n_images = len(args.file_list)
    img_hk = np.zeros(n_images, dtype=tmtc_dtype(0x350))
    hk_data = np.zeros(n_images, dtype=tmtc_dtype(0x320))
    t_exp = np.empty(n_images, dtype=float)
    t_frm = np.empty(n_images, dtype=float)
    offset = np.empty(n_images, dtype=float)
    temp = np.empty(n_images, dtype=float)

    for ii, flname in enumerate(args.file_list):
        dem_file = Path(flname)
        if not dem_file.is_file():
            raise FileNotFoundError(f'file {flname} does not exist')
        if dem_file.suffix == '.txt':
            raise ValueError("We should not try a file with suffix '.txt'")

        # read DEM settings and data
        parts = dem_file.name.split('_')
        if len(parts) == 8:
            _id = '_'.join(parts[:4])
            tstamp.append(datetime.strptime(parts[5] + parts[6] + '+00:00',
                                            '%Y%m%d%H%M%S.%f%z'))
        elif len(parts) == 10:
            _id = '_'.join(parts[:6])
            tstamp.append(datetime.strptime(parts[7] + parts[8] + '+00:00',
                                            '%Y%m%d%H%M%S.%f%z'))
        elif len(parts) == 7:
            _id = '_'.join(parts[:3])
            tstamp.append(datetime.strptime(parts[4] + parts[5] + '+00:00',
                                            '%Y%m%d%H%M%S.%f%z'))
        else:
            raise ValueError("Invalid format of data-product name")

        if msm_id is None:
            msm_id = _id
        elif msm_id != _id:
            raise ValueError("Do not combine different kind of measurements")

        dem = DEMio(flname)
        # obtain Science_HK information from header file (ASCII)
        img_hk[ii] = dem.get_sci_hk()
        # get nr_coaddings from file name
        coad_str = [x for x in parts if x.startswith('coad')][0]
        img_hk[ii]['REG_NCOADDFRAMES'] = int(coad_str[-2:])
        # determine exposure time
        t_exp[ii] = dem.exp_time()
        t_frm[ii] = dem.frame_period(int(coad_str[-2:]))
        offset[ii] = dem.offset
        # read image data
        image_list.append(dem.get_data())

    images = np.array(image_list)
    del image_list

    # sort data according to timestamps
    if tstamp != sorted(tstamp):
        print('[WARNING]: do a sort of all the data...')
        indx = sorted(range(n_images), key=tstamp.__getitem__)
        img_hk = img_hk[indx]
        tstamp = [tstamp[k] for k in indx]
        t_exp = t_exp[indx]
        t_frm = t_frm[indx]
        offset = offset[indx]
        temp = temp[indx]
        images = images[indx]

    # convert timestamps to seconds per day
    img_sec = np.empty(n_images, dtype='u4')
    img_subsec = np.empty(n_images, dtype='u2')
    for ii, tval in enumerate(tstamp):
        img_sec[ii] = (tval - EPOCH).total_seconds()
        img_subsec[ii] = tval.microsecond * 2**16 // 1000000

    ref_date, img_time = img_sec_of_day(img_sec, img_subsec, img_hk)

    # generate name of L1A product
    prod_name = spx_product.prod_name(tstamp[0], msm_id=msm_id.strip(' '))
    if args.output is not None:
        dest_dir = Path(args.output)
        if not dest_dir.is_dir():
            dest_dir.mkdir()
        prod_name = str(dest_dir / prod_name)

    # set dimensions of L1A datasets
    if images.ndim == 2:
        n_samples = images.size
    else:
        n_samples = images.shape[1] * images.shape[2]

    dims = {'number_of_images': n_images,
            'samples_per_image': n_samples,
            'SC_records': None,
            'tlm_packets': None,
            'nv': 1}

    # generate L1A product
    with L1Aio(prod_name, dims=dims, ref_date=ref_date.date()) as l1a:
        # write image data, detector telemetry and image attributes
        l1a.fill_science(images.reshape(n_images, n_samples), img_hk,
                         np.arange(n_images))
        l1a.set_dset('/image_attributes/icu_time_sec', img_sec)
        l1a.set_dset('/image_attributes/icu_time_subsec', img_subsec)
        l1a.set_dset('/image_attributes/image_time', img_time)

        # Engineering data
        l1a.fill_nomhk(hk_data)
        l1a.set_dset('/engineering_data/HK_tlm_time', img_time)

        # Global attributes
        l1a.fill_global_attrs(inflight=False)
        l1a.set_attr('input_files', [Path(x).name for x in args.file_list])

    # Add OGSE and EGSE parameters
    # - Light source
    # - DoLP
    # - AoLP
    # - FOV_begin
    # - FOV_end
    # - Detector illumination
    # - Illumination_level
    # - viewport(s)
    # - Wavelength and signal of data stimulus
    with LV1gse(prod_name) as gse:
        gse.set_attr('origin', 'SPEXone Detector Characterization (SRON)')
        gse.set_attr('measurement', msm_id)
        gse.set_attr('dem_id', args.dem_id)
        if args.reference is not None:
            sec_bgn = int(tstamp[0].timestamp())
            sec_end = int(tstamp[-1].timestamp())
            with h5py.File(args.reference, 'r') as fid:
                secnd = fid['sec'][:]
                data = fid['amps'][((secnd >= sec_bgn)
                                    & (secnd <= sec_end))]
            gse.write_reference_signal(np.median(data), data.std())


# --------------------------------------------------
if __name__ == '__main__':
    main()
