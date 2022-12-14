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
Quick and dirty script to generate simple Quick-Look figures.
"""

import argparse

from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from moniplot.lib.fig_info import FIGinfo
from moniplot.tol_colors import tol_cmap
from moniplot.mon_plot import MONplot

from pyspex.binning_tables import BinningTables


# --------------------------------------------------
def binned_to_2x2_image(coverage_start: str, table_id: int, img_binned):
    """
    Convert binned detector data to image (1024, 1024)
    """
    try:
        bin_ckd = BinningTables()
        bin_ckd.search(coverage_start)
    except Exception as exc:
        raise RuntimeError from exc

    return bin_ckd.unbin(table_id, img_binned).reshape(1024, 1024)


# --------------------------------------------------
def main():
    """
    Main function of this module
    """
    parser = argparse.ArgumentParser(
        description='create Quick-Look from SPEXone L1A product')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('--show_images', type=str, default=None,
                        help='comma seperated list, default use --max_images')
    parser.add_argument('--max_images', type=int, default=20,
                        help='maximum number of images in quick-look (20)')
    parser.add_argument('file_list', nargs='*',
                        help='provide name of L1A product')
    args = parser.parse_args()
    if args.verbose:
        print(args)

    if args.file_list:
        file_list = [Path(xx) for xx in args.file_list]
    else:
        file_list = Path('.').glob('*.nc')

    for flname in sorted(file_list):
        print(flname)

        with h5py.File(flname, 'r') as fid:
            coverage_start = fid.attrs['time_coverage_start']
            image_time = fid['/image_attributes/image_time'][:]
            exposure_time = fid['/image_attributes/exposure_time'][:]
            table_id = fid['/image_attributes/binning_table'][:]
            sci_hk = fid['/science_data/detector_telemetry'][:]
            dset = fid['/science_data/detector_images']
            # pylint: disable=no-member
            images = dset.astype(float)[:]
        try:
            coverage_start = coverage_start.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
        else:
            date_start = coverage_start.split('.')[0]

        data_dir = flname.parent / 'QuickLook'
        if not data_dir.is_dir():
            data_dir.mkdir(mode=0o755)

        # open plot object
        plot = MONplot((data_dir / flname.name).with_suffix('.pdf'))
        plot.set_institute('SRON')
        plot.set_cmap(tol_cmap('rainbow_WhBr'))

        # which images are requested?
        n_img = images.shape[0]
        if args.show_images is None:
            if n_img <= args.max_images:
                indx = range(n_img)
            else:
                indx = np.arange(0, n_img, n_img / args.max_images)
                indx = [int(np.ceil(x)) for x in indx]
        else:
            if args.show_images == 'all':
                indx = range(n_img)
            else:
                indx = [int(x) for x in args.show_images.split(',')]

        med_table_id = np.sort(table_id)[len(table_id) // 2]

        # generate pages in quick-look
        for ii in indx:
            if med_table_id > 0:
                img2d = binned_to_2x2_image(coverage_start,
                                            med_table_id,
                                            images[ii, :]) / 4
            else:
                if images[ii, :].size != 4194304:
                    continue
                img2d = images[ii, :].reshape(2048, 2048)
            img2d /= sci_hk[ii]['REG_NCOADDFRAMES']

            time_str = (
                datetime(year=2020, month=1, day=1)
                + timedelta(seconds=image_time[ii])).strftime('%H:%M:%S.%f')

            figinfo = FIGinfo()
            figinfo.add('coverage_start', date_start)
            figinfo.add('image_time', time_str)
            figinfo.add('exposure_time', exposure_time[ii], fmt='{:f}s')
            figinfo.add('signal_range',
                        (np.nanmin(img2d), np.nanmax(img2d)),
                        fmt='[{:.2f}, {:.2f}]')
            if table_id[ii] > 0:
                suptitle = f'frame [table_id={table_id[ii]}]: {ii}'
            else:
                suptitle = f'frame: {ii}'
            plot.draw_signal(img2d, fig_info=figinfo, title=suptitle)

        # close plot object
        plot.close()


# --------------------------------------------------
if __name__ == '__main__':
    main()
