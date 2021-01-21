"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Quick and dirty script to generate simple Quick-Look figures.

Copyright (c) 2019-2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from pys5p.lib.plotlib import FIGinfo
from pys5p.tol_colors import tol_cmap
from pys5p.s5p_plot import S5Pplot

from pyspex.binning_tables import BinningTables


# --------------------------------------------------
def bin_table_id(binning_table_start) -> int:
    """
    Convert the start address of the binning table to an ID [1, 2, ...]

    Notes
    -----
    'Diagnostic mode' does not use a binning table. These measurements can be
    identified by the following register settings: FRAME_MODE=1 & OUTPMODE=3

    'Science mode' using binning table to decrease the data rate. These
    measurements  can be identified by the following register settings:
    FRAME_MODE=2 & OUTPMODE=1

    The first binning table is written to address 0x80000000, each table has
    a size of 1024x1024 pointers of 4 bytes (0x400000)
    """
    return 1 + (binning_table_start - 0x80000000) // 0x400000


def binned_to_2x2_image(table_id: int, img_binned):
    """
    Convert binned detector data to image (1024, 1024)
    """
    try:
        bin_ckd = BinningTables()
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
            date_start = fid.attrs['time_coverage_start']
            image_time = fid['/image_attributes/image_time'][:]
            exposure_time = fid['/image_attributes/exposure_time'][:]
            table_id = fid['/image_attributes/binning_table'][:]
            sci_hk = fid['/science_data/detector_telemetry'][:]
            images = fid['/science_data/detector_images'][:]

        try:
            date_start = date_start.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
        else:
            date_start = date_start.split('+')[0]

        data_dir = flname.parent / 'QuickLook'
        if not data_dir.is_dir():
            data_dir.mkdir(mode=0o755)

        # open plot object
        plot = S5Pplot((data_dir / flname.name).with_suffix('.pdf'))
        plot.set_cmap(tol_cmap('rainbow_WhBr_condense'))

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

        # generate pages in quick-look
        for ii in indx:
            img = images[ii]
            if table_id[ii] > 0:
                img2d = binned_to_2x2_image(table_id[ii], img)
            else:
                if img.size != 4194304:
                    continue
                img2d = img.reshape(2048, 2048) / sci_hk[ii]['REG_NCOADDFRAMES']

            time_str = (
                datetime(year=2020, month=1, day=1)
                + timedelta(seconds=image_time[ii])).strftime('%H:%M:%S.%f')

            figinfo = FIGinfo()
            figinfo.add('coverage_start', date_start)
            figinfo.add('image_time', time_str)
            figinfo.add('exposure_time', exposure_time[ii], fmt='{:f}s')
            figinfo.add('signal_range',
                        [int(np.nanmin(img2d)), int(np.nanmax(img2d))],
                        fmt='{}')
            if table_id[ii] > 0:
                suptitle = 'frame [table_id={}]: {}'.format(table_id[ii], ii)
            else:
                suptitle = 'frame: {}'.format(ii)
            plot.draw_signal(img2d, vperc=[1, 99], fig_info=figinfo, 
                             sub_title=suptitle)
        # close plot object
        plot.close()


# --------------------------------------------------
if __name__ == '__main__':
    main()
