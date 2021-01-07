"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Quick and dirty script to generate simple Quick-Look figures.

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import sys

from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np

from pys5p.lib.plotlib import FIGinfo
from pys5p.tol_colors import tol_cmap
from pys5p.s5p_plot import S5Pplot

TEST_BINNING = False

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


def binned_to_2x2_image(table_id: int, img_data):
    """
    Convert binned detector data to 2x2 binned image (1024x1024)
    """
    ckd_dir = '/nfs/SPEXone/share/ckd'
    if not Path(ckd_dir).is_dir():
        ckd_dir = '/data/richardh/SPEXone/share/ckd'

    # only read the latest version of the binning-table CKD
    bin_tbl_ckd = sorted(list(Path(ckd_dir).glob('SPX1_OCAL_L1A_TBL_*.nc')))
    if not bin_tbl_ckd:
        raise FileNotFoundError('No CKD with binning tables found')

    with h5py.File(bin_tbl_ckd[-1], 'r') as fid:
        coad_table = fid['/Table_{:02d}/coadding_table'.format(table_id)][:]

    # write binned data to 2-D grid
    data = np.zeros((1024, 1024), dtype='f4')
    mask = coad_table > 0
    data[mask] = img_data

    # correct data for number of coaddings (pixel dependent)
    data[mask] /= coad_table[mask]

    return data


# --------------------------------------------------
def main():
    """
    Main function of this module
    """
    if len(sys.argv) > 1:
        file_list = [Path(xx) for xx in sys.argv[1:]]
    else:
        file_list = Path('.').glob('*.nc')

    for flname in sorted(file_list):
        print(flname)

        with h5py.File(flname, 'r') as fid:
            date_start = fid.attrs['time_coverage_start']
            image_time = fid['/image_attributes/image_time'][:]
            exposure_time = fid['/image_attributes/exposure_time'][:]
            sci_hk = fid['/science_data/detector_telemetry'][:]
            images = fid['/science_data/detector_images'][:]

        if TEST_BINNING:
            sci_hk['REG_FULL_FRAME'] = 2
            sci_hk['REG_CMV_OUTPUTMODE'] = 1
            sci_hk['REG_BINNING_TABLE_START'] = 0x80000000 + 0x400000
            # de gebruikte binning tabel bevat data uit 416023 pixels
            # dus selecteer ik net zoveel pixels uit het full-frame image
            images = images[:, 20480:20480+416023]
                
        try:
            date_start = date_start.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
        else:
            date_start = date_start.split('+')[0]

        data_dir = flname.parent / 'QuickLook'
        if not data_dir.is_dir():
            data_dir.mkdir(mode=0o755)

        plot = S5Pplot((data_dir / flname.name).with_suffix('.pdf'))
        plot.set_cmap(tol_cmap('rainbow_WhBr_condense'))
        for ii, img in enumerate(images):
            if sci_hk[ii]['REG_FULL_FRAME'] == 2 \
               and sci_hk[ii]['REG_CMV_OUTPUTMODE'] == 1:
                table_id = bin_table_id(sci_hk[ii]['REG_BINNING_TABLE_START'])
                img = binned_to_2x2_image(table_id, img)
            elif sci_hk[ii]['REG_FULL_FRAME'] == 1 \
                 and sci_hk[ii]['REG_CMV_OUTPUTMODE'] == 3:
                if img.size != 4194304:
                    continue
                img = img.reshape(2048, 2048) / sci_hk[ii]['REG_NCOADDFRAMES']
            else:
                raise RuntimeError('unknown FRAME_MODE, OUTPMODE combination')

            time_str = (
                datetime(year=2020, month=1, day=1)
                + timedelta(seconds=image_time[ii])).strftime('%H:%M:%S.%f')

            figinfo = FIGinfo()
            figinfo.add('coverage_start', date_start)
            figinfo.add('image_time', time_str)
            figinfo.add('exposure_time', exposure_time[ii], fmt='{:f}s')
            figinfo.add('signal_range', [int(img.min()), int(img.max())],
                        fmt='{}')
            plot.draw_signal(img, vperc=[1, 99],
                             fig_info=figinfo, sub_title='frame: {}'.format(ii))

        plot.close()


# --------------------------------------------------
if __name__ == '__main__':
    main()
