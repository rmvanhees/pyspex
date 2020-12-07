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

from pys5p.lib.plotlib import FIGinfo
from pys5p.tol_colors import tol_cmap
from pys5p.s5p_plot import S5Pplot

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
            image_time = fid['image_attributes/image_time'][:]
            nr_coadditions = fid['image_attributes/nr_coadditions'][:]
            exposure_time = fid['image_attributes/exposure_time'][:]
            images = fid['science_data/detector_images'][:] \
                / nr_coadditions[:, None]

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
            if img.size != 4194304:
                continue

            time_str = (
                datetime(year=2020, month=1, day=1)
                + timedelta(seconds=image_time[ii])).strftime('%H:%M:%S.%f')

            figinfo = FIGinfo()
            figinfo.add('coverage_start', date_start)
            figinfo.add('image_time', time_str)
            figinfo.add('exposure_time', exposure_time[ii], fmt='{:f}s')
            figinfo.add('signal_range', [int(img.min()), int(img.max())],
                        fmt='{}')
            plot.draw_signal(img.reshape(2048, 2048), vperc=[1, 99],
                             fig_info=figinfo, sub_title='frame: {}'.format(ii))

        plot.close()


# --------------------------------------------------
if __name__ == '__main__':
    main()
