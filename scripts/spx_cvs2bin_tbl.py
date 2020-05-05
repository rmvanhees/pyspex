"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Convert SPEXone binning table in cvs format to netCDF4

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from pys5p.s5p_plot import S5Pplot
from pys5p.tol_colors import tol_cmap

from pyspex.lib import sw_version


# - global parameters ------------------------------


# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """
    main program to generate a netCDF4 file with a SPEXone binning table
    """
    parser = argparse.ArgumentParser(
        description=('create SPEXone L1A product from DEM measurement(s)'))
    parser.add_argument('--figure', default=False, action='store_true',
                        help='generate (PDF) figure of binning table(s)')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('file_list', nargs='+',
                        help=("provide path to SPEXone binning table(s)"
                              " in cvs-format"))
    args = parser.parse_args()
    if args.verbose:
        print(args)

    # define FillValue
    fill_value = 0X7FFFFFFF

    # initialize netCDF file with binning tables
    date_created = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    tbl_name = 'SPX1_OCAL_L1A_TBL_{}_{}.nc'.format(date_created, '00001')
    fid = Dataset(tbl_name, 'w')
    fid.setncattr('title', 'SPEXone Level-1 binning-tables')
    fid.setncattr('processing_version', sw_version.get())
    fid.setncattr('date_created',
                  datetime.utcnow().isoformat(timespec='seconds'))
    fid.setncattr('Convensions', 'CF-1.6')
    fid.setncattr('project', 'PACE Project')
    fid.setncattr('instrument', 'SPEXone')
    fid.setncattr('institution',
                  'SRON Netherlands Institute for Space Research"')

    fid.createDimension('row', 1024)
    fid.createDimension('column', 1024)

    # add binning tables
    for table_id, flname in enumerate(args.file_list):
        flpath = Path(flname)
        if not flpath.is_file():
            raise FileNotFoundError(flname)

        # read cvs-file
        table = np.loadtxt(flname, delimiter=',', dtype=int)

        # determine unique addresses and binning-counts
        _, count = np.unique(table, return_counts=True)

        # no valid binning-counts larger than 6
        indx = np.where(count > 6)[0]
        for ii in indx:
            table[table == ii] = fill_value

        # convert cvs binning table to binning table with FillValues
        frame = np.full(table.shape, np.nan)
        frame[table != fill_value] = count[table[table != fill_value]]

        # write binning table to netCDF4 file
        gid = fid.createGroup('Table_{:02d}'.format(table_id + 1))
        dset = gid.createVariable('binning_table', 'i4', ('column', 'row'),
                                  fill_value=fill_value, chunksizes=(128, 128),
                                  zlib=True, complevel=1, shuffle=True)
        dset.comment = 'provide description of this table'
        dset.origin = flpath.name
        dset.valid_min = np.int32(0)
        dset.valid_max = np.int32(0x7ffff)
        dset[:] = frame

        if args.figure:
            plot = S5Pplot(flpath.with_suffix('.pdf').name)
            plot.set_cmap(tol_cmap('rainbow_discrete', 5))
            plot.draw_signal(frame, add_medians=False)
            plot.close()

    fid.close()


# --------------------------------------------------
if __name__ == '__main__':
    main()
