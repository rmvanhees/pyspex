#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Convert SPEXone binning table in csv format to netCDF4

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

from pyspex import version


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
                              " in csv-format"))
    args = parser.parse_args()
    if args.verbose:
        print(args)

    # define FillValue
    fill_value = 0xFFFFFFFF  # 0X7FFFFFFF

    # initialize netCDF file with binning tables
    date_created = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    tbl_name = 'SPX1_OCAL_L1A_TBL_{}_{}.nc'.format(date_created, '00001')
    fid = Dataset(tbl_name, 'w')
    fid.setncattr('title', 'SPEXone Level-1 binning-tables')
    fid.setncattr('processing_version', version.get())
    fid.setncattr('date_created',
                  datetime.utcnow().isoformat(timespec='seconds'))
    fid.setncattr('Convensions', 'CF-1.6')
    fid.setncattr('project', 'PACE Project')
    fid.setncattr('instrument', 'SPEXone')
    fid.setncattr('institution',
                  'SRON Netherlands Institute for Space Research')

    fid.createDimension('row', 1024)
    fid.createDimension('column', 1024)

    # ----- define tables with table_id equals one -----
    table_id = 1
    # write binning table to netCDF4 file
    gid = fid.createGroup('Table_{:02d}'.format(table_id))
    gid.enabled_lines = np.uint16(1024)
    gid.flex_binned_pixels = np.uint32(0)

    dset = gid.createVariable('binning_table', 'u4', ('row', 'column'),
                              fill_value=fill_value, chunksizes=(128, 128),
                              zlib=True, complevel=1, shuffle=True)
    dset.long_name = 'binning table'
    dset.valid_min = np.uint32(0)
    dset.valid_max = np.uint32(0xfffff)
    dset[:] = np.arange(1024 ** 2, dtype='u4').reshape(1024, 1024)

    dset = gid.createVariable('coadding_table', 'u1', ('row', 'column'),
                              fill_value=0xFF, chunksizes=(128, 128),
                              zlib=True, complevel=1, shuffle=True)
    dset.long_name = 'number of coadditions'
    dset.valid_min = np.uint8(0)
    dset.valid_max = np.uint8(0xff)
    dset[:] = np.ones((1024, 1024), dtype='u1')

    # add binning tables
    for flname in args.file_list:
        flpath = Path(flname)
        if not flpath.is_file():
            raise FileNotFoundError(flname)
        if flpath.suffix != '.csv':
            continue

        # read csv-file
        table = np.loadtxt(flname, delimiter=',', dtype=np.uint32)

        # determine unique addresses and binning-counts
        _, count = np.unique(table, return_counts=True)

        # no valid binning-counts larger than 6
        indx = (count > 6).nonzero()[0]
        for ii in indx:
            table[table == ii] = fill_value

        # convert csv binning table to binning table with FillValues
        frame = np.full(table.shape, 0, dtype='u1')
        frame[table != fill_value] = count[table[table != fill_value]]

        # write binning table to netCDF4 file
        table_id += 1
        gid = fid.createGroup('Table_{:02d}'.format(table_id))
        gid.origin = flpath.name
        gid.enabled_lines = np.uint16(np.sum(np.sum(frame, axis=1) > 0))
        gid.flex_binned_pixels = np.uint32(np.sum(frame > 0))

        dset = gid.createVariable('binning_table', 'u4', ('row', 'column'),
                                  fill_value=fill_value, chunksizes=(128, 128),
                                  zlib=True, complevel=1, shuffle=True)
        dset.long_name = 'binning table'
        dset.valid_min = np.uint32(0)
        dset.valid_max = np.uint32(np.max(table[table <= 0x7ffff]))
        dset[:] = table

        dset = gid.createVariable('coadding_table', 'u1', ('row', 'column'),
                                  fill_value=0xFF, chunksizes=(128, 128),
                                  zlib=True, complevel=1, shuffle=True)
        dset.long_name = 'number of coadditions'
        dset.valid_min = np.uint8(0)
        dset.valid_max = np.uint8(0xff)
        dset[:] = frame

        if args.figure:
            plot = S5Pplot(flpath.with_suffix('.pdf').name)
            plot.draw_signal(table, add_medians=False,
                             vrange=[0, np.max(table[table <= 0x7ffff])],
                             title=flpath.name, sub_title='Binning Table')
            plot.set_cmap(tol_cmap('rainbow_discrete', 5))
            plot.draw_signal(frame, add_medians=False,
                             title=flpath.name, sub_title='Number of Coaddings')
            plot.close()

    # read line-skip definitions
    for flname in args.file_list:
        flpath = Path(flname)
        if not flpath.is_file():
            raise FileNotFoundError(flname)
        if flpath.suffix != '.dat':
            continue

        # read line-skip definition
        data = np.loadtxt(flname, delimiter=',', dtype=np.uint8)
        line_indx = (data == 1).nonzero()[0]
        indx = np.arange(1024 * line_indx.size).reshape(line_indx.size, -1)

        table = np.full((1024, 1024), fill_value, dtype='u4')
        table[line_indx, :] = indx

        frame = np.zeros((1024, 1024), dtype='u1')
        frame[line_indx, :] = 1

        # write binning table to netCDF4 file
        table_id += 1
        gid = fid.createGroup('Table_{:02d}'.format(table_id))
        gid.origin = flpath.name
        gid.enabled_lines = np.uint16(line_indx.size)
        gid.flex_binned_pixels = np.uint32(0)

        dset = gid.createVariable('binning_table', 'u4', ('row', 'column'),
                                  fill_value=fill_value, chunksizes=(128, 128),
                                  zlib=True, complevel=1, shuffle=True)
        dset.long_name = 'binning table'
        dset.valid_min = np.uint32(0)
        dset.valid_max = np.uint32(np.max(table[table != fill_value]))
        dset[:] = table

        dset = gid.createVariable('coadding_table', 'u1', ('row', 'column'),
                                  fill_value=0xFF, chunksizes=(128, 128),
                                  zlib=True, complevel=1, shuffle=True)
        dset.long_name = 'number of coadditions'
        dset.valid_min = np.uint8(0)
        dset.valid_max = np.uint8(0xff)
        dset[:] = frame

        if args.figure:
            plot = S5Pplot(flpath.with_suffix('.pdf').name)
            plot.draw_signal(table, add_medians=False,
                             vrange=[0, np.max(table[table <= 0x7ffff])],
                             title=flpath.name, sub_title='Binning Table')
            plot.set_cmap(tol_cmap('rainbow_discrete', 5))
            plot.draw_signal(frame, add_medians=False,
                             title=flpath.name, sub_title='Number of Coaddings')
            plot.close()

    fid.close()


# --------------------------------------------------
if __name__ == '__main__':
    main()
