#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Create new SPEXone Level-1A product with selected data from original

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from pyspex.lv1_io import L1Aio


# --------------------------------------------------
def inv_sec_of_day(reference_day, sec_of_day,
                   epoch=datetime(1970, 1, 1, tzinfo=timezone.utc)):
    """
    Convert seconds after midnight to CCSDS timestamps

    Parameters
    ----------
    reference_day :  datetime object
    sec_of_day :  ndarray
    epoch :  datetime object, optional

    Returns
    -------
    tuple holding CCSDS timestamps
    """
    offs = np.uint32((reference_day - epoch).total_seconds())
    ccsds_sec = sec_of_day.astype('u4')
    ccsds_subsec = (65536 * (sec_of_day - ccsds_sec)).astype('u2')

    return offs + ccsds_sec, ccsds_subsec


# --------------------------------------------------
def main():
    """
    Main function of this module
    """
    parser = argparse.ArgumentParser(
        description=('Copy selected data from one SPEXone L1A product'
                     ' into a new SPEXone L1A product'))
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('--mps_id', nargs='*', type=int, default=None,
                        help='select on MPS-ID [comma separated?]')
    parser.add_argument('--mon_type', default=None,
                        help=('Specify monitoring type identifier: '
                              'MON-DARK, MON-NOISE, MON-NLIN, ...'))
    # parser.add_argument('--time', nargs=2, default=None,
    #                    help='select on image time [start, end]')
    # parser.add_argument('--', default=None, help='')
    parser.add_argument('--out', default='.',
                        help=('name of directory to store the new Level-1A'
                              ' product, default: current working directory'))
    parser.add_argument('l1a_product', default=None,
                        help='name of SPEXone Level-1A product')
    args = parser.parse_args()
    if args.verbose:
        print(args)

    l1a_product = Path(args.l1a_product)
    if not l1a_product.is_file():
        raise FileNotFoundError(f'File {args.l1a_product} does not exist')
    # ToDo: check if SPEXone Level-1a product
    # ToDo: implement check on data product

    out_dir = Path(args.out)
    if not out_dir.is_dir():
        out_dir.mkdir(mode=0o755, parents=True)

    # ----- read data from orignal product -----
    # pylint: disable=no-member, unsubscriptable-object
    with h5py.File(l1a_product) as fid:
        # read image data, detector telemetry and image attributes
        # - datasets: img_data, img_hk, img_id, img_sec, img_subsec
        img_sec = fid['/image_attributes/image_CCSDS_sec'][:]
        img_subsec = fid['/image_attributes/image_CCSDS_subsec'][:]
        # obtain reference date
        units = fid['/image_attributes/image_time'].attrs['units']
        reference_day = datetime.fromisoformat(units[14:].decode('ascii'))
        sec_of_day = fid['/image_attributes/image_time'][:]
        img_sec, img_subsec = inv_sec_of_day(reference_day, sec_of_day)
        img_id = fid['/image_attributes/image_ID'][:]
        img_data = fid['/science_data/detector_images'][:]
        img_hk = fid['/science_data/detector_telemetry'][:]

        # read engineering data
        # - datasets: nomhk_data, nomhk_sec, nomhk_subsec
        # - datasets: demhk_data
        nomhk_data = fid['/engineering_data/NomHK_telemetry'][:]
        if nomhk_data.size > 0:
            # obtain reference date
            units = fid['/engineering_data/HK_tlm_time'].attrs['units']
            print(units)
            reference_day = datetime.fromisoformat(units[14:].decode('ascii'))
            print(reference_day)
            sec_of_day = fid['/engineering_data/HK_tlm_time'][:]
            nomhk_sec, nomhk_subsec = inv_sec_of_day(reference_day, sec_of_day)

        demhk_data = fid['/engineering_data/DemHK_telemetry'][:]

        # read additional attributes:
        # - inflight
        # - EGSE/OGSE, ...
        inflight = not fid.attrs['institution'].startswith(b'SRON')
        # gse_data = 'gse_data' in fid

    # define dimensions
    dims = {'number_of_images': img_data.shape[0],
            'samples_per_image': img_data.shape[1],
            'hk_packets': nomhk_data.size}

    # ----- perform data selection -----
    # ToDo: implement data selection

    # ----- now we can update the name of the output product -----
    # - because the production time has changed
    # - and when coverage time is changed
    if (out_dir / l1a_product.name).is_file() \
       and l1a_product.samefile(out_dir / l1a_product.name):
        raise OSError('Output will overwrite original product')

    # ----- write new output product with selected data -----
    with L1Aio(out_dir / l1a_product.name, dims=dims) as l1a:
        # write image data, detector telemetry and image attributes
        l1a.fill_science(img_data, img_hk, img_id)
        l1a.fill_time(img_sec, img_subsec, group='image_attributes')

        # write engineering data
        if nomhk_data.size > 0:
            l1a.fill_nomhk(nomhk_data)
            l1a.fill_time(nomhk_sec, nomhk_subsec, group='engineering_data')

        if demhk_data.size > 0:
            l1a.fill_demhk(demhk_data)

        # write global attributes
        l1a.fill_global_attrs(inflight=inflight)
        # l1a.set_attr('input_files', [Path(x).name for x in args.file_list])

    # copy group with EGSE/OGSE data
    # *** DO NOT USE: BREAKS NETCDF4 FORMAT ***
    # if gse_data:
    #     print('copy EGSE/OGSE data')
    #     with h5py.File(l1a_product) as fid_in:
    #         with h5py.File(out_dir / l1a_product.name, 'r+') as fid_out:
    #             fid_out.copy(fid_in['gse_data'], 'gse_data')


# --------------------------------------------------
if __name__ == '__main__':
    main()
