"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation to convert SPEXone DEM measurements to L1A format

Copyright (c) 2019 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from datetime import datetime
from pathlib import Path

import numpy as np

from pyspex import spx_product
from pyspex.ccsds_io import CCSDSio
from pyspex.l1a_io import L1Aio

# - global parameters ------------------------------
LEAP_SECONDS = 27


# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """
    main function
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description='create SPEXone L1A product from CCSDS packages (L0)')
    parser.add_argument('l0_product', default=None,
                        help='name of SPEXone DEM L0 product')
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    if args.verbose:
        print(args)

    if not Path(args.l0_product).is_file():
        raise FileNotFoundError(
            'File {} does not exist'.format(args.l0_product))

    ccsds = CCSDSio(args.l0_product, verbose=args.verbose)
    packet_list = ccsds.read()

    utc_sec = []
    frac_sec = []
    mps_data = []
    image_id = []
    images = []
    for packet in packet_list:
        utc_sec.append(packet['secondary_header']['tai_sec'] - LEAP_SECONDS)
        frac_sec.append(packet['secondary_header']['sub_sec'] / 2**16)
        image_id.append(packet['primary_header']['sequence'] & 0x3fff)
        mps_data.append(packet['mps'])
        images.append(packet['image_data'])

    utc_sec = np.array(utc_sec, dtype=np.uint32)
    frac_sec = np.array(frac_sec)
    mps_data = np.array(mps_data)
    images = np.array(images)
    if args.verbose:
        print(images.size
              if images.ndim == 1 else images.shape[1], images.shape)
        print(images[:, 0], images[:, -1])

    # generate name of L1A product
    utc_sec_launch = datetime(2022, 11, 2) - datetime(1970, 1, 1)
    if utc_sec[0] < utc_sec_launch.total_seconds():
        prod_name = spx_product.prod_name(
            utc_sec[0], msm_id=Path(args.l0_product).stem)
        inflight = False
    else:
        prod_name = spx_product.prod_name(utc_sec[0])
        inflight = True

    # pylint: disable=unsubscriptable-object
    n_frame = 1 if images.ndim == 1 else images.shape[0]
    n_sample = images.size if images.ndim == 1 else images.shape[1]
    dims = {'number_of_images' : n_frame,
            'samples_per_image' : n_sample,
            'SC_records' : None,
            'tlm_packets' : None}
    #
    # Generate L1A product
    #   ToDo: correct value for measurement_type & viewport
    with L1Aio(prod_name, dims, inflight) as l1a:
        l1a.set_attr('input_file', Path(args.l0_product).name)

        l1a.fill_time(utc_sec, frac_sec)
        l1a.fill_hk_time(utc_sec, frac_sec)
        l1a.fill_images(images)
        l1a.fill_mps(mps_data)

        l1a.set_dset('/image_attributes/image_ID', image_id)


# --------------------------------------------------
if __name__ == '__main__':
    main()
