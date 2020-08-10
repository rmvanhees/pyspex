"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation to convert SPEXone DEM measurements to L1A format

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from pyspex import spx_product
from pyspex.ccsds_io import CCSDSio
from pyspex.lv1_io import L1Aio

# - global parameters ------------------------------
LAUNCH_DATE = datetime(2022, 11, 2, tzinfo=timezone.utc)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LEAP_SECONDS = 27 # ToDo use package leapseconds.py


# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """
    main function
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='create SPEXone Level-1A product from CCSDS packages (L0)')
    parser.add_argument('l0_product', default=None,
                        help='name of SPEXone ICU Level-0 product')
    parser.add_argument('--tmtc_issue', default=12, type=int, help=(
        'Specify issue of the TMTC handbook, default=12\n'
        ' < 12: Science data header format before 15-May-2020\n'
        ' 12: append 6 bytes timestamp after MPS data [15-May-2020]'))
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    if args.verbose:
        print(args)

    if not Path(args.l0_product).is_file():
        raise FileNotFoundError(
            'File {} does not exist'.format(args.l0_product))

    ccsds = CCSDSio(args.l0_product, tmtc_issue=args.tmtc_issue,
                    verbose=args.verbose)
    packet_list = ccsds.read()

    tstamp = []
    mps_data = []
    image_id = []
    images = []
    for packet in packet_list:
        utc_sec = packet['secondary_header']['tai_sec'] - LEAP_SECONDS
        frac_sec = packet['secondary_header']['sub_sec'] / 2**16
        tstamp.append(EPOCH + timedelta(seconds=utc_sec + frac_sec))

        image_id.append(packet['primary_header']['sequence'] & 0x3fff)
        mps_data.append(packet['mps'])
        images.append(packet['image_data'])

    image_id = np.array(image_id)
    mps_data = np.array(mps_data)
    images = np.array(images)
    if args.verbose:
        print(images.size
              if images.ndim == 1 else images.shape[1], images.shape)

    # generate name of L1A product
    if tstamp[0] < LAUNCH_DATE:
        prod_name = spx_product.prod_name(
            tstamp[0], msm_id=Path(args.l0_product).stem)
        inflight = False
    else:
        prod_name = spx_product.prod_name(tstamp[0])
        inflight = True

    # pylint: disable=unsubscriptable-object
    n_frame = 1 if images.ndim == 1 else images.shape[0]
    n_sample = images.size if images.ndim == 1 else images.shape[1]
    dims = {'number_of_images' : n_frame,
            'samples_per_image' : n_sample,
            'SC_records' : None,
            'tlm_packets' : None}

    # convert timestamps to seconds per day
    secnds = np.empty(len(tstamp), dtype=float)
    midnight = tstamp[0].replace(hour=0, minute=0, second=0, microsecond=0)
    for ii, tval in enumerate(tstamp):
        secnd = (tval - midnight).total_seconds()
        # time-of-measurement provided at end of integration time
        # secnd -= texp[ii] * mps_data[ii]['REG_NCOADDFRAMES']
        secnds[ii] = secnd

    #
    # Generate L1A product
    #   ToDo: correct value for measurement_type & viewport
    with L1Aio(prod_name, dims=dims, inflight=inflight) as l1a:
        l1a.set_dset('/science_data/detector_images', images)
        l1a.fill_mps(mps_data)

        l1a.fill_time(secnds, midnight)
        l1a.set_dset('/image_attributes/image_ID', image_id)

        l1a.set_dset('/engineering_data/HK_tlm_time', secnds)

        # Global attributes
        l1a.fill_global_attrs()
        l1a.set_attr('input_file', Path(args.l0_product).name)


# --------------------------------------------------
if __name__ == '__main__':
    main()
