#!/usr/bin/env python3
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
LEAP_SECONDS = 0  # only in-flight the CCSDS packages have TAI timestamps

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
    parser.add_argument('file_list', nargs='+',
                        help=("provide names of one or more files with"
                              " CCSDS packages of the same measurement"))
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    if args.verbose:
        print(args)

    args.file_list = [x for x in args.file_list if not x.endswith('.H')]

    packets = ()
    with CCSDSio(args.file_list) as ccsds:
        while True:
            packet = ccsds.read_packet()
            if packet is None or ccsds.packet_length == 0:
                break

            packets += (packet[0],)
            if args.debug:
                print('[DEBUG]: ', ccsds)

    # select NomHK packages
    nomhk_tm = ccsds.nomhk_tm(packets)
    # combine segmented packages
    science_tm = ccsds.science_tm(packets)
    if args.debug or args.verbose:
        print('[INFO]: number of CCSDS packets ', len(packets))
        print('[INFO]: number of Science images ', len(science_tm))
        print('[INFO]: number of NomHK packages ', len(nomhk_tm))
    del packets

    if args.debug:
        return

    # Exit because we need Science data to create a valid L1A product
    if not science_tm:
        print('[WARNING]: no science data found, exit')
        return

    # extract timestaps, image data & attributes from Science packages
    img_sec = []
    img_usec = []
    img_id = []
    images = []
    mps_data = []
    for packet in science_tm:
        img_sec.append(packet['secondary_header']['tai_sec'])
        img_usec.append(packet['secondary_header']['sub_sec'])
        img_id.append(packet['primary_header']['sequence'] & 0x3fff)

        mps_data.append(packet['mps'])
        images.append(packet['image_data'])

    img_sec = np.array(img_sec)
    img_usec = np.array(img_usec)
    img_id = np.array(img_id)
    mps_data = np.array(mps_data)
    images = np.array(images)
    if args.verbose:
        print('[INFO]: dimension of image ', images.size
              if images.ndim == 1 else images.shape[1], images.shape)

    # generate name of L1A product
    tstamp0 = EPOCH + timedelta(seconds=int(img_sec[0]))
    print(tstamp0, LAUNCH_DATE)
    if tstamp0 < LAUNCH_DATE:
        msm_id = Path(args.file_list[0]).stem
        try:
            new_date = datetime.strptime(
                msm_id[-22:], '%y-%j-%H:%M:%S.%f').strftime('%Y%m%dT%H%M%S.%f')
        except ValueError:
            pass
        else:
            msm_id = msm_id[:-22] + new_date

        prod_name = spx_product.prod_name(tstamp0, msm_id=msm_id)
        inflight = False
    else:
        prod_name = spx_product.prod_name(tstamp0)
        inflight = True

    # pylint: disable=unsubscriptable-object
    n_frame = 1 if images.ndim == 1 else images.shape[0]
    n_sample = images.size if images.ndim == 1 else images.shape[1]
    dims = {'number_of_images': n_frame,
            'samples_per_image': n_sample,
            'hk_packets': len(nomhk_tm),
            'SC_records': None}

    # convert timestamps NomHK to seconds-per-day
    hk_sec = []
    hk_usec = []
    hk_data = []
    for packet in nomhk_tm:
        hk_sec.append(packet['secondary_header']['tai_sec'])
        hk_usec.append(packet['secondary_header']['sub_sec'])
        hk_data.append(packet['nominal_hk'])

    hk_sec = np.array(hk_sec)
    hk_usec = np.array(hk_usec)
    hk_data = np.array(hk_data)

    # Generate L1A product
    with L1Aio(prod_name, dims=dims, inflight=inflight) as l1a:
        # write image data
        l1a.set_dset('/science_data/detector_images', images)

        # write detector telemetry and image attributes
        l1a.fill_mps(mps_data)
        l1a.fill_time(img_sec, img_usec, group='image_attributes')
        l1a.set_dset('/image_attributes/image_ID', img_id)

        # write engineering data
        l1a.fill_time(hk_sec, hk_usec, group='engineering_data')
        l1a.fill_nomhk(hk_data)

        # write global attributes
        l1a.fill_global_attrs()
        l1a.set_attr('input_files',
                     [Path(x).name for x in args.file_list])


# --------------------------------------------------
if __name__ == '__main__':
    main()
