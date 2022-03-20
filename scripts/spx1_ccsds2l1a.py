#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation to convert SPEXone DEM measurements to L1A format

Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from pyspex import spx_product
from pyspex.lib.before_launch import before_launch
from pyspex.lib.tmtc_def import tmtc_def
from pyspex.ccsds_io import CCSDSio
from pyspex.lv1_io import L1Aio

# - global parameters ------------------------------
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
LEAP_SECONDS = 0  # only in-flight the CCSDS packages have TAI timestamps(?)

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
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--select', default='all',
                        choices=['all', 'binned', 'fullFrame'])
    parser.add_argument('--datapath', type=Path, default=Path('.'),
                        help="directory to store the results")
    parser.add_argument('msmt_id', type=Path,
                        help=('name of the measurement without extension'
                              '(full path). The names of the telemetry data'
                              ' are expected to be: msmt_id+"_hk" (also'
                              ' without extension).'))
    args = parser.parse_args()
    if args.verbose:
        print(args)

    # Read Science packages
    sci_files = sorted(args.msmt_id('.[0123456789]')) \
        + sorted(args.msmt_id('.?[0123456789]'))

    packets = ()
    with CCSDSio(sci_files) as ccsds:
        while True:
            packet = ccsds.read_packet()
            # print(ccsds.fp.tell(), packet is None, ccsds.packet_length)
            if packet is None or ccsds.packet_length == 0:
                break

            if args.debug:
                print('[DEBUG]: ', ccsds)
            packets += (packet[0],)
    if args.verbose:
        print('[INFO]: number of CCSDS packets ', len(packets))

    # combine segmented packages
    science_tm = ccsds.science_tm(packets)
    del packets
    if args.verbose:
        print('[INFO]: number of Science images ', len(science_tm))

    num_packets = 0
    mps_length = 0
    mps_list = []
    for packet in science_tm:
        mps_id = packet['science_hk']['MPS_ID']
        mps_sz = packet['science_hk']['IMRLEN'] // 2
        if mps_id == 0:
            continue

        if args.select == 'fullFrame' \
           and packet['science_hk']['IMRLEN'] != 8388608:
            continue

        if args.select == 'binned'\
           and packet['science_hk']['IMRLEN'] == 8388608:
            continue

        mps_length = max(mps_sz, mps_length)
        if not mps_id in mps_list:
            mps_list.append(mps_id)
        num_packets += 1
    if args.verbose:
        print(mps_length, mps_list)
        print('[INFO]: number of Science images ', num_packets)

    # read NomHK packages
    hk_files = sorted(args.msmt_id('_hk.[0123456789]'))

    packets = ()
    with CCSDSio(hk_files) as ccsds:
        while True:
            packet = ccsds.read_packet()
            # print(ccsds.fp.tell(), packet is None, ccsds.packet_length)
            if packet is None or ccsds.packet_length == 0:
                break

            if args.debug:
                print('[DEBUG]: ', ccsds)
            packets += (packet[0],)
    if args.verbose:
        print('[INFO]: number of telemetry CCSDS packets ', len(packets))

    # select NomHK packages
    num_packets = 0
    nomhk_tm = ccsds.select_tm(packets, 0x320)
    for packet in nomhk_tm:
        if packet['nominal_hk']['MPS_ID'] not in mps_list:
            continue
        num_packets += 1

    if args.debug or args.verbose:
        print('[INFO]: number of NomHK packages ', num_packets)

    # select DemHK packages
    if args.select == 'all':
        demhk_tm = ccsds.select_tm(packets, 0x322)
        if args.debug or args.verbose:
            print('[INFO]: number of DemHK packages ', len(demhk_tm))

    if args.debug:
        return

    # Exit because we need Science data to create a valid L1A product
    if not science_tm:
        print('[WARNING]: no science data found, exit')
        return

    # extract timestaps, telemetry and image data from Science data
    img_sec = np.empty(len(science_tm), dtype='u4')
    img_subsec = np.empty(len(science_tm), dtype='u2')
    img_id = np.empty(len(science_tm), dtype='u4')
    img_hk = np.empty(len(science_tm), dtype=np.dtype(tmtc_def(0x350)))
    img_data = []
    for ii, packet in enumerate(science_tm):
        if packet['science_hk']['ICUSWVER'] > 0x123:
            img_sec[ii] = packet['icu_time']['tai_sec']
            img_subsec[ii] = packet['icu_time']['sub_sec']
        else:
            img_sec[ii] = packet['packet_header']['tai_sec']
            img_subsec[ii] = packet['packet_header']['sub_sec']
        img_id[ii] = packet['packet_header']['sequence'] & 0x3fff
        img_hk[ii] = packet['science_hk']
        img_data.append(packet['image_data'])
    img_data = np.array(img_data)

    if np.all(img_hk['ICUSWVER'] == 0x123):
        # fix bug in sub-seconds
        us100 = np.round(10000 * img_subsec.astype(float) / 65536)
        buff = us100 + img_sec - 10000
        us100 = buff.astype('u8') % 10000
        img_subsec = ((us100 << 16) // 10000).astype('u2')

    # extract timestaps and telemetry of NomHK data
    if nomhk_tm:
        nomhk_sec = np.empty(len(nomhk_tm), dtype='u4')
        nomhk_subsec = np.empty(len(nomhk_tm), dtype='u2')
        nomhk_data = np.empty(len(nomhk_tm), dtype=np.dtype(tmtc_def(0x320)))
        for ii, packet in enumerate(nomhk_tm):
            nomhk_sec[ii] = packet['packet_header']['tai_sec']
            nomhk_subsec[ii] = packet['packet_header']['sub_sec']
            nomhk_data[ii] = packet['nominal_hk']

        if np.all(img_hk['ICUSWVER'] == 0x123):
            # fix bug in sub-seconds
            us100 = np.round(10000 * nomhk_subsec.astype(float) / 65536)
            buff = us100 + nomhk_sec - 10000
            us100 = buff.astype('u8') % 10000
            med = np.median(us100)
            indx = np.where(np.abs(us100 - med) > 1000)[0]
            us100[indx] = med
            nomhk_subsec = ((us100 << 16) // 10000).astype('u2')

    # if we have DemHK data then demhk_data should be equal in size with NomHK
    if demhk_tm:
        demhk_data = np.zeros(len(nomhk_tm), dtype=np.dtype(tmtc_def(0x322)))
        for ii in range(min(len(nomhk_tm), len(demhk_tm))):
            demhk_data[ii] = demhk_tm[ii]['detector_hk']

    # generate name of L1A product
    # ToDo: rewrite this section because all onground calibration measurements
    #       which are in CCSDS format will have a different filename convesion
    #       then the data in ST3 format. Thus the distinction between inflight
    #       and onground is not accurate, the distionction should be between
    #       the data formats: CCSDS and ST3.
    tstamp0 = EPOCH + timedelta(seconds=int(img_sec[0]))
    if before_launch(tstamp0):
        msm_id = args.msmt_id.name
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
    n_frame = 1 if img_data.ndim == 1 else img_data.shape[0]
    n_sample = img_data.size if img_data.ndim == 1 else img_data.shape[1]
    if args.verbose:
        print(f'[INFO]: dimension of images [{n_frame},{n_sample}]')
    dims = {'number_of_images': n_frame,
            'samples_per_image': n_sample,
            'hk_packets': len(nomhk_tm),
            'SC_records': None}

    # Generate L1A product
    with L1Aio(args.datapath / prod_name, dims=dims) as l1a:
        # write image data, detector telemetry and image attributes
        l1a.fill_science(img_data, img_hk, img_id)
        l1a.fill_time(img_sec, img_subsec, group='image_attributes')

        # write engineering data
        if nomhk_tm:
            l1a.fill_nomhk(nomhk_data)
            l1a.fill_time(nomhk_sec, nomhk_subsec, group='engineering_data')

        if demhk_tm:
            l1a.fill_demhk(demhk_data)

        # write global attributes
        l1a.fill_global_attrs(inflight=inflight)
        l1a.set_attr('input_files', [Path(x).name for x in sci_files])


# --------------------------------------------------
if __name__ == '__main__':
    main()
