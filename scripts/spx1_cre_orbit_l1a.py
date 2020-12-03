#!/usr/bin/env python3
"""
This file is part of pySpexCal

https://gitlab.sron.nl/Richardh/pySpexCal

Generate a L1A product with simulated data
- one full science orbit (50% van 5900 sec), max 24 Gbit
- [SPX1_FLIGHT_ideal_202003] 4 scenes (‘ocean_cloud’, ‘ocean_moderate_aot’,
 ‘soil_moderate_aot’ en ‘vegetation_moderate_aot’) at 3 solar zenith angles
 (10, 40 en 70 graden)
- sampling at 3 Hz (at 50 ms, 5x coadding + 16.6 dead-time)
- binned at 2x2 with lineskipped (1024x645)

Implementation:
1. read simulated datasets (original format, lineskip)
2. generate 3x 5900 // 2 images SZA 70, 40, 10, 10, 40, 70
   each SZA filled with all 4 scenes

   >>> rng = np.random.default_rng()
   >>> indx = np.arange(3 * 5900 // 10)
   >>> rng.shuffle(indx)
   >>> indx.repeat(5)

3. fill groups: /image_attributes, /science_data of L1A product
4. fill groups: /navigation_data, /engineering_data
5. leave group: /gse_data empty

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from pyspex.lv1_io import L1Aio

FLNAME_OUT = 'SPX1_TEST_L1A_simulated_orbit.nc'

# - global parameters ------------------------------
MIDNIGHT = datetime(2020, 7, 4, tzinfo=timezone.utc)


# - local functions --------------------------------


# --------------------------------------------------
def initialize_l1a_product(l1a_nav_product: str, bin_tbl: int) -> None:
    """
    Initialize an L1A product with navigation data

    Parameters
    ----------
    l1a_nav_product : string
    bin_tbl : integer
    """
    ckd_dir = '/nfs/SPEXone/share/ckd'
    if not Path(ckd_dir).is_dir():
        ckd_dir = '/data/richardh/SPEXone/share/ckd'
    # only read the latest version of the binning-table CKD
    bin_tbl_fl = sorted(list(Path(ckd_dir).glob('SPX1_OCAL_L1A_TBL_*.nc')))
    if not bin_tbl_fl:
        raise FileNotFoundError('No binning table found')

    # read binning table information
    with h5py.File(bin_tbl_fl[-1], 'r') as fid:
        gid = fid['Table_{:02d}'.format(bin_tbl)]
        # enabled_lines = gid.attrs['enabled_lines']
        n_samples = gid['binning_table'].attrs['valid_max'] + 1

    # read navigation file information
    with h5py.File(l1a_nav_product, 'r') as fid:
        # SC_hkt_block = fid['SC_hkt_block'].size
        nav_records = fid['SC_records'].size
        # quaternion_elements = fid['quaternion_elements'].size
        # vector_elements = fid['vector_elements'].size
        gid = fid['navigation_data']
        att_time = gid['att_time'][:]
        att_quat = gid['att_quat'][:]
        orb_time = gid['orb_time'][:]
        orb_pos = gid['orb_pos'][:]
        if np.mean(np.abs(orb_pos)) < 5000:
            orb_pos *= 1000              # convert km -> meters
        orb_vel = gid['orb_vel'][:]
        if np.mean(np.linalg.norm(orb_vel, axis=1)) < 8:
            orb_vel *= 1000              # convert km -> meters
        adstate = gid['adstate'][:]

    # create L1A product
    # ToDo fix the output product name
    dims = {'number_of_images': None,
            'samples_per_image': n_samples,
            'SC_records': nav_records,
            'hk_packets': None}

    with L1Aio(FLNAME_OUT, dims=dims, inflight=True) as l1a:
        # navigation data
        l1a.set_dset('/navigation_data/att_time', att_time)
        l1a.set_dset('/navigation_data/att_quat', att_quat)
        l1a.set_dset('/navigation_data/orb_time', orb_time)
        l1a.set_dset('/navigation_data/orb_pos', orb_pos)
        l1a.set_dset('/navigation_data/orb_vel', orb_vel)
        l1a.set_dset('/navigation_data/adstate', adstate)

        # Global attributes
        l1a.fill_global_attrs(orbit=12345, bin_size='2.5km')
        l1a.set_attr('history', 'simulated orbit file')

    return min(att_time[-1], orb_time[-1]) - max(att_time[0], orb_time[0])


def add_measurements(l1a_prod_list, repeats: int, sampling=3) -> None:
    """
    Add simulated measurements to L1A product

    Parameters
    ----------
    l1a_prod_list : list of strings
    repeats : integer
    sampling : integer
    """
    # read data
    coad = []
    texp = []
    images = []
    temp_det = []
    temp_house = []
    for flname in l1a_prod_list:
        with h5py.File(flname, 'r') as fid:
            mps_dtype = fid['/science_data/detector_telemetry'].dtype
            coad.append(fid['/image_attributes/nr_coadditions'][:])
            texp.append(fid['/image_attributes/exposure_time'][:])
            images.append(fid['/science_data/detector_images'][:])
            temp_det.append(fid['/engineering_data/temp_detector'][:])
            temp_house.append(fid['/engineering_data/temp_housing'][:])

    coad = np.stack(coad)
    texp = np.stack(texp)
    images = np.stack(images)
    temp_det = np.stack(temp_det)
    temp_house = np.stack(temp_house)

    # create random index for sciences
    rng = np.random.default_rng()
    arr = np.arange(sampling * repeats / coad.shape[1], dtype=int)
    rng.shuffle(arr)
    indx = arr % len(coad)

    # repeat data to fill measurement block
    coad = coad[indx, :].reshape(-1)
    texp = texp[indx, :].reshape(-1)
    samples_per_image = images.shape[-1]
    images = images[indx, :, :].reshape(-1, samples_per_image)
    n_images = images.shape[0]
    temp_det = temp_det[indx[::sampling], :].reshape(-1)
    temp_house = temp_house[indx[::sampling], :].reshape(-1)

    # generate telemetry data
    telemetry = np.zeros(n_images, dtype=mps_dtype)

    with L1Aio(FLNAME_OUT, append=True) as l1a:
        att_time = int(l1a.get_dset('/navigation_data/att_time')[0])

        # define timing of image data
        ibgn = l1a.get_dim('number_of_images')
        print('ibgn (images): ', ibgn)
        img_time = att_time + (ibgn + np.arange(n_images)) / sampling

        # write datasets in /science_data
        l1a.set_dset('/science_data/detector_images', images)
        l1a.set_dset('/science_data/detector_telemetry', telemetry)

        # write datasets in /image_attributes
        l1a.set_dset('/image_attributes/exposure_time', texp)
        l1a.set_dset('/image_attributes/nr_coadditions', coad)
        l1a.set_dset('/image_attributes/image_ID',
                     np.arange(n_images) + ibgn)
        l1a.set_dset('/image_attributes/digital_offset',
                     np.zeros(n_images))
        l1a.set_dset('/image_attributes/binning_table',
                     np.full(n_images, 3))
        l1a.fill_time(img_time, MIDNIGHT)

        # define timing of house-keeping data
        ibgn = l1a.get_dim('hk_packets')
        print('ibgn (engineering): ', ibgn)
        hk_time = att_time + (ibgn + np.arange(temp_det.size))

        # write datasets in /engineering_data
        l1a.set_dset('/engineering_data/HK_tlm_time', hk_time)
        l1a.set_dset('/engineering_data/temp_detector', temp_det)
        l1a.set_dset('/engineering_data/temp_housing', temp_house)


# - main functions --------------------------------
def main():
    """
    main function
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        description='generate SPEXone L1A product simulating a science orbit')
    parser.add_argument('--navigation_data', default=None,
                        help=('navigation data used to initialize the L1A file,'
                              ' provide path to an existing L1A product'))
    parser.add_argument('--binTableID', default=3, type=int,
                        help=('provide binning table ID, default=3'))
    parser.add_argument('--repeats', default=1, type=int,
                        help=('specify number of repeats of each measurement,'
                              ' default=1'))
    parser.add_argument('--measurement_data', default=None, nargs='*',
                        help=('provide path one or more L1A products with,'
                              ' measurement data used to fill the L1A product'))
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    if args.verbose:
        print(args)

    if args.navigation_data is not None:
        duration = initialize_l1a_product(args.navigation_data, args.binTableID)
        print('Coverage of the navigation data is {} seconds'.format(duration))

    if args.measurement_data is not None:
        add_measurements(args.measurement_data, args.repeats, sampling=3)


# --------------------------------------------------
if __name__ == '__main__':
    main()
