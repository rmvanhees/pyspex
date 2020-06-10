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
import os

import h5py
import numpy as np

from pys5p.biweight import biweight

from pyspex import spx_product
from pyspex.dem_io import DEMio
from pyspex.l1a_io import L1Aio
from pyspex.mps_def import MPSdef

# - global parameters ------------------------------


# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """
    main program to illustate the creation of a L1A calibration product
    """
    parser = argparse.ArgumentParser(
        description=('create SPEXone L1A product from DEM measurement(s)'))
    parser.add_argument('--output', default=None,
                        help='define output directory, default=CWD')
    parser.add_argument('--dem_id', choices=('D35', 'D39'), default=None,
                        help=('provide DEM ID'
                              ' or ID will be extracted from path'))
    parser.add_argument('--reference', default=None,
                        help=('file with reference measurements'))
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('file_list', nargs='+',
                        help=("provide path to _b.bin file of one or more"
                              " SPEXone DEM characterization measurements"))
    args = parser.parse_args()
    if args.verbose:
        print(args)
    #
    buff = [name for name in args.file_list if name.endswith('.bin')]
    args.file_list = sorted(buff, key=os.path.getctime)

    # obtain DEM_ID (can be specified on command-line)
    # and list of measurements used to generate the L1A product
    list_dem_id = []
    list_name_msm = []
    for file in args.file_list:
        parts = Path(file).parts
        if len(parts) > 2:
            list_dem_id.append(parts[-2].split('_')[1])
        list_name_msm.append(parts[-1][:-6])

    if args.dem_id is None:
        if not list_dem_id:
            raise ValueError(
                'Can not determine DEM_ID, please specify --dem_id')
        if len(set(list_dem_id)) != 1:
            raise ValueError('DEM_ID of measurements must be unique')

        args.dem_id = list(set(list_dem_id))[0]

    # initialze data arrays
    msm_id = None
    tstamp = []
    image_list = []

    # Measurement Parameters Settings
    mps_data = np.zeros(len(args.file_list), dtype=MPSdef().dtype)
    texp = np.empty(len(args.file_list), dtype=float)
    temp = np.empty(len(args.file_list), dtype=float)

    for ii, flname in enumerate(args.file_list):
        dem_file = Path(flname)
        if not dem_file.is_file():
            raise FileNotFoundError('file {} does not exist'.format(flname))
        if dem_file.suffix == '.txt':
            continue

        # read DEM settings and data
        parts = dem_file.name.split('_')
        if len(parts) == 8:
            _id = '_'.join(parts[:4])
            tstamp.append(datetime.strptime(parts[5] + parts[6],
                                            '%Y%m%d%H%M%S.%f'))
        elif len(parts) == 10:
            _id = '_'.join(parts[:6])
            tstamp.append(datetime.strptime(parts[7] + parts[8],
                                            '%Y%m%d%H%M%S.%f'))
        elif len(parts) == 7:
            _id = '_'.join(parts[:3])
            tstamp.append(datetime.strptime(parts[4] + parts[5],
                                            '%Y%m%d%H%M%S.%f'))
        else:
            raise ValueError("Invalid format of data-product name")

        if msm_id is None:
            msm_id = _id
        elif msm_id != _id:
            raise ValueError("Do not combine different kind of measurements")

        dem = DEMio()
        # obtain MPS information from header file (ASCII)
        dem.read_hdr(flname.replace('b.bin', 'a.txt'))
        mps_data[ii] = MPSdef().fill(dem.hdr)
        # get nr_coaddings from file name
        coad_str = [x for x in parts if x.startswith('coad')][0]
        mps_data[ii]['REG_NCOADDFRAMES'] = int(coad_str[-2:])
        # determine exposure time
        texp[ii] = dem.exposure_time(mcp=1e-7)
        # determine detector temperature
        # temp[ii] = float(dem.temp_detector())
        temp[ii] = 293.0
        # read image data
        image_list.append(dem.read_data(flname))

    images = np.array(image_list)
    del image_list

    # sort data according to timestamps
    if tstamp != sorted(tstamp):
        indx = sorted(range(len(tstamp)), key=tstamp.__getitem__)
        mps_data = mps_data[indx]
        tstamp = [tstamp[k] for k in indx]
        texp = texp[indx]
        temp = temp[indx]
        images = images[indx]

    # convert timestamps to CCSDS time (UTC and fraction of a second)
    utc_sec = np.zeros(len(tstamp), dtype=np.uint32)
    frac_sec = np.zeros(len(tstamp), dtype=float)
    for ii, tval in enumerate(tstamp):
        secnd = (tval - datetime(1970, 1, 1)).total_seconds()
        # time-of-measurement provided at end of integration time
        secnd -= texp[ii] * mps_data[ii]['REG_NCOADDFRAMES']
        utc_sec[ii] = int(secnd)
        frac_sec[ii] = secnd % 1

    # generate name of L1A product
    prod_name = spx_product.prod_name(utc_sec[0], msm_id=msm_id.strip(' '))
    if args.output is not None:
        dest_dir = Path(args.output)
        if not dest_dir.is_dir():
            dest_dir.mkdir()
        prod_name = str(dest_dir / prod_name)

    # set dimensions of L1A datasets
    if images.ndim == 2:
        n_images = 1
        n_samples = images.size
    else:
        n_images = images.shape[0]
        n_samples = images.shape[1] * images.shape[2]

    dims = {'number_of_images': n_images,
            'samples_per_image': n_samples,
            'SC_records' : None,
            'tlm_packets' : None,
            'nv': 1}

    if args.reference is not None:
        with h5py.File(args.reference, 'r') as fid:
            secnd = fid['sec'][:]
            mask = ((secnd >= utc_sec[0]) & (secnd <= utc_sec[-1]))
            median, spread = biweight(fid['amps'][mask], spread=True)
        reference = {'value': median, 'error': spread}
    #
    # Generate L1A product
    #   ToDo: correct value for measurement_type & viewport
    with L1Aio(prod_name, dims, inflight=False) as l1a:
        l1a.set_attr('history', msm_id)
        l1a.set_attr('dem_id', args.dem_id)
        l1a.set_attr('measurements', list_name_msm)

        # Image data
        l1a.fill_images(images.reshape(n_images, n_samples))
        l1a.fill_mps(mps_data)

        # Image attributes
        l1a.fill_time(utc_sec, frac_sec)

        # Engineering data
        l1a.fill_hk_time(utc_sec, frac_sec)
        l1a.set_dset('/engineering_data/temp_detector', temp)
        l1a.set_dset('/engineering_data/temp_optics', temp)

        # GSE data
        if args.reference is not None:
            l1a.fill_gse(reference=reference)

        # l1a.set_dset('/image_attributes/image_ID', image_id)

        # ToDo: OGSE and EGSE parameters
        # Add OGSE and EGSE parameters
        # - Light source
        # - DoLP
        # - AoLP
        # - FOV_begin
        # - FOV_end
        # - Detector illumination
        # - Illumination_level
        # - viewport(s)
        # - Wavelength and signal of data stimulus


# --------------------------------------------------
if __name__ == '__main__':
    main()