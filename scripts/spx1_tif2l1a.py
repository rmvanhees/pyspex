#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation SPEXone instrument simulator output to L1A

Copyright (c) 2019-2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from pyspex import spx_product
from pyspex.lib.tmtc_def import tmtc_def
from pyspex.tif_io import TIFio
from pyspex.lv1_io import L1Aio

# - global parameters ------------------------------
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# - local functions --------------------------------
def header_as_dict(hdr, n_images):
    """
    Convert header data to Python dictionary
    """
    hdr_dict = {}
    hdr_dict['history'] = {
        'ds_type': None, 'ds_name': None, 'ds_value': None}
    hdr_dict['OCAL measurement'] = {
        'ds_type': None, 'ds_name': None, 'ds_value': None}
    hdr_dict['Row dimension'] = {
        'ds_type': None, 'ds_name': None, 'ds_value': None}
    hdr_dict['Column dimension'] = {
        'ds_type': None, 'ds_name': None, 'ds_value': None}
    hdr_dict['Number of measurements'] = {
        'ds_type': None, 'ds_name': None, 'ds_value': None}
    hdr_dict['Number of viewing angles'] = {
        'ds_type': None, 'ds_name': None, 'ds_value': None}
    hdr_dict['Integration time (s)'] = {
        'ds_type': None, 'ds_name': None, 'ds_value': None}
    hdr_dict['Exposure time (s)'] = {
        'ds_type': None, 'ds_name': None, 'ds_value': None}
    hdr_dict['Co-additions'] = {
        'ds_type': None, 'ds_name': None, 'ds_value': None}

    # keywords for binning (write as attribute
    hdr_dict['Line_skip_id'] = {
        'ds_type': 'attr',
        'ds_name': 'Line_skip_id',
        'ds_value': ''}
    if 'Line-enable-array' in hdr and hdr['Line-enable-array'] != '':
        hdr_dict['Line_skip_id']['ds_value'] = hdr['Line-enable-array']

    hdr_dict['Enabled_lines'] = {
        'ds_type': 'attr',
        'ds_name': 'Enabled_lines',
        'ds_value': np.uint16(2048)}
    if 'Enabled lines' in hdr:
        hdr_dict['Enabled_lines']['ds_value'] = np.uint16(hdr['Enabled lines'])

    hdr_dict['Binning_table'] = {
        'ds_type': 'attr',
        'ds_name': 'Binning_table',
        'ds_value': ''}
    if 'Flexible binning table' in hdr and hdr['Flexible binning table'] != '':
        hdr_dict['Binning_table']['ds_value'] = hdr['Flexible binning table']

    hdr_dict['Binned_pixels'] = {
        'ds_type': 'attr',
        'ds_name': 'Binned_pixels',
        'ds_value': np.uint32(0)}
    if 'Total flex-binned pixels' in hdr:
        hdr_dict['Binned_pixels']['ds_value'] = \
            np.uint32(hdr['Total flex-binned pixels'])

    # Datasets
    hdr_dict['Optics temperature (K)'] = {
        'ds_type': 'dset',
        'ds_name': '/engineering_data/temp_housing',
        'ds_value': np.full(n_images,
                            float(hdr['Housing temperature (K)']))}
    hdr_dict['Detector temperature (K)'] = {
        'ds_type': 'dset',
        'ds_name': '/engineering_data/temp_detector',
        'ds_value': np.full(n_images,
                            float(hdr['Detector temperature (K)']))}
    if hdr['Illuminated viewing angle'] == 'None':
        hdr_dict['Illuminated viewing angle'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        hdr_dict['Illuminated viewing angle'] = {
            'ds_type': 'dset',
            'ds_name': '/gse_data/viewport',
            'ds_value': np.uint8(1 << int(hdr['Illuminated viewing angle']))}

    # Attributes
    if hdr['Light source'] == 'None':
        hdr_dict['Light source'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        hdr_dict['Light source'] = {
            'ds_type': 'attr',
            'ds_name': 'Light_source',
            'ds_value': hdr['Light source'].lstrip()}
    if hdr['Illuminated fov begin'] == 'None':
        hdr_dict['Illuminated fov begin'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        hdr_dict['Illuminated fov begin'] = {
            'ds_type': 'attr',
            'ds_name': 'FOV_begin',
            'ds_value': float(hdr['Illuminated fov begin'])}
    if hdr['Illuminated fov end'] == 'None':
        hdr_dict['Illuminated fov end'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        hdr_dict['Illuminated fov end'] = {
            'ds_type': 'attr',
            'ds_name': 'FOV_end',
            'ds_value': float(hdr['Illuminated fov end'])}
    if hdr['Rotation stage ACT angle'] == 'None':
        hdr_dict['Rotation stage ACT angle'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        hdr_dict['Rotation stage ACT angle'] = {
            'ds_type': 'attr',
            'ds_name': 'ACT_rotationAngle',
            'ds_value': float(hdr['Rotation stage ACT angle'])}
    if hdr['Rotation stage ALT angle'] == 'None':
        hdr_dict['Rotation stage ALT angle'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        hdr_dict['Rotation stage ALT angle'] = {
            'ds_type': 'attr',
            'ds_name': 'ALT_rotationAngle',
            'ds_value': float(hdr['Rotation stage ALT angle'])}
    if hdr['Illuminated field ACT'] == 'None':
        hdr_dict['Illuminated field ACT'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        hdr_dict['Illuminated field ACT'] = {
            'ds_type': 'attr',
            'ds_name': 'ACT_illumination',
            'ds_value': float(hdr['Illuminated field ACT'])}
    if hdr['Illuminated field ALT'] == 'None':
        hdr_dict['Illuminated field ALT'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        hdr_dict['Illuminated field ALT'] = {
            'ds_type': 'attr',
            'ds_name': 'ALT_illumination',
            'ds_value': float(hdr['Illuminated field ALT'])}
    if hdr['DoLP input'] == 'None':
        hdr_dict['DoLP input'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        hdr_dict['DoLP input'] = {
            'ds_type': 'attr',
            'ds_name': 'DoLP',
            'ds_value': float(hdr['DoLP input'])}
    if hdr['AoLP input (deg)'] == 'None':
        hdr_dict['AoLP input (deg)'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        hdr_dict['AoLP input (deg)'] = {
            'ds_type': 'attr',
            'ds_name': 'AoLP',
            'ds_value': float(hdr['AoLP input (deg)'])}

    if hdr['Detector illumination'] == 'None':
        hdr_dict['Detector illumination'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
        hdr_dict['Det. illumination level'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}
    else:
        if 'Det. illumination level' in hdr:
            hdr_dict['Det. illumination level'] = {
                'ds_type': 'attr',
                'ds_name': 'Illumination_level',
                'ds_value': float(hdr['Det. illumination level'])}
            hdr_dict['Detector illumination'] = {
                'ds_type': 'attr',
                'ds_name': 'Light_source',
                'ds_value': hdr['Detector illumination']}
        if 'Detect. illumination e/ms' in hdr:
            hdr_dict['Det. illumination level'] = {
                'ds_type': 'attr',
                'ds_name': 'Illumination_level',
                'ds_value': float(hdr['Detect. illumination e/ms'])}
            hdr_dict['Detector illumination'] = {
                'ds_type': 'attr',
                'ds_name': 'Light_source',
                'ds_value': hdr['Detector illumination'] + ' (e.ms-1)'}

    if 'Spectral data stimulus' in hdr:
        ds_dict = hdr['Spectral data stimulus']
        hdr_dict['Wavelength stimulus'] = {
            'ds_type': 'dset',
            'ds_name': '/gse_data/wavelength',
            'ds_value': ds_dict['Wavelength (nm)']}
        hdr_dict['Radius stimulus'] = {
            'ds_type': 'dset',
            'ds_name': '/gse_data/signal',
            'ds_value': ds_dict['Radiance (photons/(s.nm.m^2.sr)']}
    else:
        hdr_dict['Spectral data stimulus'] = {
            'ds_type': None, 'ds_name': None, 'ds_value': None}

    return hdr_dict


def get_table_id(ckd_dir, bin_table_name):
    """
    Quick en dirty implementation to obtain table_id from binning-table CKD
    """
    if not Path(ckd_dir).is_dir():
        ckd_dir = '/data/richardh/SPEXone/share/ckd'

    # only read the latest version of the binning-table CKD
    bin_tables = sorted(list(Path(ckd_dir).glob('SPX1_OCAL_L1A_TBL_*.nc')))
    if not bin_tables:
        raise FileNotFoundError('No binning table found')

    table_id = None
    with h5py.File(bin_tables[-1], 'r') as fid:
        for grp in [x for x in fid.keys() if x.startswith('Table')]:
            gid = fid[grp]
            if 'origin' not in gid.attrs:
                continue

            if gid.attrs['origin'].decode('ascii') == bin_table_name:
                table_id = int(grp.split('_')[1])
                break

    return table_id


# - main function ----------------------------------
def main():
    """
    main program to illustate the creation of a L1A calibration product
    """
    parser = argparse.ArgumentParser(
        description=('create SPEXone L1A product from instrument simulations'))
    parser.add_argument('--lineskip', default=False, action='store_true',
                        help='read line-skip data, when available')
    parser.add_argument('--output', default=None,
                        help='define output directory, default=CWD')
    inp_grp = parser.add_argument_group('inp_grp', 'process inp_tif files')
    inp_grp.add_argument('--inp_tif', default=False, action='store_true',
                         help='use TIFF file with 32-bit image')
    inp_grp.add_argument('--nframe', default=None, type=int,
                         help='specify the number of images to be generated')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='be verbose')
    parser.add_argument('ascii_file', default=None,
                        help=('provide path to SPEXone instrument simulator'
                              ' product with extension ".dat"'))
    args = parser.parse_args()
    if args.verbose:
        print(args)

    if not Path(args.ascii_file).is_file():
        raise FileNotFoundError(
            'File {} does not exist'.format(args.ascii_file))

    if not args.inp_tif and args.nframe is not None:
        raise RuntimeError('option --nframe works only with "--inp_tif"')

    tif = TIFio(args.ascii_file, inp_tif=args.inp_tif, lineskip=args.lineskip)
    hdr = tif.header()
    tags = tif.tags()[0]
    images = tif.images(n_frame=args.nframe)

    # For now, we obtain the reference time from the TIFF.tags
    msm_id = hdr['OCAL measurement']
    for key in tags:
        if key.name == 'datetime':
            str_date, str_time = tags[key.value].split(' ')
            utc_start = datetime.fromisoformat(
                str_date.replace(':', '-') + 'T' + str_time + '+00:00')
            break

    prod_name = spx_product.prod_name(utc_start, msm_id=msm_id)
    if args.inp_tif:
        prod_name = prod_name.replace('_L1A_', '_inp_L1A_')
    if args.output is not None:
        dest_dir = Path(args.output)
        if not dest_dir.is_dir():
            dest_dir.mkdir()
        prod_name = str(dest_dir / prod_name)
    if args.verbose:
        print(prod_name)

    # define dimensions for L1A product
    if args.inp_tif:
        if images.ndim == 2:
            n_images = 1
            n_samples = images.size
        else:
            n_images = images.shape[0]
            n_samples = images.shape[1] * images.shape[2]
    else:
        n_images = int(hdr['Number of measurements'])
        if args.lineskip:
            n_samples = (int(hdr['Enabled lines'])
                         * int(hdr['Column dimension']))
        else:
            n_samples = (int(hdr['Row dimension'])
                         * int(hdr['Column dimension']))

    dims = {'number_of_images': n_images,
            'samples_per_image': n_samples,
            'hk_packets': n_images,
            'viewing_angles': int(hdr['Number of viewing angles'])}
    if 'Spectral data stimulus' in hdr:
        ds_dict = hdr['Spectral data stimulus']
        dims['wavelength'] = ds_dict['Wavelength (nm)'].size

    # define binning table ID
    if n_samples == 0x400000:
        table_id = 0
    elif n_samples == 0x100000:
        table_id = 1
    else:
        if args.lineskip:
            table_id = get_table_id('/nfs/SPEXone/share/ckd',
                                    hdr['Line-enable-array'])
        else:
            table_id = get_table_id('/nfs/SPEXone/share/ckd',
                                    hdr['Flexible binning table'])

    # Compute delta_time for each frame (seconds)
    # convert timestamps to seconds per day
    offs = (utc_start - EPOCH).total_seconds()
    intg_time = hdr['Co-additions'] * float(hdr['Exposure time (s)'])
    secnds = (1 + np.arange(n_images, dtype=float)) * intg_time
    img_sec = (offs + secnds // 1).astype('u4')
    sub_sec = np.round((secnds % 1) * 2**16).astype('u2')

    # Generate L1A product
    hdr_dict = header_as_dict(hdr, n_images)
    with L1Aio(prod_name, dims=dims, inflight=False) as l1a:
        # Add image data and telemetry
        l1a.set_dset('/science_data/detector_images',
                     images.reshape(n_images, -1))
        img_hk = np.zeros(n_images, dtype=np.dtype(tmtc_def(0x350)))
        l1a.set_dset('/science_data/detector_telemetry', img_hk)
        # -- no detector_telemetry!!
        #
        # Add image attributes
        # - use default values for binning_table, digital_offset, image_ID
        l1a.set_dset('/image_attributes/exposure_time',
                     np.full(n_images, float(hdr['Exposure time (s)'])))
        l1a.set_dset('/image_attributes/nr_coadditions',
                     np.full(n_images, int(hdr['Co-additions'])))

        l1a.set_dset('/image_attributes/image_ID',
                     np.arange(n_images))
        l1a.set_dset('/image_attributes/digital_offset',
                     np.zeros(n_images))
        if args.inp_tif:
            l1a.set_dset('/image_attributes/binning_table',
                         np.full(n_images, 101))
        else:
            l1a.set_dset('/image_attributes/binning_table',
                         np.full(n_images, table_id))
        l1a.fill_time(img_sec, sub_sec)
        #
        # Add engineering data
        hk_data = np.zeros(n_images, dtype=np.dtype(tmtc_def(0x320)))
        l1a.set_dset('/engineering_data/HK_telemetry', hk_data)
        l1a.set_dset('/engineering_data/HK_tlm_time', secnds)
        for key in hdr_dict:
            if hdr_dict[key]['ds_type'] != 'dset':
                continue
            l1a.set_dset(hdr_dict[key]['ds_name'], hdr_dict[key]['ds_value'])
        #
        # Add OGSE and EGSE parameters
        for key in hdr_dict:
            if hdr_dict[key]['ds_type'] != 'attr':
                continue

            l1a.set_attr(hdr_dict[key]['ds_name'],
                         hdr_dict[key]['ds_value'],
                         ds_name='gse_data')
        # Global attributes
        l1a.fill_global_attrs()
        l1a.set_attr('history', hdr['OCAL measurement'])


# --------------------------------------------------
if __name__ == '__main__':
    main()
