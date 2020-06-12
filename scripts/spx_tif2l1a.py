"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python implementation SPEXone instrument simulator output to L1A

Copyright (c) 2019 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from datetime import datetime
from pathlib import Path

import numpy as np

from pyspex import spx_product
from pyspex.tif_io import TIFio
from pyspex.l1a_io import L1Aio

# - global parameters ------------------------------


# - local functions --------------------------------
def header_as_dict(hdr, n_images):
    """
    Convert header data to Python dictionary
    """
    print(hdr)
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
    hdr_dict['Line-enable-array'] = {
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
        'ds_name': '/engineering_data/temp_optics',
        'ds_value': np.full(n_images,
                            float(hdr['Optics temperature (K)']))}
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


# - main function ----------------------------------
def main():
    """
    main program to illustate the creation of a L1A calibration product
    """
    parser = argparse.ArgumentParser(
        description=('create SPEXone L1A product from instrument simulations'))
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

    tif = TIFio(args.ascii_file, inp_tif=args.inp_tif)
    hdr = tif.header()
    tags = tif.tags()[0]
    images = tif.images(n_frame=args.nframe)

    # For now, we obtain the reference time from the TIFF.tags
    msm_id = hdr['OCAL measurement']
    for key in tags:
        if key.name == 'datetime':
            str_date, str_time = tags[key.value].split(' ')
            ref_time = str_date.replace(':', '-') + 'T' + str_time
            ref_sec = int((datetime.fromisoformat(ref_time)
                           - datetime(1970, 1, 1)).total_seconds())

    prod_name = spx_product.prod_name(ref_sec, msm_id=msm_id)
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
        n_samples = (int(hdr['Row dimension'])
                     * int(hdr['Column dimension']))

    dims = {'number_of_images': n_images,
            'samples_per_image': n_samples,
            'hk_packets': n_images,
            'viewing_angles': int(hdr['Number of viewing angles'])}
    if 'Spectral data stimulus' in hdr:
        ds_dict = hdr['Spectral data stimulus']
        dims['wavelength'] = ds_dict['Wavelength (nm)'].size

    # Compute delta_time for each frame (seconds)
    utc_sec = []
    frac_sec = []
    for ii in range(n_images):
        seconds = (ii * int(hdr['Co-additions'])
                   * float(hdr['Exposure time (s)']))
        utc_sec.append(ref_sec + int(seconds))
        frac_sec.append(seconds % 1)

    # Generate L1A product
    hdr_dict = header_as_dict(hdr, n_images)
    with L1Aio(prod_name, dims, inflight=False) as l1a:
        l1a.set_attr('history', hdr['OCAL measurement'])
        #
        # Add image data and telemetry
        l1a.fill_images(images.reshape(n_images, n_samples))
        # -- no detector_telemetry!!
        #
        # Add image attributes
        # - use default values for binning_table, digital_offset, image_ID
        l1a.fill_time(np.array(utc_sec), np.array(frac_sec))
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
                         np.zeros(n_images))
        #
        # Add engineering data
        l1a.fill_hk_time(np.array(utc_sec), np.array(frac_sec))
        for key in hdr_dict:
            if hdr_dict[key]['ds_type'] != 'dset':
                continue
            l1a.set_dset(hdr_dict[key]['ds_name'], hdr_dict[key]['ds_value'])
        #
        # Add OGSE and EGSE parameters
        for key in hdr_dict:
            if hdr_dict[key]['ds_type'] != 'attr':
                continue
            print(hdr_dict[key])
            l1a.set_attr(hdr_dict[key]['ds_name'],
                         hdr_dict[key]['ds_value'],
                         ds_name='gse_data')


# --------------------------------------------------
if __name__ == '__main__':
    main()
