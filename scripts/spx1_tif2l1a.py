#!/usr/bin/env python3
#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Python implementation SPEXone instrument simulator output to L1A.

Environment::

   CKD_DIR:  directory with SPEXone CKD, default is CWD.

"""
import argparse
from datetime import datetime, timezone
from os import environ
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from pyspex import spx_product
from pyspex.lib.tmtc_def import tmtc_dtype
from pyspex.tif_io import TIFio
from pyspex.lv1_io import L1Aio
from pyspex.lv1_gse import LV1gse

# - global parameters ------------------------------
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


# - local functions --------------------------------
def get_table_id(ckd_dir, bin_table_name):
    """
    Quick en dirty implementation to obtain table_id from binning-table CKD
    """
    if not Path(ckd_dir).is_dir():
        ckd_dir = environ.get('CKD_DIR', '.')

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


def get_stimulus(hdr):
    """
    Return stimulus data as a xarray::Dataset
    """
    if 'Spectral data stimulus' not in hdr:
        return None

    ds_dict = hdr['Spectral data stimulus']
    xr_wv = xr.DataArray(ds_dict['Wavelength (nm)'],
                         coords=[ds_dict['Wavelength (nm)']],
                         dims=['wavelength'],
                         attrs={'long_name': 'wavelength of stimulus',
                                'units': 'nm'})

    data = ds_dict['Radiance (photons/(s.nm.m^2.sr)']
    data *= 2.99792458 * 6.62607015e-14 / ds_dict['Wavelength (nm)']
    xr_sign = xr.DataArray(data, dims=['wavelength'],
                           coords=[ds_dict['Wavelength (nm)']],
                           attrs={'long_name': 'signal of stimulus',
                                  'units': 'W.m-2.sr-1.um-1'})

    return xr.Dataset({'wavelength': xr_wv, 'signal': xr_sign})


# - main function ----------------------------------
def main():
    """
    main program to illustate the creation of a L1A calibration product

    Environment
    -----------
    CKD_DIR :  directory with SPEXone CKD, default is CWD
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
        raise FileNotFoundError(f'File {args.ascii_file} does not exist')

    if not args.inp_tif and args.nframe is not None:
        raise RuntimeError('option --nframe works only with "--inp_tif"')

    tif = TIFio(args.ascii_file, inp_tif=args.inp_tif, lineskip=args.lineskip)
    hdr = tif.header()
    tags = tif.tags()[0]
    images = tif.images(n_frame=args.nframe)
    if args.verbose:
        for key, value in hdr.items():
            if key == 'Spectral data stimulus':
                continue
            print(f"{key:s}: '{value}'")

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
    intg_time = int(hdr['Co-additions']) * float(hdr['Exposure time (s)'])
    secnds = (1 + np.arange(n_images, dtype=float)) * intg_time
    img_sec = (offs + secnds // 1).astype('u4')
    img_subsec = np.round((secnds % 1) * 2**16).astype('u2')
    img_time = secnds

    nom_hk = np.zeros(n_images, dtype=tmtc_dtype(0x320))
    sci_hk = np.zeros(n_images, dtype=tmtc_dtype(0x350))
    if table_id == 0:
        sci_hk['REG_FULL_FRAME'] = 1
        sci_hk['REG_CMV_OUTPUTMODE'] = 3
    else:
        sci_hk['REG_FULL_FRAME'] = 2
        sci_hk['REG_CMV_OUTPUTMODE'] = 1
    sci_hk['REG_BINNING_TABLE_START'] = 0x400000 * (table_id + 1) + 0x80000000
    sci_hk['DET_OFFSET'] = 16384 - 70
    sci_hk['DET_FOTLEN'] = 20
    sci_hk['DET_EXPTIME'] = \
        int(1e7 * float(hdr['Exposure time (s)']) / 129 - 0.43 * 20)
    sci_hk['REG_NCOADDFRAMES'] = int(hdr['Co-additions'])

    # Generate L1A product
    with L1Aio(prod_name, dims=dims, ref_date=utc_start.date()) as l1a:
        # write image data, detector telemetry and image attributes
        l1a.fill_science(images.reshape(n_images, -1), sci_hk,
                         np.arange(n_images))
        l1a.set_dset('/image_attributes/icu_time_sec', img_sec)
        l1a.set_dset('/image_attributes/icu_time_subsec', img_subsec)
        l1a.set_dset('/image_attributes/image_time', img_time)

        # Engineering data
        l1a.fill_nomhk(nom_hk)
        l1a.set_dset('/engineering_data/HK_tlm_time', img_time)

        # Global attributes
        l1a.fill_global_attrs(inflight=False)
        l1a.set_attr('input_files', [Path(args.ascii_file).name])

    with LV1gse(prod_name) as gse:
        gse.set_attr('origin', hdr['history'])
        gse.set_attr('comment', hdr['OCAL measurement'])
        if 'Light source' in hdr:
            gse.set_attr('light_source', hdr['Light source'])
        if 'Detector illumination' in hdr:
            gse.set_attr('Light_source', hdr['Detector illumination'])
            gse.set_attr('Illumination_level',
                         float(hdr['Detect. illumination e/ms']))
        if 'Detector temperature (K)' in hdr:
            gse.set_attr('Detector_temperature',
                         float(hdr['Detector temperature (K)']))
        if 'Detector temperature (K)' in hdr:
            gse.set_attr('Optics_temperature',
                         float(hdr['Optics temperature (K)']))
        if 'Illuminated viewing angle' in hdr:
            vp_dict = {'0': 1, '1': 2, '2': 16, '3': 8, '4': 4}
            gse.write_viewport(vp_dict.get(hdr['Illuminated viewing angle']))
        else:
            gse.write_viewport(0)
        if 'Illuminated field ACT' in hdr:
            gse.write_attr_act(float(hdr['Rotation stage ACT angle']),
                               float(hdr['Illuminated field ACT']))
        else:
            gse.write_attr_act(float(hdr['Rotation stage ACT angle']))
        if 'Illuminated field ALT' in hdr:
            gse.write_attr_alt(float(hdr['Rotation stage ALT angle']),
                               float(hdr['Illuminated field ALT']))
        else:
            gse.write_attr_alt(float(hdr['Rotation stage ALT angle']))
        if 'DoLP input' in hdr:
            gse.write_attr_polarization(float(hdr['AoLP input (deg)']),
                                        float(hdr['DoLP input']))
        if 'Spectral data stimulus' in hdr \
           and 'Illuminated viewing angle' in hdr:
            gse.write_data_stimulus(get_stimulus(hdr))


# --------------------------------------------------
if __name__ == '__main__':
    main()
