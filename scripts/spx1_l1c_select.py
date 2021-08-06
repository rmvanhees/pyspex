"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Create new SPEXone Level-1C product with selected data from original

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from pathlib import Path

from pyspex.lv1_io import L1Cio

# --------------------------------------------------
def main():
    """
    Main function of this module
    """
    parser = argparse.ArgumentParser(
        description='create Quick-Look from SPEXone L1C product')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('--time', nargs=2, default=None,
                        help='select on image time [start, end]')
    parser.add_argument('--mps_id', default=None,
                        help='select on MPS-ID [comma separated?]')
    # parser.add_argument('--', default=None, help='')
    # parser.add_argument('--', default=None, help='')
    parser.add_argument('--out_dir', default='.',
                        help=('name of directory to store the new Level-1C'
                              ' product, default: current working directory'))
    parser.add_argument('l1c_product', default=None,
                        help='name of SPEXone Level-1C product')
    args = parser.parse_args()
    if args.verbose:
        print(args)

    l1c_product = Path(args.l1c_product)
    if not l1c_product.is_file():
        raise FileNotFoundError(
            'File {} does not exist'.format(args.l1c_product))
    # Check if SPEXone Level-1C product
    # ToDo: implement check on data product

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(mode=0o755, parents=True)

    # ----- read data from orignal product -----
    # ToDo: implement read of data
    dims = []
    nomhk_tm = []
    demhk_tm = []
    with L1Cio(l1c_product) as l1c:
        # write image data, detector telemetry and image attributes
        # - datasets: img_data, img_hk, img_id, img_sec, img_subsec

        # write engineering data
        # - datasets: nomhk_data, nomhk_sec, nomhk_subsec
        # - datasets: demhk_data

        # write global attributes
        # - parameters: inflight, selection criteria

    # ----- perform data selection -----
    # ToDo: implement data selection

    # ----- now we can update the name of the output product -----
    # - because the production time has changed
    # - and when coverage time is changed
    if ((out_dir / l1c_product.name).is_file()
        l1c_product.samefile(out_dir / l1c_product.name)):
        raise OSError('Output will overwrite original product')
        
    # ----- write new output product with selected data -----
    with L1Cio(out_dir / l1c_product.name, dims=dims) as l1c:
        # group BIN_ATTRIBUTES
        l1c.set_dset('/BIN_ATTRIBUTES/nadir_view_time', data)
        l1c.set_dset('/BIN_ATTRIBUTES/view_time_offsets', data)
        
        # group GEOLOCATION_DATA
        l1c.set_dset('/GEOLOCATION_DATA/altitude', data)
        l1c.set_dset('/GEOLOCATION_DATA/altitude_variability', data)
        l1c.set_dset('/GEOLOCATION_DATA/latitude', data)
        l1c.set_dset('/GEOLOCATION_DATA/longitude', data)
        l1c.set_dset('/GEOLOCATION_DATA/sensor_azimuth', data)
        l1c.set_dset('/GEOLOCATION_DATA/sensor_zenith', data)
        l1c.set_dset('/GEOLOCATION_DATA/solar_azimuth', data)
        l1c.set_dset('/GEOLOCATION_DATA/solar_zenith', data)

        # group OBSERVATION_DATA
        l1c.set_dset('/OBSERVATION_DATA/obs_per_view', data)
        l1c.set_dset('/OBSERVATION_DATA/QC_bitwise', data)
        l1c.set_dset('/OBSERVATION_DATA/QC', data)
        l1c.set_dset('/OBSERVATION_DATA/QC_polsample_bitwise', data)
        l1c.set_dset('/OBSERVATION_DATA/QC_polsample', data)
        l1c.set_dset('/OBSERVATION_DATA/I', data)
        l1c.set_dset('/OBSERVATION_DATA/I_noise', data)
        l1c.set_dset('/OBSERVATION_DATA/I_polsample', data)
        l1c.set_dset('/OBSERVATION_DATA/I_polsample_noise', data)
        l1c.set_dset('/OBSERVATION_DATA/DoLP', data)
        l1c.set_dset('/OBSERVATION_DATA/DoLP_noise', data)
        l1c.set_dset('/OBSERVATION_DATA/Q_over_I', data)
        l1c.set_dset('/OBSERVATION_DATA/Q_over_I_noise', data)
        l1c.set_dset('/OBSERVATION_DATA/AoLP', data)
        l1c.set_dset('/OBSERVATION_DATA/AoLP_noise', data)
        l1c.set_dset('/OBSERVATION_DATA/U_over_I', data)
        l1c.set_dset('/OBSERVATION_DATA/U_over_I_noise', data)

        # group SENSOR_VIEWS_BANDS
        l1c.set_dset('/SENSOR_VIEWS_BANDS/intensity_bandpasses', data)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/intensity_F0', data)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/intensity_wavelengths', data)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/polarization_bandpasses', data)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/polarization_F0', data)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/polarization_wavelengths', data)
        l1c.set_dset('/SENSOR_VIEWS_BANDS/view_angles', data)

        # write global attributes
        l1c.fill_global_attrs(inflight=inflight)
        # l1c.set_attr('input_files', [Path(x).name for x in args.file_list])

# --------------------------------------------------
if __name__ == '__main__':
    main()
