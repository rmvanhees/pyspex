"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Create new SPEXone Level-1B product with selected data from original

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse
from pathlib import Path

from pyspex.lv1_io import L1Bio

# --------------------------------------------------
def main():
    """
    Main function of this module
    """
    parser = argparse.ArgumentParser(
        description='create Quick-Look from SPEXone L1B product')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='be verbose, default be silent')
    parser.add_argument('--time', nargs=2, default=None,
                        help='select on image time [start, end]')
    parser.add_argument('--mps_id', default=None,
                        help='select on MPS-ID [comma separated?]')
    # parser.add_argument('--', default=None, help='')
    # parser.add_argument('--', default=None, help='')
    parser.add_argument('--out_dir', default='.',
                        help=('name of directory to store the new Level-1B'
                              ' product, default: current working directory'))
    parser.add_argument('l1b_product', default=None,
                        help='name of SPEXone Level-1B product')
    args = parser.parse_args()
    if args.verbose:
        print(args)

    l1b_product = Path(args.l1b_product)
    if not l1b_product.is_file():
        raise FileNotFoundError(
            'File {} does not exist'.format(args.l1b_product))
    # Check if SPEXone Level-1B product
    # ToDo: implement check on data product

    out_dir = Path(args.out_dir)
    if not out_dir.is_dir():
        out_dir.mkdir(mode=0o755, parents=True)

    # ----- read data from orignal product -----
    # ToDo: implement read of data
    dims = []
    nomhk_tm = []
    demhk_tm = []
    with L1Bio(l1b_product) as l1b:
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
    if ((out_dir / l1b_product.name).is_file()
        and l1b_product.samefile(out_dir / l1b_product.name)):
        raise OSError('Output will overwrite original product')
        
    # ----- write new output product with selected data -----
    with L1Bio(out_dir / l1b_product.name, dims=dims) as l1b:
        # write image data, detector telemetry and image attributes
        l1b.fill_science(img_data, img_hk, img_id)
        l1b.fill_time(img_sec, img_subsec, group='image_attributes')

        # write engineering data
        if nomhk_tm:
            l1b.fill_nomhk(nomhk_data)
            l1b.fill_time(nomhk_sec, nomhk_subsec, group='engineering_data')

        if demhk_tm:
            l1b.fill_demhk(demhk_data)

        # write global attributes
        l1b.fill_global_attrs(inflight=inflight)
        # l1b.set_attr('input_files', [Path(x).name for x in args.file_list])

# --------------------------------------------------
if __name__ == '__main__':
    main()
