"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Create a valid PACE SPEXone Level-1A product to estimate the size of a
science orbit product

Copyright (c) 2020 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from datetime import datetime, timezone

import numpy as np

from pyspex.lv1_io import L1Aio
from pyspex.lib.tmtc_def import tmtc_def

# - global parameters ------------------------------


# - local functions --------------------------------


# - main function ----------------------------------
def main():
    """
    Generate a SPEXone inflight Level-1A product
    """
    start_time = datetime(2023, 1, 15, 12, 34, 56, tzinfo=timezone.utc)
    midnight = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    duration = 2880  # seconds

    # define product: name, level, version, start of time-coverage
    prod_level = 'L1A'
    bin_size = '2.5km'
    prod_version = 'V1.0'
    l1a_flname = 'PACE_SPEX.{:s}.{:s}.{:s}.{:s}.nc'.format(
        start_time.strftime('%Y%m%dT%H%M%S'),
        prod_level, bin_size, prod_version)

    # define size of the dimensions
    dims = {
        'samples_per_image': 194029,
    }

    # define MPS data and image data
    mps_data = np.zeros(1, dtype=np.dtype(tmtc_def(0x350)))
    hk_data = np.zeros(1, dtype=np.dtype(tmtc_def(0x320)))
    image = np.arange(dims['samples_per_image'], dtype=np.uint16).reshape(1, -1)

    with L1Aio(l1a_flname, dims=dims, inflight=True) as l1a:
        secnds = (1 + np.arange(3 * duration, dtype=float)) / 3\
            + (start_time - midnight).total_seconds()
        l1a.fill_time(secnds, midnight)

        for ii in range(3 * duration):
            # Detector image data
            l1a.set_dset('/science_data/detector_images', image)

            # Detector telemetry
            # - alternative when real MPS are available - l1a.fill_mps(mps_data)
            l1a.set_dset('/science_data/detector_telemetry', mps_data)
            l1a.set_dset('/image_attributes/binning_table', [3])
            l1a.set_dset('/image_attributes/digital_offset', [0])
            l1a.set_dset('/image_attributes/nr_coadditions', [5])
            l1a.set_dset('/image_attributes/exposure_time', [0.056])
            l1a.set_dset('/image_attributes/image_ID', [ii])

        secnds = np.arange(duration, dtype=float) \
            + (start_time - midnight).total_seconds()
        l1a.set_dset('/engineering_data/HK_tlm_time', secnds)

        # Engineering data
        for _ in range(duration):
            l1a.set_dset('/engineering_data/HK_telemetry', hk_data)
            l1a.set_dset('/engineering_data/temp_detector', [293.])
            l1a.set_dset('/engineering_data/temp_optics', [294.])

        # Navigation data
        secnds = 2 * np.arange(duration // 2, dtype=float) \
            + (start_time - midnight).total_seconds()
        l1a.set_dset('/navigation_data/att_time', secnds)
        l1a.set_dset('/navigation_data/orb_time', secnds + .0132)

        for _ in range(max(1, duration // 2)):
            l1a.set_dset('/navigation_data/adstate', [255])
            l1a.set_dset('/navigation_data/att_quat',
                         [[0.0729222, -0.016155, 0.1108807, 0.9910231]])
            l1a.set_dset('/navigation_data/orb_pos',
                         [[404721.7, 1117799.9, -6963719.7]])
            l1a.set_dset('/navigation_data/orb_vel',
                         [[7295.4116, 1605.6194, 682.1598]])

        # Global attributes
        l1a.fill_global_attrs(orbit=12345, bin_size=bin_size)

    # with L1Aio(l1a_flname, append=True) as l1a:
    #    print(l1a.ref_date)


# --------------------------------------------------
if __name__ == '__main__':
    main()