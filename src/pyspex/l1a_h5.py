"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Access SPEXone L1A product using h5py (faster, cleaner)

Copyright (c) 2019 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path

import h5py
import numpy as np

# - global parameters -------------------


# - local functions ---------------------


# - class L1Aread -----------------------
class L1Aread():
    """
    Access existing SPEXone L1A product
    """
    def __init__(self, l1a_product) -> None:
        """
        Initialize access to a SPEXone L1a product
        """
        # initialize private class-attributes
        if isinstance(l1a_product, str):
            self.l1a_path = Path(l1a_product)
        else:
            self.l1a_path = l1a_product

        if not self.l1a_path.is_file():
            raise FileNotFoundError("file not found on system")

        self.fid = h5py.File(self.l1a_path, 'r')
        self.dem_id = self.fid.attrs['dem_id'].decode('ascii')

    def __enter__(self):
        """
        method called to initiate the context manager
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """
        method called when exiting the context manager
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self) -> None:
        """
        Close access to L1A product
        """
        if self.fid is None:
            return

        self.fid.close()
        self.fid = None

    def dtime(self):
        """
        Returns image time as (decimal) seconds in day
        """
        return self.fid['/image_attributes/image_time'][:]

    def exposure_time(self):
        """
        Returns image exposure-time
        """
        return self.fid['/image_attributes/exposure_time'][:]

    def nr_coadditions(self):
        """
        Return number of coadditions
        """
        return self.fid['/image_attributes/nr_coadditions'][:]

    def offset(self) -> int:
        """
        Returns digital offset including ADC offset [counts]
        """
        uval = np.unique(
            self.fid['/science_data/detector_telemetry']['DET_OFFSET'][:])[0]

        return 70 + (uval if uval < 8192 else uval - 16384)

    def pga_gain(self):
        """
        Returns PGA gain setting
        """
        mps = self.fid['/science_data/detector_telemetry'][0]

        reg_pgagain = mps['DET_PGAGAIN']
        # need first bit of address 121
        reg_pgagainfactor = mps['DET_BLACKCOL'] & 0x1

        return (1 + 0.2 * reg_pgagain) * 2 ** reg_pgagainfactor

    def adc_gain(self):
        """
        Returns ADC gain setting
        """
        return self.fid['/science_data/detector_telemetry']['DET_ADCGAIN'][0]

    def images(self):
        """
        Return detector images
        """
        return self.fid['/science_data/detector_images'][:]

    def signal(self):
        """
        Returns detector-image signal per exposure time

        The signal is corrected for coaddition and digital offset
        """
        dtime_all = self.dtime()
        coad_all = self.nr_coadditions()
        texp_all = self.exposure_time()
        images = self.images()

        texp = np.unique(texp_all)
        signal = np.zeros((len(texp), images.shape[1]), dtype=float)

        # sort measurements according to exposure times
        # and correct signal for co-additions
        for ii, tval in enumerate(texp):
            indx = np.where(texp_all == tval)[0]
            ij = np.argsort(dtime_all[indx])
            buff = images[indx[ij[2:-1]], :].astype(float)
            signal[ii, ...] = (np.sum(buff, axis=0)
                               / np.sum(coad_all[indx[ij[2:-1]]]))

        return (texp, signal - self.offset())


if __name__ == '__main__':
    DATA_DIR = '/nfs/SPEXone/DEM/L1A/D35'
    DEM_FILE = 'SPX_OCAL_lin_tsat00080_gain57x14_L1A_20191108T135826_20191216T141727_0001.nc'
    with L1Aread(Path(DATA_DIR, DEM_FILE)) as l1a:
        print('dtime ', l1a.dtime().shape)
        print('texp ', l1a.exposure_time().shape)
        print('coaddf ', l1a.nr_coadditions().shape)
        print('offset ', l1a.offset())
        print('pga_gain ', l1a.pga_gain())
        print('adc_gain ', l1a.adc_gain())
        print('images ', l1a.images().shape)
        print('signal ', l1a.signal()[1].shape)

    print('Success')
