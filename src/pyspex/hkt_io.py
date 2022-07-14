"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python class to read PACE HKT files (S-band house-keeping and navigation)

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

from moniplot.image_to_xarray import h5_to_xr
from pyspex.lib.tmtc_def import tmtc_dtype

# - global parameters -------------------


# - class HKTio -------------------------
class HKTio:
    """
    Generic class to read PACE HKT products

    Attributes
    ----------
    ...

    Methods
    -------
    ...

    Notes
    -----
    ...

    Examples
    --------
    ...

    """
    def __init__(self, filename: str, instrument='spx') -> None:
        """
        Initialize access to a PACE HKT product

        Parameters
        ----------
        filename : str
           name of the PACE HKT product
        instrument : {'spx', 'oci', 'harp', 'sc'}, default='spx'
           name of PACE instrument 'spx': SPEXone, 'oci': OCI, 'harp': HARP2,
           'sc': Space Craft.
        """
        self.filename = Path(filename)
        if not self.filename.is_file():
            FileNotFoundError('HKT product does not exists')

        self._coverage = None
        self.set_instrument(instrument)

    # ---------- PUBLIC FUNCTIONS ----------
    @property
    def coverage(self) -> tuple:
        """
        Return selection of navigation data

        Returns
        -------
        tuple of two ints
            start and end of data time-coverage (sec of day)
        """
        return self._coverage

    def set_coverage(self, sec_bgn: int, sec_end: int) -> None:
        """
        Set start and end of data coverage in the navigation data

        Parameters
        ----------
        bgn_tai, end_tai: int
           Minimum and maximum value of data time-coverage (sec of day)
        """
        self._coverage = (sec_bgn, sec_end)

    @property
    def instrument(self) -> str:
        """
        Returns name of the PACE instrument

        Returns
        -------
        str
            Name of selected PACE instrument
        """
        return self._instrument

    def set_instrument(self, name: str) -> None:
        """
        Set name of PACE instrument

        Parameters
        ----------
        name :  {'spx', 'oci', 'harp', 'sc'}, default='spx'
            name of PACE instrument
        """
        if name is None:
            self._instrument = 'spx'
        elif name.lower() in ('spx', 'oci', 'harp', 'sc'):
            self._instrument = name.lower()
        else:
            raise KeyError('invalid name of instrument')

    def navigation(self) -> xr.Dataset:
        """
        Get navigation data
        """
        res = {}
        with h5py.File(self.filename) as fid:
            gid = fid['navigation_data']
            for key in gid:
                res[key] = h5_to_xr(gid[key])
        return xr.Dataset(res)

    def housekeeping(self, apid=None) -> np.ndarray:
        """
        Get housekeeping data

        Parameters
        ----------
        apid :  int, default=None
            select housekeeping data of APID, and convert byte-blobs to
            structured arrays (currently only implemented for SPEX)
        """
        dtype = None
        if apid is not None:
            dtype = {'spx': tmtc_dtype,
                     'oci': None,
                     'harp': None,
                     'sc': None}.get(self.instrument)

        ds_set = {'spx': 'SPEXone_HKT_packets',
                  'oci': 'OCI_HKT_packets',
                  'harp': 'HARP2_HKT_packets',
                  'sc': 'SC_HKT_packets'}.get(self.instrument)
        with h5py.File(self.filename) as fid:
            # check size of dataset
            res = fid['housekeeping_data'][ds_set][:]
            if res.size > 0 and dtype is not None:
                ii = 0
                buff = np.empty(res.shape[0], dtype=dtype(apid))
                for packet in res:
                    packet_id = np.frombuffer(packet, count=1, offset=0,
                                              dtype='>u2')[0]
                    if (packet_id & 0x7FF) == apid:
                        buff[ii] = np.frombuffer(packet, count=1, offset=0,
                                                 dtype=dtype(apid))
                        ii += 1
                res = buff[:ii]

        return res

def test():
    """
    function test
    """
    filename = '/data/richardh/SPEXone/HKT/20220617/PACE.20220617T025000.HKT.nc'
    filename = '/data/richardh/SPEXone/HKT/20220617/PACE.20220621T142822.HKT.nc'

    hkt = HKTio(filename)
    print(hkt.filename, hkt.instrument, hkt.coverage)
    print('navigation data')
    print(hkt.navigation())
    print('housekeeping data')
    print('spx: ', hkt.housekeeping(apid=0x320))

    hkt.set_instrument('sc')
    print('sc: ', hkt.housekeeping(apid=0x6c))


# --------------------------------------------------
if __name__ == '__main__':
    test()
