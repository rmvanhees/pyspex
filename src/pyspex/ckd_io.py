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
Contains the class CKDio to read SPEXone CKD.

References
----------
* https://spexone-cal-doc.readthedocs.io/en/latest/
"""
__all__ = ['CKDio']

from datetime import datetime, timezone
from pathlib import Path

import h5py
import xarray as xr

from moniplot.image_to_xarray import h5_to_xr


# - global parameters ------------------------------

# - local functions --------------------------------

# - class CKDio -------------------------
class CKDio:
    """Defines a class to read SPEXone CKD parameters.

    Parameters
    ----------
    ckd_file :  str
        Name of CKD file
    verbose :  bool, default=False
        Be verbose

    Examples
    --------
    Read several CKD parameters:

    >>> with CKDio('SPX1_CKD.nc') as ckd:
    >>>    dark = ckd.dark()
    >>>    fov = ckd.fov()

    """
    def __init__(self, ckd_file: Path, verbose=False) -> None:
        """Initialize class attributes.
        """
        self.verbose = verbose

        # open access to CKD product
        self.fid = h5py.File(ckd_file, "r")
        if 'processor_configuration' not in self.fid:
            raise RuntimeError('SPEXone CKD product corrupted?')

    def __enter__(self):
        """Method called to initiate the context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Method called when exiting the context manager.
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self) -> None:
        """Make sure that we close all resources.
        """
        if self.fid is not None:
            self.fid.close()

    @property
    def processor_version(self) -> str:
        """Return the version of the spexone_cal program.
        """
        # pylint: disable=no-member
        return self.fid.attrs['processor_version'].decode()

    def date_created(self, compact=False) -> str:
        """Return creation date of the CKD product.

        Parameters
        ----------
        compact :  bool
           return date in isoformat if not compact else return 'YYYYmmddHHMMSS'
        """
        # pylint: disable=no-member
        date_str = self.fid.attrs['date_created'].decode()
        date_t = datetime.strptime(date_str, "%Y %B %d %a %Z%z %H:%M:%S")
        if compact:
            return date_t.astimezone(tz=timezone.utc).strftime("%Y%m%d%H%M%S")

        return date_t.astimezone(tz=timezone.utc).isoformat()[:-6]

    @property
    def git_commit(self) -> str:
        """Return git hash of repository spexone_cal, used to generate the CKD.
        """
        # pylint: disable=no-member
        return self.fid.attrs['git_commit'].decode()

    def dark(self) -> xr.Dataset:
        """Read Dark CKD.

        Returns
        -------
        xarray.Dataset
           parameters of the SPEXone Dark CKD
        """
        try:
            gid = self.fid['DARK']
        except KeyError:
            return None
        res = ()
        if 'dark_offset' in gid:
            res += (h5_to_xr(gid['dark_offset']),)
        else:
            res += (h5_to_xr(gid['offset_long']),)
            res += (h5_to_xr(gid['offset_short']),)
        res += (h5_to_xr(gid['dark_current']),)
        return xr.merge(res, combine_attrs='drop_conflicts')

    def noise(self) -> xr.Dataset:
        """Read Noise CKD.

        Returns
        -------
        xarray.Dataset
           parameters of the SPEXone Noise CKD
        """
        try:
            gid = self.fid['NOISE']
        except KeyError:
            return None
        res = ()
        res += (h5_to_xr(gid['g']),)
        res += (h5_to_xr(gid['n']),)
        return xr.merge(res, combine_attrs='drop_conflicts')

    def nlin(self) -> xr.Dataset:
        """Read non-linearity CKD.

        Returns
        -------
        xarray.Dataset
           parameters of the SPEXone non-linearity CKD
        """
        try:
            gid = self.fid['NON_LINEARITY']
            sigmoidal = 'A' in gid
        except KeyError:
            return None

        res = ()
        if sigmoidal:
            res += (h5_to_xr(gid['A']),)
            res += (h5_to_xr(gid['B']),)
            res += (h5_to_xr(gid['C']),)
            if '/DEBUG/NON_LINEARITY' in self.fid:
                gid = self.fid['/DEBUG/NON_LINEARITY']
                res += (h5_to_xr(gid['f1']),)
                res += (h5_to_xr(gid['f2']),)
                res += (h5_to_xr(gid['c']),)
                res += (h5_to_xr(gid['r0']),)
                res += (h5_to_xr(gid['r1']),)
                res += (h5_to_xr(gid['r2']),)
                res += (h5_to_xr(gid['r3']),)
                res += (h5_to_xr(gid['r4']),)
                res += (h5_to_xr(gid['m0']),)
                # res += (h5_to_xr(gid['m1']),)
                # res += (h5_to_xr(gid['m2']),)
        else:
            res += (h5_to_xr(gid['nonlin_order']),)
            res += (h5_to_xr(gid['nonlin_knots']),)
            res += (h5_to_xr(gid['nonlin_exptimes']),)
            res += (h5_to_xr(gid['nonlin_signal_scale']),)
            res += (h5_to_xr(gid['nonlin_fit']),)
        return xr.merge(res, combine_attrs='drop_conflicts')

    def prnu(self) -> xr.DataArray:
        """Read PRNU CKD.

        Returns
        -------
        xr.DataArray
           parameters of the SPEXone PRNU CKD
        """
        try:
            gid = self.fid['PRNU']
        except KeyError:
            return None
        return h5_to_xr(gid['prnu'])

    def fov(self) -> xr.Dataset:
        """Read field-of-view CKD.

        Returns
        -------
        xarray.Dataset
           parameters of the SPEXone field-of-view CKD
        """
        try:
            gid = self.fid['FIELD_OF_VIEW']
        except KeyError:
            return None
        res = ()
        res += (h5_to_xr(gid['fov_nfov_vp']),)
        res += (h5_to_xr(gid['fov_ifov_start_vp']),)
        res += (h5_to_xr(gid['fov_act_angles']),)
        res += (h5_to_xr(gid['fov_ispat']),)
        return xr.merge(res, combine_attrs='drop_conflicts')

    def wavelength(self) -> xr.Dataset:
        """Read Wavelength CKD.

        Returns
        -------
        xarray.Dataset
           parameters of the SPEXone Wavelength CKD
        """
        try:
            gid = self.fid['WAVELENGTH']
        except KeyError:
            return None
        res = ()
        # Before radiometric calibration S and P have separate wavelength grids
        res += (h5_to_xr(gid['wave_full']),)
        # After radiometric calibration S and P are interpolated to a common
        # wavelength grid.
        res += (h5_to_xr(gid['wave_common']),)
        return xr.merge(res, combine_attrs='drop_conflicts')

    def radiometric(self) -> xr.DataArray:
        """Read Radiometric CKD.

        Returns
        -------
        xr.DataArray
           parameters of the SPEXone Radiometric CKD
        """
        try:
            gid = self.fid['RADIOMETRIC']
        except KeyError:
            return None
        return h5_to_xr(gid['rad_spectra'])

    def polarimetric(self) -> xr.Dataset:
        """Read Polarimetric CKD.

        Returns
        -------
        xarray.Dataset
           parameters of the SPEXone Polarimetric CKD
        """
        try:
            gid = self.fid['POLARIMETRIC']
        except KeyError:
            return None
        res = ()
        res += (h5_to_xr(gid['pol_m_q']),)
        res += (h5_to_xr(gid['pol_m_u']),)
        res += (h5_to_xr(gid['pol_m_t']),)
        return xr.merge(res, combine_attrs='drop_conflicts')


# - main function ----------------------------------
def __test():
    """Small function to test this module.
    """
    ckd_dir = Path('/data/richardh/SPEXone/CKD')
    if not ckd_dir.is_dir():
        ckd_dir = Path('/data/richardh/SPEXone/share/ckd')
    ckd_file = ckd_dir / 'CKD_reference_20220916_174632.nc'
    if not ckd_file.is_file():
        raise FileNotFoundError(f'{ckd_file} does not exist')

    with CKDio(ckd_file) as ckd:
        print(ckd.processor_version)
        print(ckd.date_created())
        print('# --- dark ---')
        print(ckd.dark())
        print('# --- noise ---')
        print(ckd.noise())
        print('# --- nlin ---')
        print(ckd.nlin())
        print('# --- prnu ---')
        print(ckd.prnu())
        print('# --- fov ---')
        print(ckd.fov())
        print('# --- wavelength ---')
        print(ckd.wavelength())
        print('# --- radiometric ---')
        print(ckd.radiometric())
        print('# --- polarimetric ---')
        print(ckd.polarimetric())


# --------------------------------------------------
if __name__ == '__main__':
    __test()
