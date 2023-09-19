#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Contains the class `HKTio` to read PACE HKT products."""

from __future__ import annotations

__all__ = ['HKTio', 'check_coverage_nav', 'read_hkt_nav']

import logging
from datetime import datetime, time, timedelta, timezone
from enum import IntFlag, auto
from typing import TYPE_CHECKING

import h5py
import numpy as np
import xarray as xr
from moniplot.image_to_xarray import h5_to_xr

# pylint: disable=no-name-in-module
from netCDF4 import Dataset

from .lib.ccsds_hdr import CCSDShdr
from .lib.leap_sec import get_leap_seconds

if TYPE_CHECKING:
    from pathlib import Path

# - global parameters -----------------------
module_logger = logging.getLogger('pyspex.hkt_io')

EPOCH = datetime(1958, 1, 1, tzinfo=timezone.utc)

# valid data coverage range
VALID_COVERAGE_MIN = datetime(2021, 1, 1, tzinfo=timezone.utc)
VALID_COVERAGE_MAX = datetime(2035, 1, 1, tzinfo=timezone.utc)

# expect the navigation data to extend at least 2 minutes at start and end
TIMEDELTA_MIN = timedelta(seconds=2 * 60)


class CoverageFlag(IntFlag):
    """Define flags for coverage_quality (navigation_data)."""

    GOOD = 0
    MISSING_SAMPLES = auto()
    TOO_SHORT_EXTENDS = auto()
    NO_EXTEND_AT_START = auto()
    NO_EXTEND_AT_END = auto()


# - high-level r/w functions ------------
def read_hkt_nav(hkt_list: list[Path, ...]) -> xr.Dataset:
    """Read navigation data from one or more HKT products.

    Parameters
    ----------
    hkt_list : list[Path, ...]
       list of PACE-HKT products collocated with SPEXone measurements

    Returns
    -------
    xr.Dataset
       xarray dataset with PACE navigation data
    """
    dim_dict = {'att_': 'att_time',
                'orb_': 'orb_time',
                'tilt': 'tilt_time'}

    # concatenate DataArrays with navigation data
    res = {}
    rdate = None
    for name in sorted(hkt_list):
        hkt = HKTio(name)
        nav = hkt.navigation()
        if not res:
            rdate = hkt.reference_date
            res = nav.copy()
            continue

        dtime = None
        if rdate != hkt.reference_date:
            dtime = hkt.reference_date - rdate

        for key1, value in nav.items():
            if not value:
                continue

            hdim = dim_dict.get(key1, None)
            if dtime is None:
                res[key1] = xr.concat((res[key1], value), dim=hdim)
            else:
                parm = key1 + '_time' if key1[-1] != '_' else key1 + 'time'
                val_new = value.assign_coords(
                    {parm: value[parm] + dtime.total_seconds()})
                res[key1] = xr.concat((res[key1], val_new), dim=hdim)

    # make sure that the data is sorted and unique
    dsets = ()
    for key, coord in dim_dict.items():
        if not res[key]:
            continue

        res[key] = res[key].sortby(coord)
        res[key] = res[key].drop_duplicates(dim=coord)
        dsets += (res[key],)

    # create Dataset from DataArrays
    xds_nav = xr.merge(dsets, combine_attrs='drop_conflicts')

    # remove confusing attributes from Dataset
    key_list = list(xds_nav.attrs)
    for key in key_list:
        del xds_nav.attrs[key]

    # add attribute 'reference_date'
    return xds_nav.assign_attrs({'reference_date': rdate.isoformat()})


def check_coverage_nav(l1a_file: Path, xds_nav: xr.Dataset) -> None:
    """Check time coverage of navigation data.

    Parameters
    ----------
    l1a_file :  Path
       name of the SPEXone level-1A product
    xds_nav :  xr.Dataset
       xarray dataset with PACE navigation data
    """
    coverage_quality = CoverageFlag.GOOD
    # obtain the reference date of the navigation data
    ref_date = datetime.fromisoformat(xds_nav.attrs['reference_date'])

    # obtain time_coverage_range from the Level-1A product
    with h5py.File(l1a_file) as fid:
        # pylint: disable=no-member
        val = fid.attrs['time_coverage_start'].decode()
        coverage_start = datetime.fromisoformat(val)
        val = fid.attrs['time_coverage_end'].decode()
        coverage_end = datetime.fromisoformat(val)
    module_logger.debug('SPEXone time-coverage: %s - %s',
                        coverage_start, coverage_end)

    # check at the start of the data
    sec_of_day = xds_nav['att_time'].values[0]
    att_coverage_start = ref_date + timedelta(seconds=sec_of_day)
    module_logger.debug('PACE-HKT time-coverage-start: %s', att_coverage_start)
    if coverage_start - att_coverage_start < timedelta(0):
        coverage_quality |= CoverageFlag.NO_EXTEND_AT_START
        module_logger.error('time coverage of navigation data starts'
                            ' after "time_coverage_start"')
    if coverage_start - att_coverage_start < TIMEDELTA_MIN:
        coverage_quality |= CoverageFlag.TOO_SHORT_EXTENDS
        module_logger.warning('time coverage of navigation data starts after'
                              ' "time_coverage_start - %s"', TIMEDELTA_MIN)

    # check at the end of the data
    sec_of_day = xds_nav['att_time'].values[-1]
    att_coverage_end = ref_date + timedelta(seconds=sec_of_day)
    module_logger.debug('PACE-HKT time-coverage-end: %s', att_coverage_end)
    if att_coverage_end - coverage_end < timedelta(0):
        coverage_quality |= CoverageFlag.NO_EXTEND_AT_END
        module_logger.error('time coverage of navigation data ends'
                            ' before "time_coverage_end"')
    if att_coverage_end - coverage_end < TIMEDELTA_MIN:
        coverage_quality |= CoverageFlag.TOO_SHORT_EXTENDS
        module_logger.warning('time coverage of navigation data ends before'
                              ' "time_coverage_end + %s"', TIMEDELTA_MIN)

    # check for completeness
    dtime = (att_coverage_end - att_coverage_start).total_seconds()
    dim_expected = round(dtime / np.median(np.diff(xds_nav['att_time'])))
    module_logger.debug('expected navigation samples %d found %d',
                        len(xds_nav['att_time']), dim_expected)
    if len(xds_nav['att_time']) / dim_expected < 0.95:
        coverage_quality |= CoverageFlag.MISSING_SAMPLES
        module_logger.warning('navigation data poorly sampled')

    # add coverage flag and attributes to Level-1A product
    with Dataset(l1a_file, 'a') as fid:
        gid = fid['/navigation_data']
        gid.time_coverage_start = att_coverage_start.isoformat(
            timespec='milliseconds')
        gid.time_coverage_end = att_coverage_end.isoformat(
            timespec='milliseconds')
        dset = gid.createVariable('coverage_quality', 'u1', fill_value=255)
        dset[:] = coverage_quality
        dset.long_name = 'coverage quality of navigation data'
        dset.standard_name = 'status_flag'
        dset.valid_range = np.array([0, 15], dtype='u2')
        dset.flag_values = np.array([0, 1, 2, 4, 8], dtype='u2')
        dset.flag_meanings = ('good missing-samples too_short_extends'
                              ' no_extend_at_start no_extend_at_end')

    # generate warning if time-coverage of navigation data is too short
    if coverage_quality & CoverageFlag.TOO_SHORT_EXTENDS:
        return False

    return True


# - class HKTio -------------------------
class HKTio:
    """Class to read housekeeping and navigation data from PACE-HKT products.

    Parameters
    ----------
    filename : Path
        name of the PACE HKT product

    Notes
    -----
    This class has the following methods::

     - reference_date -> datetime
     - set_reference_date()
     - coverage() -> tuple[datetime, datetime]
     - housekeeping(instrument: str) -> tuple[np.ndarray, ...]
     - navigation() -> dict
    """

    def __init__(self: HKTio, filename: Path) -> None:
        """Initialize access to a PACE HKT product."""
        self._coverage = None
        self._reference_date = None
        self.filename = filename
        if not self.filename.is_file():
            raise FileNotFoundError(f'file {filename} not found')
        self.set_reference_date()

    # ---------- PUBLIC FUNCTIONS ----------
    @property
    def reference_date(self: HKTio) -> datetime:
        """Return reference date of all time_of_day variables."""
        return self._reference_date

    def set_reference_date(self: HKTio) -> None:
        """Set reference date of current PACE HKT product."""
        ref_date = None
        with h5py.File(self.filename) as fid:
            grp = fid['navigation_data']
            if 'att_time' in grp and 'units' in grp['att_time'].attrs:
                # pylint: disable=no-member
                words = grp['att_time'].attrs['units'].decode().split(' ')
                if len(words) > 2:
                    # Note timezone 'Z' is only accepted by Python 3.11+
                    ref_date = datetime.fromisoformat(words[2]
                                                      + 'T00:00:00+00:00')

        if ref_date is None:
            coverage = self.coverage()
            ref_date = datetime.combine(coverage[0].date(), time(0),
                                        tzinfo=timezone.utc)

        self._reference_date = ref_date

    def coverage(self: HKTio) -> tuple[datetime, datetime]:
        """Return data coverage."""
        one_day = timedelta(days=1)
        with h5py.File(self.filename) as fid:
            # pylint: disable=no-member
            # Note timezone 'Z' is only accepted by Python 3.11+
            val = fid.attrs['time_coverage_start'].decode()
            coverage_start = datetime.fromisoformat(val.replace('Z', '+00:00'))
            val = fid.attrs['time_coverage_end'].decode()
            coverage_end = datetime.fromisoformat(val.replace('Z', '+00:00'))

        if abs(coverage_end - coverage_start) < one_day:
            return coverage_start, coverage_end

        # Oeps, now we have to check the timestamps of the measurement data
        hk_dset_names = ('HARP2_HKT_packets', 'OCI_HKT_packets',
                         'SPEXone_HKT_packets', 'SC_HKT_packets')

        tstamp_mn_list = []
        tstamp_mx_list = []
        with h5py.File(self.filename) as fid:
            for ds_set in hk_dset_names:
                dt_list = ()
                if ds_set not in fid['housekeeping_data']:
                    continue

                res = fid['housekeeping_data'][ds_set][:]
                for packet in res:
                    try:
                        ccsds_hdr = CCSDShdr()
                        ccsds_hdr.read('raw', packet)
                    except ValueError as exc:
                        module_logger.warning(
                            'CCSDS header read error with "%s"', exc)
                        break

                    val = ccsds_hdr.tstamp(EPOCH)
                    if (val > VALID_COVERAGE_MIN) & (val < VALID_COVERAGE_MAX):
                        dt_list += (val,)

                if not dt_list:
                    continue

                dt_arr = np.array(dt_list)
                ii = dt_arr.size // 2
                leap_sec = get_leap_seconds(dt_arr[ii].timestamp(),
                                            epochyear=1970)
                dt_arr -= timedelta(seconds=leap_sec)
                mn_val = min(dt_arr)
                mx_val = max(dt_arr)
                if mx_val - mn_val > one_day:
                    indx_close_to_mn = (dt_arr - mn_val) <= one_day
                    indx_close_to_mx = (mx_val - dt_arr) <= one_day
                    module_logger.warning('coverage_range: %s[%d] - %s[%d]',
                                   mn_val, np.sum(indx_close_to_mn),
                                   mx_val, np.sum(indx_close_to_mx))
                    if np.sum(indx_close_to_mn) > np.sum(indx_close_to_mx):
                        mx_val = max(dt_arr[indx_close_to_mn])
                    else:
                        mn_val = min(dt_arr[indx_close_to_mx])

                tstamp_mn_list.append(mn_val)
                tstamp_mx_list.append(mx_val)

        if len(tstamp_mn_list) == 1:
            return tstamp_mn_list[0], tstamp_mx_list[0]

        return min(*tstamp_mn_list), max(*tstamp_mx_list)

    def housekeeping(self: HKTio,
                     instrument: str = 'spx') -> tuple[np.ndarray, ...]:
        """Get housekeeping telemetry data.

        Parameters
        ----------
        instrument : {'spx', 'oci', 'harp', 'sc'}, default='spx'
           name of PACE instrument: 'harp': HARP2, 'oci': OCI,
           'sc': spacecraft, 'spx': SPEXone.

        Notes
        -----
        Current implementation only works for SPEXone.
        """
        ds_set = {'spx': 'SPEXone_HKT_packets',
                  'sc': 'SC_HKT_packets',
                  'oci': 'OCI_HKT_packets',
                  'harp': 'HARP2_HKT_packets'}.get(instrument)

        with h5py.File(self.filename) as fid:
            if ds_set not in fid['housekeeping_data']:
                return ()
            res = fid['housekeeping_data'][ds_set][:]

        ccsds_hk = ()
        for packet in res:
            try:
                ccsds_hdr = CCSDShdr()
                ccsds_hdr.read('raw', packet)
            except ValueError as exc:
                module_logger.warning(
                    'CCSDS header read error with "%s"', exc)
                break

            try:
                dtype_apid = ccsds_hdr.data_dtype
            except ValueError:
                print(f'APID: 0x{ccsds_hdr.apid:x};'
                      f' Packet Length: {ccsds_hdr.packet_size:d}')
                dtype_apid = None

            if dtype_apid is not None:           # all valid APIDs
                buff = np.frombuffer(packet, count=1, offset=0,
                                     dtype=dtype_apid)
                ccsds_hk += (buff,)
            else:
                module_logger.warning(
                     'package with APID 0x%x and length %d is not implemented',
                     ccsds_hdr.apid, ccsds_hdr.packet_size)

        return ccsds_hk

    def navigation(self: HKTio) -> dict:
        """Get navigation data."""
        res = {'att_': (), 'orb_': (), 'tilt': ()}
        with h5py.File(self.filename) as fid:
            gid = fid['navigation_data']
            for key in gid:
                if key.startswith('att_'):
                    res['att_'] += (h5_to_xr(gid[key]),)
                elif key.startswith('orb_'):
                    res['orb_'] += (h5_to_xr(gid[key]),)
                elif key.startswith('tilt'):
                    res['tilt'] += (h5_to_xr(gid[key]),)
                else:
                    module_logger.warning('fail to find dataset %s', key)

        # repair the dimensions
        xds1 = xr.merge(res['att_'], combine_attrs='drop_conflicts')
        xds1 = xds1.set_coords(['att_time'])
        xds1 = xds1.swap_dims({'att_records': 'att_time'})
        xds2 = xr.merge(res['orb_'], combine_attrs='drop_conflicts')
        xds2 = xds2.set_coords(['orb_time'])
        xds2 = xds2.swap_dims({'orb_records': 'orb_time'})
        xds3 = ()
        if res['tilt']:
            xds3 = xr.merge(res['tilt'], combine_attrs='drop_conflicts')
            xds3 = xds3.set_coords(['tilt_time'])
            xds3 = xds3.swap_dims({'tilt_records': 'tilt_time'})
        return {'att_': xds1, 'orb_': xds2, 'tilt': xds3}
