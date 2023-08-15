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

__all__ = ['HKTio', 'read_hkt_nav', 'check_coverage_nav']

import datetime
from enum import IntFlag, auto
from pathlib import Path

import h5py
import numpy as np
import xarray as xr
# pylint: disable=no-name-in-module
from netCDF4 import Dataset
from moniplot.image_to_xarray import h5_to_xr

from .lib.ccsds_hdr import CCSDShdr
from .lib.leap_sec import get_leap_seconds
from .lv0_lib import ap_id, dtype_tmtc

# - global parameters -----------------------
DTIME_MIN = 2 * 60
EPOCH = datetime.datetime(1958, 1, 1, tzinfo=datetime.timezone.utc)


class CoverageFlag(IntFlag):
    """Define flags for coverage_quality."""

    GOOD = 0
    MISSING_SAMPLES = auto()
    SHORT_EXTEND_START = auto()
    SHORT_EXTEND_END = auto()
    TOO_SHORT_COVERAGE = auto()


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
    for name in hkt_list:
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


def check_coverage_nav(l1a_file: Path, xds_nav: xr.Dataset,
                       verbose: bool = False):
    """Check time coverage of navigation data.

    Parameters
    ----------
    l1a_file :  Path
       name of the SPEXone level-1A product
    xds_nav :  xr.Dataset
       xarray dataset with PACE navigation data
    verbose :  bool, default=False
       be verbose
    """
    # obtain the reference date of the navigation data
    ref_date = datetime.datetime.fromisoformat(xds_nav.attrs['reference_date'])

    # obtain time_coverage_range from the Level-1A product
    coverage_start = None
    coverage_end = None
    if l1a_file.is_file():
        # pylint: disable=no-member
        with Dataset(l1a_file, 'r') as fid:
            if 'time_coverage_start' in fid.ncattrs():
                dtime = fid.time_coverage_start
                if isinstance(dtime, bytes):
                    dtime = dtime.decode()
                coverage_start = datetime.datetime.fromisoformat(dtime)
            if 'time_coverage_end' in fid.ncattrs():
                dtime = fid.time_coverage_end
                if isinstance(dtime, bytes):
                    dtime = dtime.decode()
                coverage_end = datetime.datetime.fromisoformat(dtime)

    # check at the start of the data
    coverage_quality = CoverageFlag.GOOD
    sec_of_day = xds_nav['att_time'].values[0]
    att_coverage_start = ref_date + datetime.timedelta(seconds=sec_of_day)
    dtime = (coverage_start - att_coverage_start).total_seconds()
    if verbose:
        print(f'[INFO]: difference at start: {dtime} sec')
    if dtime < 0:
        coverage_quality |= CoverageFlag.TOO_SHORT_COVERAGE
        print('[ERROR]: time coverage of navigation data starts'
              ' after "time_coverage_start"')
    if dtime < DTIME_MIN:
        coverage_quality |= CoverageFlag.SHORT_EXTEND_START
        print('[WARNING]: time coverage of navigation data starts'
              f' after "time_coverage_start - {DTIME_MIN} seconds"')

    # check at the end of the data
    sec_of_day = xds_nav['att_time'].values[-1]
    att_coverage_end = ref_date + datetime.timedelta(seconds=sec_of_day)
    dtime = (att_coverage_end - coverage_end).total_seconds()
    if verbose:
        print(f'[INFO]: difference at end: {dtime} sec')
    if dtime < 0:
        coverage_quality |= CoverageFlag.TOO_SHORT_COVERAGE
        print('[ERROR]: time coverage of navigation data ends'
              ' before "time_coverage_end"')
    if dtime < DTIME_MIN:
        coverage_quality |= CoverageFlag.SHORT_EXTEND_END
        print('[WARNING]: time coverage of navigation data ends'
              f' before "time_coverage_end + {DTIME_MIN} seconds"')

    # check for completeness
    dtime = (att_coverage_end - att_coverage_start).total_seconds()
    dim_expected = round(dtime / np.median(np.diff(xds_nav['att_time'])))
    if verbose:
        print(f"[INFO]: expected navigation samples {dim_expected}"
              f" found {len(xds_nav['att_time'])}")
    if len(xds_nav['att_time']) / dim_expected < 0.95:
        coverage_quality |= CoverageFlag.MISSING_SAMPLES
        print('[WARNING]: navigation data poorly sampled')

    # add coverage flag and attributes to Level-1A product
    with Dataset(l1a_file, 'a') as fid:
        gid = fid['/navigation_data']
        gid.time_coverage_start = att_coverage_start.isoformat(
            timespec='milliseconds')
        gid.time_coverage_end = att_coverage_end.isoformat(
            timespec='milliseconds')
        dset = gid.createVariable('coverage_quality', 'u1', fill_value=255)
        dset.long_name = 'coverage quality of navigation data'
        dset.standard_name = 'status_flag'
        dset.valid_range = np.array([0, 15], dtype='u2')
        dset.flag_values = np.array([0, 1, 2, 4, 8], dtype='u2')
        dset.flag_meanings = ('good missing-samples short_extend_start'
                              ' short_extend_start too_short_coverage')

    # generate warning if time-coverage of navigation data is too short
    if coverage_quality & CoverageFlag.TOO_SHORT_COVERAGE:
        raise UserWarning('time-coverage of navigation data is too short')


# - class HKTio -------------------------
class HKTio:
    """Read PACE HKT products.

    Parameters
    ----------
    filename : Path
        name of the PACE HKT product
    """

    def __init__(self, filename: Path) -> None:
        """Initialize access to a PACE HKT product."""
        self._coverage = None
        self._reference_date = None
        self.filename = filename
        if not self.filename.is_file():
            raise FileNotFoundError('HKT product does not exists')
        self.set_reference_date()

    # ---------- PUBLIC FUNCTIONS ----------
    @property
    def reference_date(self) -> datetime.datetime:
        """Return reference date of all time_of_day variables."""
        return self._reference_date

    def set_reference_date(self):
        """Set reference date of current PACE HKT product."""
        ref_date = None
        with h5py.File(self.filename, 'r') as fid:
            grp = fid['navigation_data']
            if 'att_time' in grp and 'units' in grp['att_time'].attrs:
                # pylint: disable=no-member
                words = grp['att_time'].attrs['units'].decode().split(' ')
                if len(words) > 2:
                    ref_date = datetime.datetime.fromisoformat(
                        words[2] + 'T00Z')

        if ref_date is None:
            words = self.filename.name.split('.')[1].split('T')
            self._reference_date = datetime.datetime.strptime(
                words[0] + 'T00Z', '%Y%m%dT%H%z')
        else:
            self._reference_date = ref_date

    def coverage(self) -> tuple[datetime.datetime, datetime.datetime]:
        """Return data coverage."""
        one_day = datetime.timedelta(days=1)
        hk_dict = {
            'HARP2_HKT_packets': (),
            'OCI_HKT_packets': (),
            'SPEXone_HKT_packets': (),
            'SC_HKT_packets': ()
        }

        with h5py.File(self.filename) as fid:
            for ds_set, tstamp in hk_dict.items():
                if ds_set not in fid['housekeeping_data']:
                    continue

                res = fid['housekeeping_data'][ds_set][:]
                for packet in res:
                    try:
                        hdr = CCSDShdr(packet)
                    except ValueError as exc:
                        print(f'[WARNING]: header reading error with "{exc}"')
                        break
                    else:
                        tstamp += (hdr.tstamp(EPOCH),)
                hk_dict[ds_set] = tstamp

        mn_list = []
        mx_list = []
        for ds_set, tstamp in hk_dict.items():
            # use SC_HKT_packets only when other data is not available
            if ds_set == 'SC_HKT_packets' and mn_list:
                break
            
            if not hk_dict[ds_set]:
                continue

            values = np.array(tstamp)
            ii = values.size // 2
            leap_sec = get_leap_seconds(values[ii].timestamp(),
                                        epochyear=1970)
            values -= datetime.timedelta(seconds=leap_sec)
            mn_val = min(values)
            mx_val = max(values)
            if mx_val - mn_val > one_day:
                indx_close_to_mn = (values - mn_val) <= one_day
                indx_close_to_mx = (mx_val - values) <= one_day
                if np.sum(indx_close_to_mn) > np.sum(indx_close_to_mx):
                    mx_val = max(values[indx_close_to_mn])
                else:
                    mn_val = min(values[indx_close_to_mx])

            mn_list.append(mn_val)
            mx_list.append(mx_val)

        if len(mn_list) == 1:
            return mn_list[0], mx_list[0]

        return min(*mn_list), max(*mx_list)

    def navigation(self) -> dict:
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
                    print(f'[WARNING]: fail to find dataset {key}')

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

    def housekeeping(self, instrument: str = 'spx') -> tuple[np.ndarray, ...]:
        """Get housekeeping telemetry data.

        Parameters
        ----------
        instrument : {'spx', 'oci', 'harp', 'sc'}, default='spx'
           name of PACE instrument: 'harp': HARP2, 'oci': OCI, 
           'sc': Space Craft, 'spx': SPEXone.

        Notes
        -----
        Current implementation only works for SPEXone.
        """
        ds_set = {'spx': 'SPEXone_HKT_packets',
                  'sc': 'SC_HKT_packets',
                  'oci': 'OCI_HKT_packets',
                  'harp': 'HARP2_HKT_packets'}.get(instrument, None)

        with h5py.File(self.filename) as fid:
            if ds_set not in fid['housekeeping_data']:
                return ()
            res = fid['housekeeping_data'][ds_set][:]

        hdr_dtype = np.dtype([('type', '>u2'),
                              ('sequence', '>u2'),
                              ('length', '>u2'),
                              ('tai_sec', '>u4'),
                              ('sub_sec', '>u2')])
        ccsds_hk = ()
        for packet in res:
            try:
                hdr = np.frombuffer(packet, count=1, offset=0,
                                    dtype=hdr_dtype)[0]
            except ValueError as exc:
                print(f'[WARNING]: header reading error with "{exc}"')
                break

            if 0x320 <= ap_id(hdr) < 0x335:           # all valid APIDs
                buff = np.frombuffer(packet, count=1, offset=0,
                                     dtype=dtype_tmtc(hdr))
                ccsds_hk += (buff,)

        return ccsds_hk


def _test():
    data_dir0 = Path('/nfs/SPEXone/ocal/pace-sds/pace_hkt/V1.0/2023/07/20')
    # data_dir1 = Path('/nfs/SPEXone/ocal/pace-sds/pace_hkt/V1.0/2023/07/21')
    file_list = [
        data_dir0 / 'PACE.20230720T230606.HKT.nc',
        data_dir0 / 'PACE.20230720T230910.HKT.nc',
        data_dir0 / 'PACE.20230720T231216.HKT.nc',
        data_dir0 / 'PACE.20230720T231526.HKT.nc',
        data_dir0 / 'PACE.20230720T231836.HKT.nc',
        # data_dir0 / 'PACE.20230720T232146.HKT.nc',
        # data_dir0 / 'PACE.20230720T232456.HKT.nc',
        # data_dir1 / 'PACE.20230721T000054.HKT.nc'
    ]
    data_dir = Path('/data/richardh/SPEXone/pace-sds/pace_hkt/V1.0/2023/07/25')
    flname = data_dir / 'PACE.20230725T045720.HKT.nc'
    
    hkt = HKTio(flname)
    print(hkt.coverage())
    return

    l1a_file = 'test_hkt_io.nc'
    with Dataset(l1a_file, 'w') as fid:
        fid.time_coverage_start = '2023-07-20T23:12:16.874Z'
        fid.time_coverage_end = '2023-07-20T23:19:36.374Z'

    # read PACE navigation data from HKT files.
    xds_nav = read_hkt_nav(file_list)
    # add PACE navigation data to existing Level-1A product.
    xds_nav.to_netcdf(l1a_file, group='navigation_data', mode='a')
    # check time coverage of navigation data.
    check_coverage_nav(Path(l1a_file), xds_nav, True)

    


if __name__ == '__main__':
    _test()
