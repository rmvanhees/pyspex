#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Contains the class `SPXtlm` to read/access/convert telemetry house-keeping
parameters of SPEXone.
"""
from __future__ import annotations
__all__ = ['SPXtlm']

from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np

from .lib.leap_sec import get_leap_seconds
from .lib.tlm_utils import UNITS_DICT, convert_hk
from .hkt_io import HKTio
from .lv0_io import ap_id, grouping_flag, read_lv0_data

# - global parameters -----------------------
MCP_TO_SEC = 1e-7

TSTAMP_TYPE = np.dtype(
    [('tai_sec', int), ('sub_sec', int), ('dt', 'O')])


# - helper functions ------------------------
def get_epoch(tstamp: int) -> datetime:
    """Return epoch of timestamp.
    """
    if tstamp < 1956528000:
        return datetime(1970, 1, 1, tzinfo=timezone.utc)

    return (datetime(1958, 1, 1, tzinfo=timezone.utc)
            - timedelta(seconds=get_leap_seconds(tstamp)))


def subsec2musec(sub_sec: int) -> int:
    """Return subsec as microseconds.
    """
    return 100 * int(sub_sec / 65536 * 10000)


def extract_l0_hk(ccsds_hk: tuple, verbose: bool):
    """Return dictionary with NomHK telemetry data
    """
    if not ccsds_hk:
        return {}

    if verbose:
        print('[INFO]: processing housekeeping data')
    epoch = get_epoch(int(ccsds_hk[0]['hdr']['tai_sec']))

    hdr = np.empty(len(ccsds_hk),
                   dtype=ccsds_hk[0]['hdr'].dtype)
    tlm = np.empty(len(ccsds_hk),
                   dtype=ccsds_hk[0]['data']['hk'].dtype)
    tstamp = []
    for ii, segment in enumerate(ccsds_hk):
        hdr[ii] = segment['hdr']
        if ap_id(hdr[ii]) != 0x320:
            continue

        tlm[ii] = segment['data']['hk']
        tstamp.append(epoch + timedelta(
            seconds=int(segment['hdr']['tai_sec']),
            microseconds=subsec2musec(segment['hdr']['sub_sec'])))

    return {'hdr': hdr[:len(tstamp)],
            'tlm': tlm[:len(tstamp)],
            'tstamp': tstamp}


def extract_l0_sci(ccsds_sci: tuple, verbose: bool):
    """Return dictionary with Science telemetry data.
    """
    if not ccsds_sci:
        return {}

    # define epoch and allocate memory
    if verbose:
        print('[INFO]: processing DemHK data')
    epoch = get_epoch(int(ccsds_sci[0]['hdr']['tai_sec']))

    n_frames = 0
    found_start_first = False
    for segment in ccsds_sci:
        hdr = segment['hdr']
        if grouping_flag(hdr) == 1:
            found_start_first = True
            if n_frames == 0:
                hdr_dtype = segment['hdr'].dtype
                hk_dtype = segment['data']['hk'].dtype
                continue

        if not found_start_first:
            continue

        if grouping_flag(hdr) == 2:
            found_start_first = False
            n_frames += 1

    # print(f'n_frames: {n_frames}')
    if n_frames == 0:
        raise RuntimeError('no valid Science package found')

    # allocate memory
    hdr_arr = np.empty(n_frames, dtype=hdr_dtype)
    tlm_arr = np.empty(n_frames, dtype=hk_dtype)
    tstamp = np.empty(n_frames, dtype=TSTAMP_TYPE)
    images = ()

    # extract data from ccsds_sci
    ii = 0
    found_start_first = False
    for segment in ccsds_sci:
        hdr = segment['hdr']
        if grouping_flag(hdr) == 1:
            found_start_first = True
            data = segment['data'][0]
            hdr_arr[ii] = hdr
            tlm_arr[ii] = data['hk']
            tstamp[ii] = (data['icu_tm']['tai_sec'],
                          data['icu_tm']['sub_sec'],
                          epoch + timedelta(
                              seconds=int(data['icu_tm']['tai_sec']),
                              microseconds=subsec2musec(
                                  data['icu_tm']['sub_sec'])))
            img = (data['frame'],)
            continue

        if not found_start_first:
            continue

        data = segment['data'][0]
        if grouping_flag(hdr) == 0:
            img += (data['frame'],)
        elif grouping_flag(hdr) == 2:
            found_start_first = False
            img += (data['frame'],)
            images += (np.concatenate(img),)
            ii += 1
            if ii == n_frames:
                break

    return {'hdr': hdr_arr,
            'tlm': tlm_arr,
            'tstamp': tstamp,
            'images': images}


# - class SPXtlm ----------------------------
class SPXtlm:
    """Access/convert parameters of SPEXone Science telemetry data.
    """
    def __init__(self, verbose: bool = False):
        """Initialize class SPXtlm.
        """
        self.filename = None
        self._verbose = verbose
        self._hk = {}
        self._sci = {}

    @property
    def hdr(self):
        """Return CCSDS header data.
        """
        return self._sci['hdr'] if 'hdr' in self._sci else self._hk['hdr']

    @property
    def hk_hdr(self):
        """Return CCSDS NomHK header data.
        """
        return self._hk['hdr'] if 'hdr' in self._hk else ()

    @property
    def tlm(self):
        """Return HomHK data.
        """
        return self._sci['tlm'] if 'tlm' in self._sci else self._hk['tlm']

    @property
    def hk_tlm(self):
        """Return housekeeping packages.
        """
        return self._hk['tlm'] if 'hdr' in self._hk else ()

    @property
    def tstamp(self):
        """Return timestamps of CCSDS packages.
        """
        return self._sci['tstamp'] \
            if 'tstamp' in self._sci else self._hk['tstamp']

    @property
    def hk_tstamp(self):
        """Return timestamps of CCSDS NomHK packages.
        """
        return self._hk['tstamp'] if 'hdr' in self._hk else ()

    @property
    def images(self):
        """Return image-frames of CCSDS packages.
        """
        return self._sci['images'] if 'images' in self._sci else ()

    def from_hkt(self, flnames: Path | list[Path],
                 instrument: str | None = None, apid: int | None = None):
        """Read telemetry data from a PACE HKT product.

        Parameters
        ----------
        flnames :  Path
        instrument :  {'spx', 'sc', 'oci', 'harp'}, optional
        apid :  int, optional
        """
        if isinstance(flnames, Path):
            flnames = [flnames]
        if instrument is None:
            instrument = 'spx'
        elif instrument not in ['spx', 'sc', 'oci', 'harp']:
            raise KeyError("instrument not in ['spx', 'sc', 'oci', 'harp']")
        if apid is None and instrument == 'spx':
            apid = 0x320

        # self.filename = flname
        hdr = ()
        tlm = ()
        self._sci = {}
        for name in flnames:
            hkt = HKTio(name, instrument)
            res = hkt.housekeeping(apid)
            hdr += (res['hdr'],)
            tlm += (res['hk'],)

        self._hk = {'hdr': np.concatenate(hdr),
                    'tlm': np.concatenate(tlm)}
        epoch = get_epoch(int(self._hk['hdr']['tai_sec'][0]))
        self._hk['tstamp'] = []
        for sec, subsec in [(x['tai_sec'], x['sub_sec'])
                            for x in self._hk['hdr']]:
            self._hk['tstamp'].append(epoch + timedelta(
                seconds=int(sec), microseconds=subsec2musec(subsec)))

    def from_lv0(self, flnames: Path | list[Path],
                 tlm_type: str | None = None):
        """Read telemetry data from SPEXone Level-0 product.

        Parameters
        ----------
        flnames :  Path | list[Path]
        tlm_type :  {'hk', 'sci', 'all'}, optional

        Returns
        -------
        np.ndarray
        """
        if isinstance(flnames, Path):
            flnames = [flnames]
        if tlm_type is None:
            tlm_type = 'all'
        elif tlm_type not in ['hk', 'sci', 'all']:
            raise KeyError("tlm_type not in ['hk', 'sci', 'all']")

        # self.file_list = flnames
        self._hk = {}
        self._sci = {}
        res = read_lv0_data(flnames, file_format='dsb',
                            verbose=self._verbose)
        ccsds_sci, ccsds_hk = res

        # collect Science telemetry data
        if tlm_type != 'hk':
            self._sci = extract_l0_sci(ccsds_sci, self._verbose)

        del ccsds_sci

        # collected NomHK telemetry data
        if tlm_type != 'sci':
            self._hk = extract_l0_hk(ccsds_hk, self._verbose)

    def from_l1a(self, flname: Path, tlm_type: str | None = None):
        """Read telemetry dta from SPEXone Level-1A product.

        Parameters
        ----------
        flname :  Path
        tlm_type :  str

        Returns
        -------
        np.ndarray
        """
        if tlm_type is None:
            tlm_type = 'all'
        elif tlm_type not in ['hk', 'sci', 'all']:
            raise KeyError("tlm_type not in ['hk', 'sci', 'all']")

        self.filename = flname
        self._hk = {}
        self._sci = {}
        with h5py.File(flname) as fid:
            if tlm_type != 'hk':
                self._sci['tlm'] = fid['/science_data/detector_telemetry'][:]
                seconds = fid['/image_attributes/icu_time_sec'][:]
                subsec = fid['/image_attributes/icu_time_subsec'][:]
                epoch = get_epoch(int(seconds[0]))
                self._sci['tstamp'] = np.empty(len(seconds), dtype=TSTAMP_TYPE)
                self._sci['tstamp']['tai_sec'] = seconds
                self._sci['tstamp']['sub_sec'] = subsec
                _dt = []
                for ii, sec in enumerate(seconds):
                    _dt.append(epoch + timedelta(
                        seconds=int(sec),
                        milliseconds=-self.start_integration[ii],
                        microseconds=subsec2musec(subsec[ii])))
                self._sci['tstamp']['dt'] = _dt

            if tlm_type != 'sci':
                self._hk['tlm'] = fid['/engineering_data/NomHK_telemetry'][:]
                dset = fid['/engineering_data/HK_tlm_time']
                ref_date = dset.attrs['units'].decode()[14:] + 'Z'
                epoch = datetime.fromisoformat(ref_date)
                self._hk['tstamp'] = []
                for sec in dset[:]:
                    self._hk['tstamp'].append(epoch + timedelta(seconds=sec))

    @property
    def binning_table(self):
        """Return binning table identifier (zero for full-frame images).

        Notes
        -----
        Requires SPEXone DemHK, will not work with NomHK

        v126: Sometimes the MPS information is not updated for the first \
              images. We try to fix this and warn the user.
        v129: REG_BINNING_TABLE_START is stored in BE instead of LE

        Returns
        -------
        np.ndarray, dtype=int
        """
        if 'tlm' not in self._sci:
            return None

        if 'REG_FULL_FRAME' not in self._sci['tlm'].dtype.names:
            print('[WARNING]: can not determine binning table identifier')
            return np.full(len(self._sci['tlm']), -1, dtype='i1')

        full_frame = np.unique(self._sci['tlm']['REG_FULL_FRAME'])
        if len(full_frame) > 1:
            print('[WARNING]: value of REG_FULL_FRAME not unique')
            print(self._sci['tlm']['REG_FULL_FRAME'])
        full_frame = self._sci['tlm']['REG_FULL_FRAME'][-1]

        cmv_outputmode = np.unique(self._sci['tlm']['REG_CMV_OUTPUTMODE'])
        if len(cmv_outputmode) > 1:
            print('[WARNING]: value of REG_CMV_OUTPUTMODE not unique')
            print(self._sci['tlm']['REG_CMV_OUTPUTMODE'])
        cmv_outputmode = self._sci['tlm']['REG_CMV_OUTPUTMODE'][-1]

        if full_frame == 1:
            if cmv_outputmode != 3:
                raise KeyError('Diagnostic mode with REG_CMV_OUTPMODE != 3')
            return np.zeros(len(self._sci['tlm']), dtype='i1')

        if full_frame == 2:
            if cmv_outputmode != 1:
                raise KeyError('Science mode with REG_CMV_OUTPUTMODE != 1')
            bin_tbl_start = self._sci['tlm']['REG_BINNING_TABLE_START']
            indx0 = (self._sci['tlm']['REG_FULL_FRAME'] != 2).nonzero()[0]
            if indx0.size > 0:
                indx2 = (self._sci['tlm']['REG_FULL_FRAME'] == 2).nonzero()[0]
                bin_tbl_start[indx0] = bin_tbl_start[indx2[0]]
            res = 1 + (bin_tbl_start - 0x80000000) // 0x400000
            return res & 0xFF

        raise KeyError('REG_FULL_FRAME not equal to 1 or 2')

    @property
    def start_integration(self):
        """Return offset wrt start-of-integration [msec].

        Notes
        -----
        Requires SPEXone DemHK, will not work with NomHK

        Determine offset wrt start-of-integration (IMRO + 1)
        Where the default is defined as IMRO::

        - [full-frame] COADDD + 2  (no typo, this is valid for the later MPS's)
        - [binned] 2 * COADD + 1   (always valid)
        """
        if 'tlm' not in self._sci:
            return 0
        if self._sci['tlm']['ICUSWVER'][0] <= 0x123:
            return 0

        if np.bincount(self.binning_table).argmax() == 0:
            imro = self._sci['tlm']['REG_NCOADDFRAMES'] + 2
        else:
            imro = 2 * self._sci['tlm']['REG_NCOADDFRAMES'] + 1
        return self._sci['tlm']['FTI'] * (imro + 1) / 10

    def convert(self, key: str) -> np.ndarray:
        """Convert telemetry parameter to physical units.

        Parameters
        ----------
        key :  str
           Name of telemetry parameter

        Returns
        -------
        np.ndarray
        """
        tlm = self._sci['tlm'] if 'tlm' in self._sci else self._hk['tlm']
        if key.upper() not in tlm[0].dtype.names:
            raise KeyError(f'Parameter: {key.upper()} not found'
                           f' in {tlm[0].dtype.names}')

        raw_data = np.array([x[key.upper()] for x in tlm])
        return convert_hk(key.upper(), raw_data)


def __test1():
    data_dir = Path('/data2/richardh/SPEXone/spx1_lv0/0x12d/2023/05/25')
    if not data_dir.is_dir():
        data_dir = Path('/nfs/SPEXone/ocal/pace-sds/spx1_l0/0x12d/2023/05/25')
    flnames = [data_dir / 'SPX000000879.spx', data_dir / 'SPX000000880.spx']
    #flnames = [data_dir / 'SPX000000896.spx', data_dir / 'SPX000000897.spx']

    tlm = SPXtlm()
    #tlm.from_lv0(flnames, 'hk')
    #print(tlm.hdr)
    #print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
    #      UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    #print('deltaT: ', np.unique(np.diff([tm.timestamp() for tm in tlm.tstamp])))

    tlm.from_lv0(flnames, 'all')
    print('hk_hdr: ', len(tlm.hk_hdr))
    print('hk_tlm: ', len(tlm.hk_tlm))
    print('hk_tstamp: ', len(tlm.hk_tstamp))
    print()
    print('hdr: ', len(tlm.hdr))
    print('tlm: ', len(tlm.tlm))
    print('tstamp: ', len(tlm.tstamp))
    print('images: ', len(tlm.images))
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('ICU_5p0V_V', tlm.convert('ICU_5p0V_V'),
          UNITS_DICT.get('ICU_5p0V_V', '1'))
    print('REG_CMV_OUTPUTMODE', tlm.convert('REG_CMV_OUTPUTMODE'),
          UNITS_DICT.get('REG_CMV_OUTPUTMODE', '1'))
    print('binning_table: ', tlm.binning_table)
    print('offs_msec: ', tlm.start_integration)
    print('deltaT: ', np.unique(np.diff(
        [tm['dt'].timestamp() for tm in tlm.tstamp])))


def __test2():
    data_dir = Path('/data2/richardh/SPEXone/pace_hkt/V1.0/2023/05/25')
    if not data_dir.is_dir():
        data_dir = Path('/nfs/SPEXone/ocal/pace-sds/pace_hkt/V1.0/2023/05/25')
    flnames = [data_dir / 'PACE.20230525T043614.HKT.nc',
               data_dir / 'PACE.20230525T043911.HKT.nc']
    tlm = SPXtlm()
    tlm.from_hkt(flnames, instrument='spx', apid=0x320)
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('deltaT: ', np.unique(np.diff([tm.timestamp() for tm in tlm.tstamp])))


def __test3():
    data_dir = Path('/data2/richardh/SPEXone/spx1_l1a/0x12d/2023/05/25')
    if not data_dir.is_dir():
        data_dir = Path('/data/richardh/SPEXone/spx1_l1a/0x12d/2023/05/25')
    flname = data_dir / 'PACE_SPEXONE_OCAL.20230525T025431.L1A.nc'
    tlm = SPXtlm()
    tlm.from_l1a(flname, 'hk')
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('deltaT: ', np.unique(np.diff([tm.timestamp() for tm in tlm.tstamp])))

    tlm.from_l1a(flname, 'sci')
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('binning_table: ', tlm.binning_table)
    print('offs_msec: ', tlm.start_integration)
    print('deltaT: ', np.unique(np.diff(
        [tm['dt'].timestamp() for tm in tlm.tstamp])))


if __name__ == '__main__':
    __test1()
