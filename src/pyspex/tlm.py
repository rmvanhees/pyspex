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

from dataclasses import dataclass
from pathlib import Path
import datetime

import h5py
import numpy as np

from .hkt_io import HKTio
from .lib.leap_sec import get_leap_seconds
from .lib.tlm_utils import UNITS_DICT, convert_hk
from .lib.tmtc_def import tmtc_dtype
from .lv0_io import (ap_id, grouping_flag, packet_length,
                     read_lv0_data, sequence)
from .lv1_args import get_l1a_settings
from .lv1_io import get_l1a_name, L1Aio


# - global parameters -----------------------
MCP_TO_SEC = 1e-7
FULLFRAME_BYTES = 2 * 2048 * 2048

TSTAMP_TYPE = np.dtype(
    [('tai_sec', int), ('sub_sec', int), ('dt', 'O')])


# - helper functions ------------------------
def dump_hk(flname: Path, ccsds_hk: tuple):
    """Dump telemetry header info."""
    with flname.open('w', encoding='ascii') as fp:
        fp.write('APID Grouping Counter Length     TAI_SEC    SUB_SEC'
                 ' ICUSWVER MPS_ID TcSeqControl TcErrorCode\n')
        for buf in ccsds_hk:
            msg = (f"{ap_id(buf['hdr']):4x} {grouping_flag(buf['hdr']):8d}"
                   f" {sequence(buf['hdr']):7d} {packet_length(buf['hdr']):6d}"
                   f" {buf['hdr']['tai_sec']:11d} {buf['hdr']['sub_sec']:10d}")

            if ap_id(buf['hdr']) == 0x320:
                msg += (f" {buf['hk']['ICUSWVER']:8x}"
                        f" {buf['hk']['MPS_ID']:6d}")
            elif ap_id(buf['hdr']) in (0x331, 0x332, 0x333, 0x334):
                msg += f" {-1:8x} {-1:6d} {buf['TcSeqControl']:12d}"
                if ap_id(buf['hdr']) == 0x332:
                    msg += (f" {bin(buf['TcErrorCode'])}"
                            f" {buf['RejectParameter1']}"
                            f" {buf['RejectParameter2']}")
                if ap_id(buf['hdr']) == 0x334:
                    msg += (f" {bin(buf['TcErrorCode'])}"
                            f" {buf['FailParameter1']}"
                            f" {buf['FailParameter2']}")
            fp.write(msg + "\n")


def get_epoch(tstamp: int) -> datetime:
    """Return epoch of timestamp.
    """
    if tstamp < 1956528000:
        return datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

    return (datetime.datetime(1958, 1, 1, tzinfo=datetime.timezone.utc)
            - datetime.timedelta(seconds=get_leap_seconds(tstamp)))


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
    tlm = np.empty(len(ccsds_hk), dtype=tmtc_dtype(0x320))
    tstamp = []
    for ii, buf in enumerate(ccsds_hk):
        hdr[ii] = buf['hdr']
        if ap_id(hdr[ii]) != 0x320:
            continue

        tlm[ii] = buf['hk']
        tstamp.append(epoch + datetime.timedelta(
            seconds=int(buf['hdr']['tai_sec']),
            microseconds=subsec2musec(buf['hdr']['sub_sec'])))

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
    epoch = get_epoch(int(ccsds_sci[0]['hdr']['tai_sec'][0]))

    n_frames = 0
    found_start_first = False
    for buf in ccsds_sci:
        if grouping_flag(buf['hdr']) == 1:
            found_start_first = True
            if n_frames == 0:
                hdr_dtype = buf['hdr'].dtype
                hk_dtype = buf['hk'].dtype
                continue

        if not found_start_first:
            continue

        if grouping_flag(buf['hdr']) == 2:
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
    for buf in ccsds_sci:
        if grouping_flag(buf['hdr']) == 1:
            found_start_first = True
            hdr_arr[ii] = buf['hdr']
            tlm_arr[ii] = buf['hk']
            tstamp[ii] = (buf['icu_tm']['tai_sec'][0],
                          buf['icu_tm']['sub_sec'][0],
                          epoch + datetime.timedelta(
                              seconds=int(buf['icu_tm']['tai_sec'][0]),
                              microseconds=subsec2musec(
                                  buf['icu_tm']['sub_sec'][0])))
            img = (buf['frame'][0],)
            continue

        if not found_start_first:
            continue

        if grouping_flag(buf['hdr']) == 0:
            img += (buf['frame'][0],)
        elif grouping_flag(buf['hdr']) == 2:
            found_start_first = False
            img += (buf['frame'][0],)
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
        self.file_list: list | None = None
        self._verbose: bool = verbose
        self._hk = {}
        self._sci = {}

    @property
    def sci_hdr(self) -> np.ndarray | None:
        """Return CCSDS header data of Science telemetry packages.
        """
        return self._sci['hdr'] if 'hdr' in self._sci else None

    @property
    def hk_hdr(self) -> np.ndarray | None:
        """Return CCSDS header data of telemetry packages @1Hz.
        """
        return self._hk['hdr'] if 'hdr' in self._hk else None

    @property
    def sci_tlm(self) -> np.ndarray | None:
        """Return Science telemetry packages.
        """
        return self._sci['tlm'] if 'tlm' in self._sci else None

    @property
    def hk_tlm(self) -> np.ndarray | None:
        """Return telemetry packages @1Hz.
        """
        return self._hk['tlm'] if 'hdr' in self._hk else None

    @property
    def sci_tstamp(self) -> np.ndarray | None:
        """Return timestamps of Science telemetry packages.
        """
        return self._sci['tstamp'] if 'tstamp' in self._sci else None

    @property
    def hk_tstamp(self) -> np.ndarray | None:
        """Return timestamps of telemetry packages @1Hz.
        """
        return self._hk['tstamp'] if 'hdr' in self._hk else None

    @property
    def images(self) -> np.ndarray | None:
        """Return image-frames of Science telemetry packages.
        """
        return self._sci['images'] if 'images' in self._sci else None

    @property
    def dims_l1a(self):
        """Obtain image and housekeeping dimension"""
        binned_dims = None
        full_dims = None

        mask = [] if self.sci_tlm is None else \
            self.sci_tlm['IMRLEN'] == FULLFRAME_BYTES
        if np.sum(mask) > 0:
            mps_list = [int(i) for i in np.unique(self.sci_tlm['MPS_ID'])]
            if self._verbose:
                print(f'[INFO]: unique Diagnostic MPS: {mps_list}')

            full_dims = {
                'number_of_images': np.sum(mask),
                'samples_per_image': 2048 * 2048,
                'hk_packets': len([x for x in self.hk_tlm
                                   if x['MPS_ID'] in mps_list])}

        mask = [] if self.sci_tlm is None else \
            self.sci_tlm['IMRLEN'] < FULLFRAME_BYTES
        if np.sum(mask) > 0:
            mps_list = [int(i) for i in np.unique(self.sci_tlm['MPS_ID'])]
            if self._verbose:
                print(f'[INFO]: unique Science MPS: {mps_list}')

            binned_dims = {
                'number_of_images': np.sum(mask),
                'samples_per_image': np.max([x.size for x in self.images
                                             if x.size < FULLFRAME_BYTES]),
                'hk_packets': len([x for x in self.hk_tlm
                                   if x['MPS_ID'] in mps_list])}
        elif full_dims is None:
            binned_dims = {
                'number_of_images': 0,
                'samples_per_image': 2048,
                'hk_packets': len(self.hk_hdr)}

        return {'full': full_dims, 'binned': binned_dims}

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

    def from_hkt(self, flnames: Path | list[Path], *,
                 instrument: str | None = None,
                 dump_dir: Path | None = None):
        """Read telemetry data from a PACE HKT product.

        Parameters
        ----------
        flnames :  Path | list[Path]
           list of PACE_HKT filenames (netCDF-4 format)
        instrument :  {'spx', 'sc', 'oci', 'harp'}, optional
        apid :  int, optional
        """
        if isinstance(flnames, Path):
            flnames = [flnames]
        if instrument is None:
            instrument = 'spx'
        elif instrument not in ['spx', 'sc', 'oci', 'harp']:
            raise KeyError("instrument not in ['spx', 'sc', 'oci', 'harp']")

        self.file_list = flnames
        for name in flnames:
            hkt = HKTio(name, instrument)
            ccsds_hk = hkt.housekeeping()
            if not ccsds_hk:
                return

        if dump_dir is not None:
            dump_hk(dump_dir / (self.file_list[0].stem + '_hk.dump'), ccsds_hk)

        self._hk = extract_l0_hk(ccsds_hk, self._verbose)

    def from_lv0(self, flnames: Path | list[Path], *,
                 file_format: str,
                 dump_dir: Path | None = None,
                 tlm_type: str | None = None):
        """Read telemetry data from SPEXone Level-0 product.

        Parameters
        ----------
        flnames :  Path | list[Path]
           list of CCSDS filenames
        file_format : {'raw', 'st3', 'dsb'}
           type of CCSDS data
        dump_dir :  Path, default=False
           if dump_dir is not None then all data is read, but the header
           information of the telemetry packages @1Hz are dumped in the
           directory 'dump_dir' for debugging purposes.
        tlm_type :  {'hk', 'sci', 'all'}, optional
           select type of telemetry packages

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

        self.file_list = flnames
        self._hk = {}
        self._sci = {}
        ccsds_sci, ccsds_hk = \
            read_lv0_data(flnames, file_format, verbose=self._verbose)

        if dump_dir is not None:
            dump_hk(dump_dir / (self.file_list[0].stem + '_hk.dump'), ccsds_hk)

        # collect Science telemetry data
        if tlm_type != 'hk':
            self._sci = extract_l0_sci(ccsds_sci, self._verbose)
        del ccsds_sci

        # collected NomHK telemetry data
        if tlm_type != 'sci':
            self._hk = extract_l0_hk(ccsds_hk, self._verbose)

    def from_l1a(self, flname: Path, *, tlm_type: str | None = None):
        """Read telemetry data from SPEXone Level-1A product.

        Parameters
        ----------
        flname :  Path
           name of SPEXone Level-1A product
        tlm_type :  {'hk', 'sci', 'all'}, optional
           select type of telemetry packages

        Returns
        -------
        np.ndarray
        """
        if tlm_type is None:
            tlm_type = 'all'
        elif tlm_type not in ['hk', 'sci', 'all']:
            raise KeyError("tlm_type not in ['hk', 'sci', 'all']")

        self.file_list = [flname]
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
                    _dt.append(epoch + datetime.timedelta(
                        seconds=int(sec),
                        milliseconds=-self.start_integration[ii],
                        microseconds=subsec2musec(subsec[ii])))
                self._sci['tstamp']['dt'] = _dt

            if tlm_type != 'sci':
                self._hk['tlm'] = fid['/engineering_data/NomHK_telemetry'][:]
                dset = fid['/engineering_data/HK_tlm_time']
                ref_date = dset.attrs['units'].decode()[14:] + 'Z'
                epoch = datetime.datetime.fromisoformat(ref_date)
                self._hk['tstamp'] = []
                for sec in dset[:]:
                    self._hk['tstamp'].append(epoch
                                              + datetime.timedelta(seconds=sec))

    def init_l1a(self, config: dataclass):
        """Initialize SPEXone Level-1A product"""
        dims = self.dims_l1a
        if (dims['full'] is not None
            and dims['full']['number_of_images'] > 0):
            mode = 'all' if config.eclipse is None else 'full'
            prod_name = get_l1a_name(config, mode, self.sci_tstamp['dt'][0])
            dims = dims['full']
            # define reference data (timezone aware!)
            ref_date = datetime.datetime.combine(
                self.sci_tstamp['dt'][0].date(), datetime.time(0),
                self.sci_tstamp['dt'][0].tzinfo)

        if (dims['binned'] is not None
            and dims['binned']['number_of_images']) > 0:
            mode = 'all' if config.eclipse is None else 'binned'
            prod_name = get_l1a_name(config, mode, self.sci_tstamp['dt'][0])
            dims = dims['binned']
            # define reference data (timezone aware!)
            ref_date = datetime.datetime.combine(
                self.sci_tstamp['dt'][0].date(), datetime.time(0),
                self.sci_tstamp['dt'][0].tzinfo)
        else:
            prod_name = get_l1a_name(config, 'all', self.hk_tstamp['dt'][0])
            dims = dims['binned']
            # define reference data (timezone aware!)
            ref_date = datetime.datetime.combine(
                self.hk_tstamp['dt'][0].date(), datetime.time(0),
                self.hk_tstamp['dt'][0].tzinfo)
        if self._verbose:
            print(f'name of the SPEXone Level-1A product: {prod_name}')

        fid = L1Aio(config.outdir / prod_name, ref_date.date(),
                    dims, compression=config.compression)
        fid.close()
        
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
    # parse command-line parameters and YAML file for settings
    try:
        config = get_l1a_settings()
    except FileNotFoundError as exc:
        print(f'[FATAL]: FileNotFoundError exception raised with "{exc}".')
        return 100
    except TypeError as exc:
        print(f'[FATAL]: TypeError exception raised with "{exc}".')
        return 101

    # show the user command-line settings after calling `check_input_files`
    if config.verbose:
        print(config)

    # read level 0 data
    tlm = SPXtlm(config.verbose)
    try:
        # ToDo: add options debug and l0_format
        tlm.from_lv0(config.l0_list, file_format=config.l0_format,
                     tlm_type='all')
    except ValueError as exc:
        print(f'[FATAL]: ValueError exception raised with "{exc}".')
        return 110

    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('deltaT: ', np.unique(np.diff([tm.timestamp()
                                         for tm in tlm.hk_tstamp])))

    print('hk_hdr: ', len(tlm.hk_hdr), type(tlm.hk_hdr), tlm.hk_hdr.shape)
    print('hk_tlm: ', len(tlm.hk_tlm), type(tlm.hk_tlm), tlm.hk_tlm.shape)
    print('hk_tstamp: ', len(tlm.hk_tstamp), type(tlm.hk_tstamp),
          tlm.hk_tstamp[0])
    print()
    print('hdr: ', len(tlm.sci_hdr), type(tlm.sci_hdr), tlm.sci_hdr.shape)
    print('tlm: ', len(tlm.sci_tlm), type(tlm.sci_tlm), tlm.sci_tlm.shape)
    print('tstamp: ', len(tlm.sci_tstamp), type(tlm.sci_tstamp),
          tlm.sci_tstamp.shape)
    print('images: ', len(tlm.images), type(tlm.images), tlm.images[0].shape)
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('ICU_5p0V_V', tlm.convert('ICU_5p0V_V'),
          UNITS_DICT.get('ICU_5p0V_V', '1'))
    print('REG_CMV_OUTPUTMODE', tlm.convert('REG_CMV_OUTPUTMODE'),
          UNITS_DICT.get('REG_CMV_OUTPUTMODE', '1'))
    print('binning_table: ', tlm.binning_table)
    print('offs_msec: ', tlm.start_integration)
    print('deltaT: ', np.unique(np.diff(
        [tm['dt'].timestamp() for tm in tlm.sci_tstamp])))
    tlm.init_l1a(config)
    return 0


def __test2():
    tlm = SPXtlm()
    data_dir = Path('/nfs/SPEXone/ocal/pace-sds/pace_hkt/V1.0/2023/05/18/')
    for flname in data_dir.glob('*.nc'):
        print(f'filename: {flname}')
        tlm.from_hkt(flname, instrument='spx',
                     dump_dir=Path('/data/richardh'))
        print('deltaT: ', np.unique(np.diff([tm.timestamp()
                                             for tm in tlm.hk_tstamp])))
    return

    data_dir = Path('/data2/richardh/SPEXone/pace_hkt/V1.0/2023/05/25')
    if not data_dir.is_dir():
        data_dir = Path('/nfs/SPEXone/ocal/pace-sds/pace_hkt/V1.0/2023/05/25')
    flnames = [data_dir / 'PACE.20230525T043614.HKT.nc',
               data_dir / 'PACE.20230525T043911.HKT.nc']
    tlm = SPXtlm()
    tlm.from_hkt(flnames, instrument='spx')
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('deltaT: ', np.unique(np.diff([tm.timestamp()
                                         for tm in tlm.hk_tstamp])))


def __test3():
    data_dir = Path('/data2/richardh/SPEXone/spx1_l1a/0x12d/2023/05/25')
    if not data_dir.is_dir():
        data_dir = Path('/data/richardh/SPEXone/spx1_l1a/0x12d/2023/05/25')
    flname = data_dir / 'PACE_SPEXONE_OCAL.20230525T025431.L1A.nc'
    tlm = SPXtlm()
    tlm.from_l1a(flname, tlm_type='hk')
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('deltaT: ', np.unique(np.diff([tm.timestamp()
                                         for tm in tlm.sci_tstamp])))

    tlm.from_l1a(flname, tlm_type='sci')
    print('TS2_HOUSING_N_T', tlm.convert('TS2_HOUSING_N_T'),
          UNITS_DICT.get('TS2_HOUSING_N_T', '1'))
    print('binning_table: ', tlm.binning_table)
    print('offs_msec: ', tlm.start_integration)
    print('deltaT: ', np.unique(np.diff(
        [tm['dt'].timestamp() for tm in tlm.sci_tstamp])))


if __name__ == '__main__':
    __test1()
