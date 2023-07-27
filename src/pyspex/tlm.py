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

import datetime
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
# pylint: disable=no-name-in-module
from netCDF4 import Dataset

from .hkt_io import HKTio, read_hkt_nav, check_coverage_nav
from .lib.leap_sec import get_leap_seconds
from .lib.tlm_utils import UNITS_DICT, convert_hk
from .lib.tmtc_def import tmtc_dtype
from .lv0_io import (ap_id, dump_numhk, dump_science, grouping_flag,
                     read_lv0_data)
from .lv1_io import L1Aio, get_l1a_name

# - global parameters -----------------------
FULLFRAME_BYTES = 2 * 2048 * 2048
MCP_TO_SEC = 1e-7
TSTAMP_MIN = 1577833200

TSTAMP_TYPE = np.dtype(
    [('tai_sec', int), ('sub_sec', int), ('dt', 'O')])


# - helper functions ------------------------
def get_epoch(tstamp: int) -> datetime.datetime:
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


def extract_l0_hk(ccsds_hk: tuple, verbose: bool) -> dict | None:
    """Return dictionary with NomHK telemetry data
    """
    if not ccsds_hk:
        return None

    if verbose:
        print('[INFO]: processing housekeeping data')
    epoch = get_epoch(int(ccsds_hk[0]['hdr']['tai_sec'][0]))

    hdr = np.empty(len(ccsds_hk),
                   dtype=ccsds_hk[0]['hdr'].dtype)
    tlm = np.empty(len(ccsds_hk), dtype=tmtc_dtype(0x320))
    tstamp = []
    ii = 0
    for buf in ccsds_hk:
        hdr[ii] = buf['hdr']
        if ap_id(hdr[ii]) != 0x320:
            continue

        tlm[ii] = buf['hk']
        tstamp.append(epoch + datetime.timedelta(
            seconds=int(hdr['tai_sec'][ii]),
            microseconds=subsec2musec(hdr['sub_sec'][ii])))
        ii += 1

    return {'hdr': hdr[:ii],
            'tlm': tlm[:ii],
            'tstamp': np.array(tstamp)}


def extract_l0_sci(ccsds_sci: tuple, verbose: bool) -> dict | None:
    """Return dictionary with Science telemetry data.
    """
    if not ccsds_sci:
        return None

    # define epoch and allocate memory
    if verbose:
        print('[INFO]: processing DemHK data')
    epoch = get_epoch(int(ccsds_sci[0]['hdr']['tai_sec'][0]))

    n_frames = 0
    found_start_first = False
    for buf in ccsds_sci:
        hdr = buf['hdr'][0]
        if grouping_flag(hdr) == 1:
            found_start_first = True
            if n_frames == 0:
                hdr_dtype = buf['hdr'].dtype
                hk_dtype = buf['hk'].dtype
                continue

        if not found_start_first:
            continue

        if grouping_flag(hdr) == 2:
            found_start_first = False
            n_frames += 1

    # print(f'n_frames: {n_frames}')
    if n_frames == 0:
        print('[WARNING]: no valid Science package found')
        return None

    # allocate memory
    hdr_arr = np.empty(n_frames, dtype=hdr_dtype)
    tlm_arr = np.empty(n_frames, dtype=hk_dtype)
    tstamp = np.empty(n_frames, dtype=TSTAMP_TYPE)
    images = ()

    # extract data from ccsds_sci
    ii = 0
    found_start_first = False
    for buf in ccsds_sci:
        hdr = buf['hdr'][0]
        if grouping_flag(hdr) == 1:
            found_start_first = True
            hdr_arr[ii] = buf['hdr']
            tlm_arr[ii] = buf['hk']
            tstamp[ii] = (buf['icu_tm']['tai_sec'][0],
                          buf['icu_tm']['sub_sec'][0],
                          epoch + datetime.timedelta(
                              seconds=int(buf['icu_tm']['tai_sec'][0]),
                              microseconds=subsec2musec(
                                  buf['icu_tm']['sub_sec'][0])))
            img: tuple[np.ndarray] = (buf['frame'][0],)
            continue

        if not found_start_first:
            continue

        if grouping_flag(hdr) == 0:
            img += (buf['frame'][0],)
        elif grouping_flag(hdr) == 2:
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


def add_hkt_navigation(l1a_file: Path, hkt_list: list[Path],
                       verbose: bool = False):
    """add PACE navigation information from PACE_HKT products.

    Parameters
    ----------
    l1a_file :  Path
       name of an existing L1A product.
    hkt_list :  list[Path]
       listing of files from which the navigation data has to be read
    verbose :  bool, default=False
       be verbose
    """
    # read PACE navigation data from HKT files.
    xds_nav = read_hkt_nav(hkt_list)
    # add PACE navigation data to existing Level-1A product.
    xds_nav.to_netcdf(l1a_file, group='navigation_data', mode='a')
    # check time coverage of navigation data.
    check_coverage_nav(l1a_file, xds_nav, verbose)


def add_proc_conf(l1a_file: Path, yaml_conf: Path):
    """Add dataset 'processor_configuration' to an existing L1A product.

    Parameters
    ----------
    l1a_file :  Path
       name of an existing L1A product.
    yaml_conf :  Path
       name of the YAML file with the processor settings
    """
    with Dataset(l1a_file, 'r+') as fid:
        dset = fid.createVariable('processor_configuration', str)
        dset.comment = ('Configuration parameters used during'
                        ' the processor run that produced this file.')
        dset[0] = ''.join(
            [s for s in yaml_conf.open(encoding='ascii').readlines()
             if not (s == '\n' or s.startswith('#'))])


# - class SPXtlm ----------------------------
class SPXtlm:
    """Access/convert parameters of SPEXone Science telemetry data.
    """
    def __init__(self, verbose: bool = False):
        """Initialize class SPXtlm.
        """
        self.file_list: list | None = None
        self._verbose: bool = verbose
        self._hk = None
        self._sci = None
        self._selection = None

    @property
    def hk_hdr(self) -> np.ndarray | None:
        """Return CCSDS header data of telemetry packages @1Hz.
        """
        if self._hk is None:
            return None

        if self._selection is None or self._selection:
            return self._hk['hdr']
        return self._hk['hdr'][self._selection['hk_mask']]

    @property
    def hk_tlm(self) -> np.ndarray | None:
        """Return telemetry packages @1Hz.
        """
        if self._hk is None:
            return None

        if self._selection is None:
            return self._hk['tlm']
        return self._hk['tlm'][self._selection['hk_mask']]

    @property
    def hk_tstamp(self) -> np.ndarray | None:
        """Return timestamps of telemetry packages @1Hz.
        """
        if self._hk is None:
            return None

        if self._selection is None:
            return self._hk['tstamp']
        return self._hk['tstamp'][self._selection['hk_mask']]

    @property
    def sci_hdr(self) -> np.ndarray | None:
        """Return CCSDS header data of Science telemetry packages.
        """
        if self._sci is None:
            return None

        if self._selection is None:
            return self._sci['hdr']
        return self._sci['hdr'][self._selection['sci_mask']]

    @property
    def sci_tlm(self) -> np.ndarray | None:
        """Return Science telemetry packages.
        """
        if self._sci is None:
            return None

        if self._selection is None:
            return self._sci['tlm']
        return self._sci['tlm'][self._selection['sci_mask']]

    @property
    def sci_tstamp(self) -> np.ndarray | None:
        """Return timestamps of Science telemetry packages.
        """
        if self._sci is None:
            return None

        if self._selection is None:
            return self._sci['tstamp']
        return self._sci['tstamp'][self._selection['sci_mask']]

    @property
    def images(self) -> tuple | None:
        """Return image-frames of Science telemetry packages.
        """
        if self._sci is None:
            return None

        if self._selection is None:
            return self._sci['images']

        images = ()
        for ii, img in enumerate(self._sci['images']):
            if self._selection['sci_mask'][ii]:
                images += (img,)
        return images

    def __get_valid_tstamps(self) -> np.ndarray | None:
        """Return valid timestamps from Science or nomHK packages."""
        if self.sci_tstamp is None \
                or np.all(self.sci_tstamp['tai_sec'] < TSTAMP_MIN):
            indx = self.hk_tstamp > datetime.datetime(
                2020, 1, 1, 1, tzinfo=datetime.timezone.utc)
            return self.hk_tstamp[indx] if indx.size > 0 else None

        indx = np.where(self.sci_tstamp['tai_sec'] > TSTAMP_MIN)[0]
        return self.sci_tstamp['dt'][indx] if indx.size > 0 else None

    @property
    def reference_date(self) -> datetime.date:
        """Return date of reference day (tzone aware)."""
        tstamp = self.__get_valid_tstamps()
        if tstamp is None:
            raise ValueError('no valid timestamps found')

        return datetime.datetime.combine(
                tstamp[0].date(), datetime.time(0), tstamp[0].tzinfo)

    @property
    def time_coverage_start(self) -> datetime:
        """Return a string for the time_coverage_start."""
        tstamp = self.__get_valid_tstamps()
        if tstamp is None:
            raise ValueError('no valid timestamps found')

        return tstamp[0]

    @property
    def time_coverage_end(self) -> datetime:
        """Return a string for the time_coverage_end."""
        tstamp = self.__get_valid_tstamps()
        if tstamp is None:
            raise ValueError('no valid timestamps found')

        return tstamp[-1]

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
        if self.sci_tlm is None:
            return None

        bin_tbl = np.zeros(len(self.sci_tlm), dtype='i1')
        _mm = self.sci_tlm['IMRLEN'] == FULLFRAME_BYTES
        if np.sum(_mm) == len(self.sci_tlm):
            return bin_tbl

        bin_tbl_start = self.sci_tlm['REG_BINNING_TABLE_START']
        bin_tbl[~_mm] = 1 + (bin_tbl_start[~_mm] - 0x80000000) // 0x400000
        return bin_tbl

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
        if self.sci_tlm is None:
            return 0
        if self.sci_tlm['ICUSWVER'][0] <= 0x123:
            return 0

        if np.bincount(self.binning_table).argmax() == 0:
            imro = self.sci_tlm['REG_NCOADDFRAMES'] + 2
        else:
            imro = 2 * self.sci_tlm['REG_NCOADDFRAMES'] + 1
        return self.sci_tlm['FTI'] * (imro + 1) / 10

    @property
    def digital_offset(self) -> np.ndarray:
        """Returns digital offset including ADC offset [count].
        """
        buff = self.sci_tlm['DET_OFFSET'].astype('i4')
        buff[buff >= 8192] -= 16384

        return buff + 70

    def from_hkt(self, flnames: Path | list[Path], *,
                 instrument: str | None = None, dump: bool = False):
        """Read telemetry data from a PACE HKT product.

        Parameters
        ----------
        flnames :  Path | list[Path]
           list of PACE_HKT filenames (netCDF-4 format)
        instrument :  {'spx', 'sc', 'oci', 'harp'}, optional
        dump :  bool, default=False
           dump header information of the telemetry packages @1Hz for
           debugging purposes
        """
        if isinstance(flnames, Path):
            flnames = [flnames]
        if instrument is None:
            instrument = 'spx'
        elif instrument not in ['spx', 'sc', 'oci', 'harp']:
            raise KeyError("instrument not in ['spx', 'sc', 'oci', 'harp']")

        self.file_list = flnames
        ccsds_hk: tuple[np.ndarray] | tuple = ()
        for name in flnames:
            hkt = HKTio(name, instrument)
            ccsds_hk += hkt.housekeeping()

        if not ccsds_hk:
            return

        if dump:
            dump_numhk(flnames[0].stem + '_hk.dump', ccsds_hk)

        self._hk = extract_l0_hk(ccsds_hk, self._verbose)

    def from_lv0(self, flnames: Path | list[Path], *,
                 file_format: str, tlm_type: str | None = None,
                 debug: bool = False, dump: bool = False):
        """Read telemetry data from SPEXone Level-0 product.

        Parameters
        ----------
        flnames :  Path | list[Path]
           list of CCSDS filenames
        file_format : {'raw', 'st3', 'dsb'}
           type of CCSDS data
        tlm_type :  {'hk', 'sci', 'all'}, optional
           select type of telemetry packages.
           Note that we allways read the complete Level-0 producs.
        debug : bool, default=False
           run in debug mode, read only packages heades
        dump :  bool, default=False
           dump header information of the telemetry packages @1Hz for
           debugging purposes
        """
        if isinstance(flnames, Path):
            flnames = [flnames]
        if tlm_type is None:
            tlm_type = 'all'
        elif tlm_type not in ['hk', 'sci', 'all']:
            raise KeyError("tlm_type not in ['hk', 'sci', 'all']")

        self.file_list = flnames
        self._hk = None
        self._sci = None
        ccsds_sci, ccsds_hk = read_lv0_data(
            flnames, file_format, debug=debug, verbose=self._verbose)
        if dump:
            dump_numhk(flnames[0].stem + '_hk.dump', ccsds_hk)
            dump_science(flnames[0].stem + '_sci.dump', ccsds_sci)
        if debug or dump:
            return

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
           name of one SPEXone Level-1A product
        tlm_type :  {'hk', 'sci', 'all'}, optional
           select type of telemetry packages
        """
        if tlm_type is None:
            tlm_type = 'all'
        elif tlm_type not in ['hk', 'sci', 'all']:
            raise KeyError("tlm_type not in ['hk', 'sci', 'all']")

        self.file_list = [flname]
        self._hk = None
        self._sci = None
        with h5py.File(flname) as fid:
            if tlm_type != 'hk':
                seconds = fid['/image_attributes/icu_time_sec'][:]
                subsec = fid['/image_attributes/icu_time_subsec'][:]
                epoch = get_epoch(int(seconds[0]))
                self._sci = {
                    'tlm': fid['/science_data/detector_telemetry'][:],
                    'tstamp': np.empty(len(seconds), dtype=TSTAMP_TYPE)
                }
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
                self._hk = {
                    'tlm': fid['/engineering_data/NomHK_telemetry'][:],
                    'tstamp': []
                }
                dset = fid['/engineering_data/HK_tlm_time']
                # pylint: disable=no-member
                ref_date = dset.attrs['units'].decode()[14:] + 'Z'
                epoch = datetime.datetime.fromisoformat(ref_date)
                for sec in dset[:]:
                    self._hk['tstamp'].append(
                        epoch + datetime.timedelta(seconds=sec))

    def set_selection(self, mode: str):
        """Obtain image and housekeeping dimensions.

        Parameters
        ----------
        mode :  {'full', 'binned', 'all'}
        """
        self._selection = None
        if mode == 'full':
            sci_mask = [] if self.sci_tlm is None else \
                self.sci_tlm['IMRLEN'] == FULLFRAME_BYTES
            if np.sum(sci_mask) == 0:
                return

            mps_list = np.unique(self.sci_tlm['MPS_ID'][sci_mask])
            if self._verbose:
                print(f'[INFO]: unique Diagnostic MPS: {mps_list}')
            hk_mask = np.in1d(self.hk_tlm['MPS_ID'], mps_list)

            self._selection = {
                'sci_mask': sci_mask,
                'hk_mask': hk_mask,
                'dims': {
                    'number_of_images': np.sum(sci_mask),
                    'samples_per_image': 2048 * 2048,
                    'hk_packets': np.sum(hk_mask)}
            }
            return

        if mode == 'binned':
            sci_mask = [] if self.sci_tlm is None else \
                self.sci_tlm['IMRLEN'] < FULLFRAME_BYTES
            if np.sum(sci_mask) == 0:
                return

            mps_list = np.unique(self.sci_tlm['MPS_ID'][sci_mask])
            if self._verbose:
                print(f'[INFO]: unique Science MPS: {mps_list}')
            hk_mask = np.in1d(self.hk_tlm['MPS_ID'], mps_list)
            self._selection = {
                'sci_mask': sci_mask,
                'hk_mask': hk_mask,
                'dims': {
                    'number_of_images': np.sum(sci_mask),
                    'samples_per_image': np.max(
                        [len(self.images[ii])
                         for ii in sci_mask.nonzero()[0]]),
                    'hk_packets': np.sum(hk_mask)}
            }
            return

        if mode == 'all':
            nr_hk = 0 if self.hk_hdr is None else len(self.hk_hdr)
            nr_sci = 0 if self.sci_hdr is None else len(self.sci_hdr)
            self._selection = {
                'hk_mask': np.full(nr_hk, True),
                'sci_mask': np.full(nr_sci, True),
                'dims': {
                    'number_of_images': nr_sci,
                    'samples_per_image': 2048
                    if nr_sci == 0 else np.max([len(x) for x in self.images]),
                    'hk_packets': nr_hk}
            }

    def gen_l1a(self, config: dataclass, mode: str):
        """Generate a SPEXone Level-1A product"""
        self.set_selection(mode)
        if self._selection is None:
            return

        prod_mode = 'all' if config.eclipse is None else mode
        ref_date = self.reference_date
        prod_name = get_l1a_name(config, prod_mode, self.time_coverage_start)
        with L1Aio(config.outdir / prod_name,
                   ref_date, self._selection['dims'],
                   compression=config.compression) as l1a:
            if self.hk_tlm is None:
                l1a.set_attr('icu_sw_version',
                             f'0x{self.hk_tlm["ICUSWVER"][0]:x}')
            l1a.fill_global_attrs(inflight=config.l0_format != 'raw')
            l1a.set_attr('time_coverage_start',
                         self.time_coverage_start.isoformat(
                             timespec='milliseconds'))
            l1a.set_attr('time_coverage_end',
                         self.time_coverage_end.isoformat(
                             timespec='milliseconds'))
            l1a.set_attr('input_files', [x.name for x in config.l0_list])
            if self._verbose:
                print('[INFO]: 1) initialized Level-1A product')

            self._fill_engineering(l1a)
            if self._verbose:
                print('[INFO]: 2) added engineering data')
            self._fill_science(l1a)
            if self._verbose:
                print('[INFO]: 3) added science data')
            self._fill_image_attrs(l1a, config.l0_format)
            if self._verbose:
                print('[INFO]: 4) added image attributes')

        # add PACE navigation information from HKT products
        if config.hkt_list:
            add_hkt_navigation(config.outdir / prod_name,
                               config.hkt_list, self._verbose)
            if self._verbose:
                print('[INFO]: 5) added PACE navigation data')

        # add processor_configuration
        if config.yaml_fl:
            add_proc_conf(config.outdir / prod_name, config.yaml_fl)

        if self._verbose:
            print(f'[INFO]: successfully generated: {prod_name}')

    def _fill_engineering(self, l1a):
        """Fill datasets in group '/engineering_data'."""
        if self.hk_tlm is None:
            return
        l1a.set_dset('/engineering_data/NomHK_telemetry', self.hk_tlm)
        ref_date = self.reference_date
        l1a.set_dset('/engineering_data/HK_tlm_time',
                     [(x - ref_date).total_seconds() for x in self.hk_tstamp])
        l1a.set_dset('/engineering_data/temp_detector',
                     self.convert('TS1_DEM_N_T', tm_type='hk'))
        l1a.set_dset('/engineering_data/temp_housing',
                     self.convert('TS2_HOUSING_N_T', tm_type='hk'))
        l1a.set_dset('/engineering_data/temp_radiator',
                     self.convert('TS3_RADIATOR_N_T', tm_type='hk'))

    def _fill_science(self, l1a):
        """Fill datasets in group '/science_data'."""
        if self.sci_tlm is None:
            return

        img_sz = [img.size for img in self.images]
        if len(np.unique(img_sz)) != 1:
            images = np.zeros((len(img_sz), np.max(img_sz)), dtype='u2')
            for ii, img in enumerate(self.images):
                images[ii, :len(img)] = img
        else:
            images = np.vstack(self.images)
        l1a.set_dset('/science_data/detector_images', images)
        l1a.set_dset('/science_data/detector_telemetry', self.sci_tlm)

    def _fill_image_attrs(self, l1a, lv0_format: str):
        """Fill datasets in group '/image_attributes'."""
        if self.sci_tlm is None:
            return

        l1a.set_dset('/image_attributes/icu_time_sec',
                     self.sci_tstamp['tai_sec'])
        # modify attribute units for non-DSB products
        if lv0_format != 'dsb':
            l1a.set_attr('valid_min', np.uint32(1577800000),
                         ds_name='/image_attributes/icu_time_sec')
            l1a.set_attr('valid_max', np.uint32(1735700000),
                         ds_name='/image_attributes/icu_time_sec')
            l1a.set_attr('units', "seconds since 1970-01-01 00:00:00",
                         ds_name='/image_attributes/icu_time_sec')
        l1a.set_dset('/image_attributes/icu_time_subsec',
                     self.sci_tstamp['sub_sec'])
        ref_date = self.reference_date
        l1a.set_dset('/image_attributes/image_time',
                     [(x - ref_date).total_seconds()
                      for x in self.sci_tstamp['dt']])
        l1a.set_dset('/image_attributes/image_ID',
                     np.bitwise_and(self.sci_hdr['sequence'], 0x3fff))
        l1a.set_dset('/image_attributes/binning_table', self.binning_table)
        l1a.set_dset('/image_attributes/digital_offset', self.digital_offset)
        l1a.set_dset('/image_attributes/exposure_time',
                     1.29e-5 * (0.43 * self.sci_tlm['DET_FOTLEN']
                                + self.sci_tlm['DET_EXPTIME']))
        l1a.set_dset('/image_attributes/nr_coadditions',
                     self.sci_tlm['REG_NCOADDFRAMES'])

    def convert(self, key: str, tm_type: str = 'both') -> np.ndarray:
        """Convert telemetry parameter to physical units.

        Parameters
        ----------
        key :  str
           Name of telemetry parameter
        tm_type :  {'hk', 'sci', 'both'}, default 'both'
           Default is to check if key is present in sci_tlm else hk_tlm
        
        Returns
        -------
        np.ndarray
        """
        if tm_type == 'hk':
            tlm = self.hk_tlm
        elif tm_type == 'sci':
            tlm = self.sci_tlm
        else:
            tlm = self.sci_tlm if self.sci_tlm is not None else self.hk_tlm
        if key.upper() not in tlm.dtype.names:
            raise KeyError(f'Parameter: {key.upper()} not found'
                           f' in {tlm.dtype.names}')

        raw_data = np.array([x[key.upper()] for x in tlm])
        return convert_hk(key.upper(), raw_data)

    @staticmethod
    def units(key: str) -> str:
        """Obtain units of converted telemetry parameter.

        Parameters
        ----------
        key :  str
           Name of telemetry parameter

        Returns
        -------
        str
        """
        return UNITS_DICT.get(key, '1')
