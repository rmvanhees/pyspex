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
import netCDF4 as nc4
import numpy as np

from .hkt_io import HKTio, read_hkt_nav, write_hkt_nav
from .lib.leap_sec import get_leap_seconds
from .lib.tlm_utils import convert_hk
from .lib.tmtc_def import tmtc_dtype
from .lv0_io import (ap_id, grouping_flag, packet_length,
                     read_lv0_data, sequence)
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
    ii = 0
    for buf in ccsds_hk:
        hdr[ii] = buf['hdr']
        if ap_id(hdr[ii]) != 0x320:
            continue

        tlm[ii] = buf['hk']
        tstamp.append(epoch + datetime.timedelta(
            seconds=int(buf['hdr']['tai_sec']),
            microseconds=subsec2musec(buf['hdr']['sub_sec'])))
        ii += 1

    return {'hdr': hdr[:len(tstamp)],
            'tlm': tlm[:len(tstamp)],
            'tstamp': np.array(tstamp)}


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
            img: tuple[np.ndarray] = (buf['frame'][0],)
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


def add_hkt_navigation(l1a_file: Path, hkt_list: list[Path]):
    """add PACE navigation information from PACE_HKT products.

    Parameters
    ----------
    l1a_file :  Path
       name of an existing L1A product.
    hkt_list :  list[Path]
       listing of files from which the navigation data has to be read
    """
    hkt_nav = read_hkt_nav(hkt_list)
    # select HKT data collocated with Science data
    # - issue a warning if selection is empty
    write_hkt_nav(l1a_file, hkt_nav)


def add_proc_conf(l1a_file: Path, yaml_conf: Path):
    """Add dataset 'processor_configuration' to an existing L1A product.

    Parameters
    ----------
    l1a_file :  Path
       name of an existing L1A product.
    yaml_conf :  Path
       name of the YAML file with the processor settings
    """
    # pylint: disable=no-member
    with nc4.Dataset(l1a_file, 'r+') as fid:
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
        if self._sci is None:
            return None

        if self._selection is None:
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

    @property
    def reference_date(self) -> datetime.date:
        """Return date of reference day (tzone aware)."""
        tstamp = self._sci['tstamp']['dt'][0] \
            if 'tstamp' in self._sci else self._hk['tstamp'][0]
        return datetime.datetime.combine(
                tstamp.date(), datetime.time(0), tstamp.tzinfo)

    @property
    def time_coverage_start(self) -> str:
        """Return a string for the time_coverage_start."""
        tstamp = self._sci['tstamp']['dt'][0] \
            if 'tstamp' in self._sci else self._hk['tstamp'][0]
        return tstamp.isoformat(timespec='milliseconds')

    @property
    def time_coverage_end(self) -> str:
        """Return a string for the time_coverage_end."""
        tstamp = self._sci['tstamp']['dt'][-1] \
            if 'tstamp' in self._sci else self._hk['tstamp'][-1]
        return tstamp.isoformat(timespec='milliseconds')

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

        if 'REG_FULL_FRAME' not in self.sci_tlm.dtype.names:
            print('[WARNING]: can not determine binning table identifier')
            return np.full(len(self.sci_tlm), -1, dtype='i1')

        full_frame = np.unique(self.sci_tlm['REG_FULL_FRAME'])
        if len(full_frame) > 1:
            print('[WARNING]: value of REG_FULL_FRAME not unique')
        full_frame = self.sci_tlm['REG_FULL_FRAME'][-1]

        cmv_outputmode = np.unique(self.sci_tlm['REG_CMV_OUTPUTMODE'])
        if len(cmv_outputmode) > 1:
            print('[WARNING]: value of REG_CMV_OUTPUTMODE not unique')
        cmv_outputmode = self.sci_tlm['REG_CMV_OUTPUTMODE'][-1]

        if full_frame == 1:
            if cmv_outputmode != 3:
                raise KeyError('Diagnostic mode with REG_CMV_OUTPMODE != 3')
            return np.zeros(len(self.sci_tlm), dtype='i1')

        if full_frame == 2:
            if cmv_outputmode != 1:
                raise KeyError('Science mode with REG_CMV_OUTPUTMODE != 1')
            bin_tbl_start = self.sci_tlm['REG_BINNING_TABLE_START']
            indx0 = (self.sci_tlm['REG_FULL_FRAME'] != 2).nonzero()[0]
            if indx0.size > 0:
                indx2 = (self.sci_tlm['REG_FULL_FRAME'] == 2).nonzero()[0]
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
                # pylint: disable=no-member
                ref_date = dset.attrs['units'].decode()[14:] + 'Z'
                epoch = datetime.datetime.fromisoformat(ref_date)
                self._hk['tstamp'] = []
                for sec in dset[:]:
                    self._hk['tstamp'].append(
                        epoch + datetime.timedelta(seconds=sec))

    def set_selection(self, mode: str):
        """Obtain image and housekeeping dimensions"""
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
            self._selection = {
                'sci_mask': np.full(True, len(self.sci_hdr)),
                'hk_mask': np.full(True, len(self.hk_hdr)),
                'dims': {
                    'number_of_images': len(self.hk_hdr),
                    'samples_per_image': np.max(
                        [len(x) for x in self.images]),
                    'hk_packets': len(self.hk_hdr)}
            }

    def gen_l1a(self, config: dataclass.Dataclass, mode: str):
        """Generate a SPEXone Level-1A product"""
        self.set_selection(mode)
        if (self._selection is None
            or self._selection['dims']['number_of_images'] == 0):
            return

        prod_mode = 'all' if config.eclipse is None else mode
        prod_name = get_l1a_name(config, prod_mode, self.sci_tstamp['dt'][0])

        with L1Aio(config.outdir / prod_name,
                   self.reference_date,
                   self._selection['dims'],
                   compression=config.compression) as l1a:
            if self.hk_tlm is None:
                l1a.set_attr('icu_sw_version',
                             f'0x{self.hk_tlm["ICUSWVER"][0]:x}')
            l1a.fill_global_attrs(inflight=config.l0_format != 'raw')
            l1a.set_attr('time_coverage_start', self.time_coverage_start)
            l1a.set_attr('time_coverage_end', self.time_coverage_end)
            l1a.set_attr('input_files', [x.name for x in config.l0_list])

            self._fill_engineering(l1a)
            self._fill_science(l1a)
            self._fill_image_attrs(l1a, config.l0_format)

        # add PACE navigation information from HKT products
        if config.hkt_list:
            add_hkt_navigation(config.outdir / prod_name, config.hkt_list)

        # add processor_configuration
        if config.yaml_fl:
            add_proc_conf(config.outdir / prod_name, config.yaml_fl)

    def _fill_engineering(self, l1a):
        """Fill datasets in group '/engineering_data'."""
        if self.hk_tlm is None:
            return
        l1a.set_dset('/engineering_data/NomHK_telemetry', self.hk_tlm)
        l1a.set_dset('/engineering_data/HK_tlm_time',
                     [(x - self.reference_date).total_seconds()
                      for x in self.hk_tstamp])
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

        l1a.set_dset('/science_data/detector_images', self.images)
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
        l1a.set_dset('/image_attributes/image_time',
                     [(x - self.reference_date).total_seconds()
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
            tlm = self.sci_tlm \
                if key.upper() in self.sci_tlm.dtype.names else self.hk_tlm
        if key.upper() not in tlm.dtype.names:
            raise KeyError(f'Parameter: {key.upper()} not found'
                           f' in {tlm.dtype.names}')

        raw_data = np.array([x[key.upper()] for x in tlm])
        return convert_hk(key.upper(), raw_data)
