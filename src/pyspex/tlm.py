#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""`SPXtlm` can read telemetry house-keeping data from SPEXone."""
from __future__ import annotations

__all__ = ['SPXtlm']

import datetime as dt
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

# pylint: disable=no-name-in-module
from netCDF4 import Dataset

from .hkt_io import HKTio, check_coverage_nav, read_hkt_nav
from .l1a_io import L1Aio
from .lib.ccsds_hdr import CCSDShdr
from .lib.leap_sec import get_leap_seconds
from .lib.tlm_utils import UNITS_DICT, convert_hk
from .lib.tmtc_def import tmtc_dtype
from .lv0_lib import dump_hkt, dump_science, read_lv0_data
from .version import pyspex_version

if TYPE_CHECKING:
    from dataclasses import dataclass

# - global parameters -----------------------
module_logger = logging.getLogger('pyspex.tlm')

TSTAMP_MIN = 1561939200           # 2019-07-01T00:00:00+00:00
TSTAMP_TYPE = np.dtype(
    [('tai_sec', int), ('sub_sec', int), ('dt', 'O')])

DET_CONSTS = {
    'dimRow': 2048,
    'dimColumn': 2048,
    'dimFullFrame': 2048 * 2048,
    'DEM_frequency': 10,            # [MHz]
    'FTI_science': 1000 / 15,       # [ms]
    'FTI_diagnostic': 240.,         # [ms]
    'FTI_margin': 212.4,            # [ms]
    'overheadTime': 0.4644,         # [ms]
    'FOT_length': 20
}
FULLFRAME_BYTES = 2 * DET_CONSTS['dimFullFrame']


# - helper functions ------------------------
def subsec2musec(sub_sec: int) -> int:
    """Return subsec as microseconds."""
    return 100 * int(sub_sec / 65536 * 10000)


def __exposure_time__(science: np.ndarray) -> np.ndarray:
    """Return exposure time [ms]."""
    return 129e-4 * (0.43 * science['DET_FOTLEN'] + science['DET_EXPTIME'])


def __frame_period__(science: np.ndarray) -> np.ndarray:
    """Return frame period of detector measurement [ms]."""
    n_coad = science['REG_NCOADDFRAMES']
    # binning mode
    if science['REG_FULL_FRAME'] == 2:
        return np.full(len(science), n_coad * DET_CONSTS['FTI_science'])

    # full-frame mode
    return n_coad * np.clip(DET_CONSTS['FTI_margin']
                            + DET_CONSTS['overheadTime']
                            + __exposure_time__(science),
                            a_min=DET_CONSTS['FTI_diagnostic'], a_max=None)


def __readout_offset__(science: np.ndarray) -> float:
    """Return offset wrt start-of-integration [ms]."""
    n_coad = science['REG_NCOADDFRAMES']
    n_frm = n_coad + 3 if science['IMRLEN'] == FULLFRAME_BYTES \
        else 2 * n_coad + 2

    return n_frm * __frame_period__(science)


def __binning_table__(science: np.ndarray) -> np.ndarray:
    """Return binning table identifier (zero for full-frame images)."""
    bin_tbl = np.zeros(len(science), dtype='i1')
    _mm = science['IMRLEN'] == FULLFRAME_BYTES
    if np.sum(_mm) == len(science):
        return bin_tbl

    bin_tbl_start = science['REG_BINNING_TABLE_START']
    bin_tbl[~_mm] = 1 + (bin_tbl_start[~_mm] - 0x80000000) // 0x400000
    return bin_tbl


def __digital_offset__(science: np.ndarray) -> np.ndarray:
    """Return digital offset including ADC offset [count]."""
    buff = science['DET_OFFSET'].astype('i4')
    buff[buff >= 8192] -= 16384

    return buff + 70


def extract_l0_hk(ccsds_hk: tuple, epoch: dt.datetime) -> dict | None:
    """Return dictionary with NomHk telemetry data."""
    if not ccsds_hk:
        return None

    hdr = np.empty(len(ccsds_hk),
                   dtype=ccsds_hk[0]['hdr'].dtype)
    tlm = np.empty(len(ccsds_hk), dtype=tmtc_dtype(0x320))
    tstamp = []
    ii = 0
    for buf in ccsds_hk:
        hdr[ii] = buf['hdr']
        ccsds_hdr = CCSDShdr(buf['hdr'])
        if ccsds_hdr.apid != 0x320 or buf['hdr']['tai_sec'] < len(ccsds_hk):
            continue

        tlm[ii] = buf['hk']
        tstamp.append(epoch + dt.timedelta(
            seconds=int(buf['hdr']['tai_sec']),
            microseconds=subsec2musec(buf['hdr']['sub_sec'])))
        ii += 1

    # These values are originally stored in little-endian, but
    # Numpy does not accepts a mix of little & big-endian values
    # in a structured array.
    tlm['HTR1_CALCPVAL'][:] = tlm['HTR1_CALCPVAL'].byteswap()
    tlm['HTR2_CALCPVAL'][:] = tlm['HTR2_CALCPVAL'].byteswap()
    tlm['HTR3_CALCPVAL'][:] = tlm['HTR3_CALCPVAL'].byteswap()
    tlm['HTR4_CALCPVAL'][:] = tlm['HTR4_CALCPVAL'].byteswap()
    tlm['HTR1_CALCIVAL'][:] = tlm['HTR1_CALCIVAL'].byteswap()
    tlm['HTR2_CALCIVAL'][:] = tlm['HTR2_CALCIVAL'].byteswap()
    tlm['HTR3_CALCIVAL'][:] = tlm['HTR3_CALCIVAL'].byteswap()
    tlm['HTR4_CALCIVAL'][:] = tlm['HTR4_CALCIVAL'].byteswap()

    return {'hdr': hdr[:ii],
            'tlm': tlm[:ii],
            'tstamp': np.array(tstamp)}


def extract_l0_sci(ccsds_sci: tuple, epoch: dt.datetime) -> dict | None:
    """Return dictionary with Science telemetry data."""
    if not ccsds_sci:
        return None

    n_frames = 0
    hdr_dtype = None
    hk_dtype = None
    found_start_first = False
    for buf in ccsds_sci:
        ccsds_hdr = CCSDShdr(buf['hdr'][0])
        if ccsds_hdr.grouping_flag == 1:
            found_start_first = True
            if n_frames == 0:
                hdr_dtype = buf['hdr'].dtype
                hk_dtype = buf['hk'].dtype
                continue

        if not found_start_first:
            continue

        if ccsds_hdr.grouping_flag == 2:
            found_start_first = False
            n_frames += 1

    # do we have any detector images?
    if n_frames == 0:
        module_logger.warning('no valid Science package found')
        return None

    # allocate memory
    hdr_arr = np.empty(n_frames, dtype=hdr_dtype)
    tlm_arr = np.empty(n_frames, dtype=hk_dtype)
    tstamp = np.empty(n_frames, dtype=TSTAMP_TYPE)
    images = ()

    # extract data from ccsds_sci
    ii = 0
    img = None
    found_start_first = False
    for buf in ccsds_sci:
        ccsds_hdr = CCSDShdr(buf['hdr'][0])
        if ccsds_hdr.grouping_flag == 1:
            found_start_first = True
            hdr_arr[ii] = buf['hdr']
            tlm_arr[ii] = buf['hk']
            tstamp[ii] = (buf['icu_tm']['tai_sec'][0],
                          buf['icu_tm']['sub_sec'][0],
                          epoch + dt.timedelta(
                              seconds=int(buf['icu_tm']['tai_sec'][0]),
                              microseconds=subsec2musec(
                                  buf['icu_tm']['sub_sec'][0])))
            img = (buf['frame'][0],)
            continue

        if not found_start_first:
            continue

        if ccsds_hdr.grouping_flag == 0:
            img += (buf['frame'][0],)
        elif ccsds_hdr.grouping_flag == 2:
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


def add_hkt_navigation(l1a_file: Path, hkt_list: list[Path]) -> int:
    """Add PACE navigation information from PACE_HKT products.

    Parameters
    ----------
    l1a_file :  Path
       name of an existing L1A product.
    hkt_list :  list[Path]
       listing of files from which the navigation data has to be read
    """
    # read PACE navigation data from HKT files.
    xds_nav = read_hkt_nav(hkt_list)
    # add PACE navigation data to existing Level-1A product.
    xds_nav.to_netcdf(l1a_file, group='navigation_data', mode='a')
    # check time coverage of navigation data.
    return check_coverage_nav(l1a_file, xds_nav)


def add_proc_conf(l1a_file: Path, yaml_conf: Path) -> None:
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
        dset.markup_language = 'YAML'
        dset[0] = ''.join(
            [s for s in yaml_conf.open(encoding='ascii').readlines()
             if not (s == '\n' or s.startswith('#'))])


# - class SPXtlm ----------------------------
class SPXtlm:
    """Access/convert parameters of SPEXone Science telemetry data.

    Notes
    -----
    This class has the following methods::

     - set_coverage(coverage: tuple[datetime, datetime] | None) -> None
     - hk_hdr() -> np.ndarray | None
     - hk_tlm() -> np.ndarray | None
     - hk_tstamp() -> np.ndarray | None
     - sci_hdr() -> np.ndarray | None
     - sci_tlm() -> np.ndarray | None
     - sci_tstamp() -> np.ndarray | None
     - images() -> tuple | None
     - reference_date() -> datetime
     - time_coverage_start() -> datetime
     - time_coverage_end() -> datetime
     - from_hkt(flnames: Path | list[Path], *,
                instrument: str | None = None, dump: bool = False) -> None
     - from_lv0(flnames: Path | list[Path], *,
                file_format: str, tlm_type: str | None = None,
                debug: bool = False, dump: bool = False) -> None
     - from_l1a(flname: Path, *, tlm_type: str | None = None) -> None
     - set_selection(mode: str) -> None
     - gen_l1a(config: dataclass, mode: str) -> None
     - convert(key: str, tm_type: str = 'both') -> np.ndarray
     - units(key: str) -> str
    """

    def __init__(self: SPXtlm) -> None:
        """Initialize class SPXtlm."""
        self.logger = logging.getLogger(__name__)
        self.file_list: list | None = None
        self._coverage: tuple[dt.datetime, dt.datetime] | None = None
        self._hk = None
        self._sci = None
        self._selection = None

    def set_coverage(self: SPXtlm,
                     coverage: tuple[dt.datetime, dt.datetime] | None) -> None:
        """Store or update the class attribute `coverage`."""
        if coverage is None:
            self._coverage = None
        elif self._coverage is None:
            self._coverage = coverage
        else:
            self._coverage = (min(self._coverage[0], coverage[0]),
                              max(self._coverage[1], coverage[1]))

    @property
    def hk_hdr(self: SPXtlm) -> np.ndarray | None:
        """Return CCSDS header data of telemetry packages @1Hz."""
        if self._hk is None:
            return None

        if self._selection is None or self._selection:
            return self._hk['hdr']
        return self._hk['hdr'][self._selection['hk_mask']]

    @property
    def hk_tlm(self: SPXtlm) -> np.ndarray | None:
        """Return telemetry packages @1Hz."""
        if self._hk is None:
            return None

        if self._selection is None:
            return self._hk['tlm']
        return self._hk['tlm'][self._selection['hk_mask']]

    @property
    def hk_tstamp(self: SPXtlm) -> np.ndarray | None:
        """Return timestamps of telemetry packages @1Hz."""
        if self._hk is None:
            return None

        if self._selection is None:
            return self._hk['tstamp']
        return self._hk['tstamp'][self._selection['hk_mask']]

    @property
    def sci_hdr(self: SPXtlm) -> np.ndarray | None:
        """Return CCSDS header data of Science telemetry packages."""
        if self._sci is None:
            return None

        if self._selection is None:
            return self._sci['hdr']
        return self._sci['hdr'][self._selection['sci_mask']]

    @property
    def sci_tlm(self: SPXtlm) -> np.ndarray | None:
        """Return Science telemetry packages."""
        if self._sci is None:
            return None

        if self._selection is None:
            return self._sci['tlm']
        return self._sci['tlm'][self._selection['sci_mask']]

    @property
    def sci_tstamp(self: SPXtlm) -> np.ndarray | None:
        """Return timestamps of Science telemetry packages."""
        if self._sci is None:
            return None

        if self._selection is None:
            return self._sci['tstamp']
        return self._sci['tstamp'][self._selection['sci_mask']]

    @property
    def images(self: SPXtlm) -> tuple[np.ndarray, ...] | None:
        """Return image-frames of Science telemetry packages."""
        if self._sci is None or 'images' not in self._sci:
            return None

        if self._selection is None:
            return self._sci['images']

        images = ()
        for ii, img in enumerate(self._sci['images']):
            if self._selection['sci_mask'][ii]:
                images += (img,)
        return images

    def __get_valid_tstamps(self: SPXtlm) -> np.ndarray | None:
        """Return valid timestamps from Science or NomHk packages."""
        if self.sci_tstamp is None \
                or np.all(self.sci_tstamp['tai_sec'] < TSTAMP_MIN):
            indx = self.hk_tstamp > dt.datetime(
                2020, 1, 1, 1, tzinfo=dt.timezone.utc)
            return self.hk_tstamp[indx] if indx.size > 0 else None

        indx = np.where(self.sci_tstamp['tai_sec'] > TSTAMP_MIN)[0]
        return self.sci_tstamp['dt'][indx] if indx.size > 0 else None

    @property
    def reference_date(self: SPXtlm) -> dt.datetime:
        """Return date of reference day (tzone aware)."""
        tstamp = self.__get_valid_tstamps()
        if tstamp is None:
            raise ValueError('no valid timestamps found')

        return dt.datetime.combine(
                tstamp[0].date(), dt.time(0), tstamp[0].tzinfo)

    @property
    def time_coverage_start(self: SPXtlm) -> dt.datetime:
        """Return a string for the time_coverage_start."""
        if self._coverage is not None:
            return self._coverage[0]
        tstamp = self.__get_valid_tstamps()
        if tstamp is None:
            raise ValueError('no valid timestamps found')

        return tstamp[0]

    @property
    def time_coverage_end(self: SPXtlm) -> dt.datetime:
        """Return a string for the time_coverage_end."""
        if self._coverage is not None:
            return self._coverage[1]
        tstamp = self.__get_valid_tstamps()
        if tstamp is None:
            raise ValueError('no valid timestamps found')

        frame_period = __frame_period__(self.sci_tlm[-1:])[0] \
            if self.sci_tlm is not None else  1.
        return tstamp[-1] + dt.timedelta(milliseconds=frame_period)

    def from_hkt(self: SPXtlm, flnames: Path | list[Path], *,
                 instrument: str | None = None, dump: bool = False) -> None:
        """Read telemetry data from a PACE HKT product.

        Parameters
        ----------
        flnames :  Path | list[Path]
           list of PACE_HKT filenames (netCDF4 format)
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
            hkt = HKTio(name)
            self.set_coverage(hkt.coverage())
            ccsds_hk += hkt.housekeeping(instrument)

        if not ccsds_hk:
            return

        if dump:
            dump_hkt(flnames[0].stem + '_hkt.dump', ccsds_hk)

        epoch = dt.datetime(1958, 1, 1, tzinfo=dt.timezone.utc)
        ii = len(ccsds_hk) // 2
        leap_sec = get_leap_seconds(ccsds_hk[ii]['hdr']['tai_sec'][0])
        epoch -= dt.timedelta(seconds=leap_sec)
        self._hk = extract_l0_hk(ccsds_hk, epoch)

    def from_lv0(self: SPXtlm, flnames: Path | list[Path], *,
                 file_format: str, tlm_type: str | None = None,
                 debug: bool = False, dump: bool = False) -> None:
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
        if file_format not in ['raw', 'st3', 'dsb']:
            raise KeyError("file_format not in ['raw', 'st3', 'dsb']")

        self.file_list = flnames
        self._hk = None
        self._sci = None
        ccsds_sci, ccsds_hk = read_lv0_data(flnames, file_format, debug=debug)
        if dump:
            dump_hkt(flnames[0].stem + '_hkt.dump', ccsds_hk)
            dump_science(flnames[0].stem + '_sci.dump', ccsds_sci)
        if debug or dump:
            return

        # set epoch
        if file_format == 'dsb':
            epoch = dt.datetime(1958, 1, 1,
                                      tzinfo=dt.timezone.utc)
            ii = len(ccsds_hk) // 2
            leap_sec = get_leap_seconds(ccsds_hk[ii]['hdr']['tai_sec'][0])
            epoch -= dt.timedelta(seconds=leap_sec)
        else:
            epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

        # collect Science telemetry data
        if tlm_type != 'hk':
            self._sci = extract_l0_sci(ccsds_sci, epoch)
        del ccsds_sci

        # collected NomHk telemetry data
        if tlm_type != 'sci':
            self._hk = extract_l0_hk(ccsds_hk, epoch)

    def from_l1a(self: SPXtlm, flname: Path, *,
                 tlm_type: str | None = None) -> None:
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
                dset = fid['/image_attributes/icu_time_sec']
                seconds = dset[:]
                try:
                    # pylint: disable=no-member
                    _ = dset.attrs['units'].index(b'1958')
                except ValueError:
                    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
                else:
                    epoch = dt.datetime(1958, 1, 1, tzinfo=dt.timezone.utc)
                    epoch -= dt.timedelta(seconds=get_leap_seconds(seconds[0]))

                subsec = fid['/image_attributes/icu_time_subsec'][:]
                self._sci = {
                    'tlm': fid['/science_data/detector_telemetry'][:],
                    'images': fid['/science_data/detector_images'][:],
                    'tstamp': np.empty(len(seconds), dtype=TSTAMP_TYPE)
                }
                self._sci['tstamp']['tai_sec'] = seconds
                self._sci['tstamp']['sub_sec'] = subsec
                _dt = []
                for ii, sec in enumerate(seconds):
                    _dt.append(epoch + dt.timedelta(
                        seconds=int(sec),
                        milliseconds=-__readout_offset__(self._sci['tlm'][0]),
                        microseconds=subsec2musec(subsec[ii])))
                self._sci['tstamp']['dt'] = _dt

            if tlm_type != 'sci':
                self._hk = {
                    'tlm': fid['/engineering_data/NomHK_telemetry'][:],
                    'tstamp': []
                }
                dset = fid['/engineering_data/HK_tlm_time']
                # pylint: disable=no-member
                ref_date = dset.attrs['units'].decode()[14:] + '+00:00'
                epoch = dt.datetime.fromisoformat(ref_date)
                for sec in dset[:]:
                    self._hk['tstamp'].append(epoch + dt.timedelta(seconds=sec))

    def set_selection(self: SPXtlm, mode: str) -> None:
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
            self.logger.debug('unique Diagnostic MPS: %s', mps_list)
            hk_mask = np.in1d(self.hk_tlm['MPS_ID'], mps_list)

            self._selection = {
                'sci_mask': sci_mask,
                'hk_mask': hk_mask,
                'dims': {
                    'number_of_images': np.sum(sci_mask),
                    'samples_per_image': DET_CONSTS['dimFullFrame'],
                    'hk_packets': np.sum(hk_mask)}
            }
            return

        if mode == 'binned':
            sci_mask = [] if self.sci_tlm is None else \
                self.sci_tlm['IMRLEN'] < FULLFRAME_BYTES
            if np.sum(sci_mask) == 0:
                return

            mps_list = np.unique(self.sci_tlm['MPS_ID'][sci_mask])
            self.logger.debug('unique Science MPS: %s', mps_list)
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
                    'samples_per_image': DET_CONSTS['dimRow']
                    if nr_sci == 0 else np.max([len(x) for x in self.images]),
                    'hk_packets': nr_hk}
            }

    def l1a_file(self: SPXtlm, config: dataclass, mode: str) -> Path:
        """Return filename of Level-1A product.

        Parameters
        ----------
        config :  dataclass
           Settings for the L0->l1A processing.
        mode :  {'all', 'full', 'binned'}
           Select Science packages with full-frame image or binned images

        Returns
        -------
        Path
           Filename of Level-1A product.

        Notes
        -----
        === Inflight ===
        L1A file name format, following the NASA ... naming convention:
           PACE_SPEXONE[_TTT].YYYYMMDDTHHMMSS.L1A[.Vnn].nc
        where
           TTT is an optional data type (e.g., for the calibration data files)
           YYYYMMDDTHHMMSS is time stamp of the first image in the file
           Vnn file-version number (omitted when nn=1)
        for example (file-version=1):
           [Science Product] PACE_SPEXONE.20230115T123456.L1A.nc
           [Calibration Product] PACE_SPEXONE_CAL.20230115T123456.L1A.nc
           [Dark science Product] PACE_SPEXONE_DARK.20230115T123456.L1A.nc

        === OCAL ===
        L1A file name format:
           SPX1_OCAL_<msm_id>[_YYYYMMDDTHHMMSS]_L1A_vvvvvvv.nc
        where
           msm_id is the measurement identifier
           YYYYMMDDTHHMMSS is time stamp of the first image in the file
           vvvvvvv is the git-hash string of the pyspex repository
        """
        if config.outfile:
            return config.outdir / config.outfile

        if config.l0_format != 'raw':
            if config.eclipse is None:
                subtype = '_OCAL'
            elif not config.eclipse:
                subtype = ''
            else:
                subtype = '_CAL' if mode == 'full' else '_DARK'

            prod_ver = '' if config.file_version == 1 \
                else f'.V{config.file_version:02d}'

            return config.outdir / (
                f'PACE_SPEXONE{subtype}'
                f'.{self.time_coverage_start.strftime("%Y%m%dT%H%M%S"):15s}'
                f'.L1A{prod_ver}.nc')

        # OCAL product name
        # determine measurement identifier
        msm_id = config.l0_list[0].stem
        try:
            new_date = dt.datetime.strptime(
                msm_id[-22:], '%y-%j-%H:%M:%S.%f').strftime('%Y%m%dT%H%M%S.%f')
        except ValueError:
            pass
        else:
            msm_id = msm_id[:-22] + new_date

        return (config.outdir /
                f'SPX1_OCAL_{msm_id}_L1A_{pyspex_version(githash=True)}.nc')

    def gen_l1a(self: SPXtlm, config: dataclass, mode: str) -> None:
        """Generate a SPEXone Level-1A product."""
        self.set_selection(mode)
        if self._selection is None:
            return

        l1a_file = self.l1a_file(config, mode)
        ref_date = self.reference_date
        with L1Aio(l1a_file, ref_date, self._selection['dims'],
                   compression=config.compression) as l1a:
            l1a.fill_global_attrs(inflight=config.l0_format != 'raw')
            if self.hk_tlm is not None:
                l1a.set_attr('icu_sw_version',
                             f'0x{self.hk_tlm["ICUSWVER"][0]:x}')
            l1a.set_attr('time_coverage_start',
                         self.time_coverage_start.isoformat(
                             timespec='milliseconds'))
            l1a.set_attr('time_coverage_end',
                         self.time_coverage_end.isoformat(
                             timespec='milliseconds'))
            l1a.set_attr('input_files', [x.name for x in config.l0_list])
            self.logger.debug('(1) initialized Level-1A product')

            self._fill_engineering(l1a)
            self.logger.debug('(2) added engineering data')
            self._fill_science(l1a)
            self.logger.debug('(3) added science data')
            self._fill_image_attrs(l1a, config.l0_format)
            self.logger.debug('(4) added image attributes')

        # add processor_configuration
        if config.yaml_fl:
            add_proc_conf(l1a_file, config.yaml_fl)

        # add PACE navigation information from HKT products
        if config.hkt_list:
            status_ok = add_hkt_navigation(l1a_file, config.hkt_list)
            self.logger.debug('(5) added PACE navigation data')

            if not status_ok:
                raise UserWarning(
                    'time-coverage of navigation data is too short')

        self.logger.info('successfully generated: %s', l1a_file.name)

    def _fill_engineering(self: SPXtlm, l1a: h5py.File) -> None:
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

    def _fill_science(self: SPXtlm, l1a: h5py.File) -> None:
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

    def _fill_image_attrs(self: SPXtlm, l1a: h5py.File,
                          lv0_format: str) -> None:
        """Fill datasets in group '/image_attributes'."""
        if self.sci_tlm is None:
            return

        l1a.set_dset('/image_attributes/icu_time_sec',
                     self.sci_tstamp['tai_sec'])
        # modify attribute units for non-DSB products
        if lv0_format != 'dsb':
            # timestamp of 2020-01-01T00:00:00+00:00
            l1a.set_attr('valid_min', np.uint32(1577836800),
                         ds_name='/image_attributes/icu_time_sec')
            # timestamp of 2024-01-01T00:00:00+00:00
            l1a.set_attr('valid_max', np.uint32(1704067200),
                         ds_name='/image_attributes/icu_time_sec')
            l1a.set_attr('units', 'seconds since 1970-01-01 00:00:00',
                         ds_name='/image_attributes/icu_time_sec')
        l1a.set_dset('/image_attributes/icu_time_subsec',
                     self.sci_tstamp['sub_sec'])
        ref_date = self.reference_date
        l1a.set_dset('/image_attributes/image_time',
                     [(x - ref_date).total_seconds()
                      for x in self.sci_tstamp['dt']])
        l1a.set_dset('/image_attributes/image_ID',
                     np.bitwise_and(self.sci_hdr['sequence'], 0x3fff))
        l1a.set_dset('/image_attributes/binning_table',
                     __binning_table__(self.sci_tlm))
        l1a.set_dset('/image_attributes/digital_offset',
                     __digital_offset__(self.sci_tlm))
        l1a.set_dset('/image_attributes/exposure_time',
                     __exposure_time__(self.sci_tlm) / 1000)
        l1a.set_dset('/image_attributes/nr_coadditions',
                     self.sci_tlm['REG_NCOADDFRAMES'])

    def convert(self: SPXtlm, key: str, tm_type: str = 'both') -> np.ndarray:
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
