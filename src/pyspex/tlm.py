#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""`SPXtlm`, class to read/access PACE/SPEXone telemetry data."""
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
from .lib import pyspex_version
from .lib.ccsds_hdr import CCSDShdr
from .lib.leap_sec import get_leap_seconds
from .lib.tlm_utils import convert_hk
from .lib.tmtc_def import tmtc_dtype
from .lv0_lib import dump_hkt, dump_science, read_lv0_data

if TYPE_CHECKING:
    from dataclasses import dataclass

    import numpy.typing as npt

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

def mask2slice(mask: npt.NDArray[bool]) -> (None | slice | tuple
                                            | npt.NDArray[bool]):
    """Try to slice (faster), instead of boolean indexing (slow)."""
    if np.all(~mask):
        return None
    if np.all(mask):
        return np.s_[:]                       # read everything

    indx = np.nonzero(mask)[0]
    if np.all(np.diff(indx) == 1):
        # perform start-stop indexing
        return np.s_[indx[0]:indx[-1]+1]

    # perform boolean indexing
    return mask



def add_hkt_navigation(l1a_file: Path, hkt_list: tuple[Path, ...]) -> int:
    """Add PACE navigation information from PACE_HKT products.

    Parameters
    ----------
    l1a_file :  Path
       name of an existing L1A product.
    hkt_list :  list[Path, ...]
       listing of files from which the navigation data has to be read
    """
    # read PACE navigation data from HKT files.
    xds_nav = read_hkt_nav(hkt_list)
    # add PACE navigation data to existing level-1A product.
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


# - class SCItlm ----------------------------
class HKtlm:
    """Class to handle SPEXone housekeeping telemetry packets."""

    def __init__(self: HKtlm) -> None:
        """Initialize HKtlm object."""
        self.hdr: np.ndarray | None = None
        self.tlm: np.ndarray | None = None
        self.tstamp: list[dt.datetime, ...] | list = []
        self.events: list[np.ndarray, ...] | list = []

    def init_attrs(self: HKtlm) -> None:
        """Initialize class attributes."""
        self.hdr = None
        self.tlm = None
        self.tstamp = []
        self.events = []

    def extract_l0_hk(self: HKtlm, ccsds_hk: tuple,
                      epoch: dt.datetime) -> None:
        """Extract data from SPEXone level-0 housekeeping telemetry packets.

        Parameters
        ----------
        ccsds_hk :  tuple[np.ndarray, ...]
           SPEXone level-0 housekeeping telemetry packets
        epoch :  dt.datetime
           Epoch of the telemetry packets (1958 or 1970)
        """
        self.init_attrs()
        if not ccsds_hk:
            return

        self.hdr = np.empty(len(ccsds_hk),
                            dtype=ccsds_hk[0]['hdr'].dtype)
        self.tlm = np.empty(len(ccsds_hk), dtype=tmtc_dtype(0x320))
        ii = 0
        for buf in ccsds_hk:
            ccsds_hdr = CCSDShdr(buf['hdr'][0])

            # Catch TcAccept, TcReject, TcExecute, TcFail and EventRp as events
            if ccsds_hdr.apid != 0x320 \
               or buf['hdr']['tai_sec'] < len(ccsds_hk):
                if 0x331 <= ccsds_hdr.apid <= 0x335:
                    self.events.append(buf)
                continue

            self.hdr[ii] = buf['hdr']
            self.tlm[ii] = buf['hk']
            self.tstamp.append(epoch + dt.timedelta(
                seconds=int(buf['hdr']['tai_sec'][0]),
                microseconds=subsec2musec(buf['hdr']['sub_sec'][0])))
            ii += 1

        self.hdr = self.hdr[:ii]
        self.tlm = self.tlm[:ii]

        # These values are originally stored in little-endian, but
        # Numpy does not accept a mix of little & big-endian values
        # in a structured array.
        self.tlm['HTR1_CALCPVAL'][:] = self.tlm['HTR1_CALCPVAL'].byteswap()
        self.tlm['HTR2_CALCPVAL'][:] = self.tlm['HTR2_CALCPVAL'].byteswap()
        self.tlm['HTR3_CALCPVAL'][:] = self.tlm['HTR3_CALCPVAL'].byteswap()
        self.tlm['HTR4_CALCPVAL'][:] = self.tlm['HTR4_CALCPVAL'].byteswap()
        self.tlm['HTR1_CALCIVAL'][:] = self.tlm['HTR1_CALCIVAL'].byteswap()
        self.tlm['HTR2_CALCIVAL'][:] = self.tlm['HTR2_CALCIVAL'].byteswap()
        self.tlm['HTR3_CALCIVAL'][:] = self.tlm['HTR3_CALCIVAL'].byteswap()
        self.tlm['HTR4_CALCIVAL'][:] = self.tlm['HTR4_CALCIVAL'].byteswap()

    def extract_l1a_hk(self: HKtlm,
                       fid: h5py.File, mps_id: int | None) -> None:
        """Extract data from SPEXone level-1a housekeeping telemetry packets.

        Parameters
        ----------
        fid :  h5py.File
           File pointer to a SPEXone level-1a product
        mps_id : int, optional
           Select data performed with MPS equals 'mps_id'
        """
        self.init_attrs()

        # pylint: disable=no-member
        dset = fid['/engineering_data/NomHK_telemetry']
        if mps_id is None:
            data_sel = np.s_[:]
        else:
            data_sel = mask2slice(dset.fields('MPS_ID')[:] == mps_id)
            if data_sel is None:
                return
        self.tlm = dset[data_sel]

        dset = fid['/engineering_data/HK_tlm_time']
        ref_date = dset.attrs['units'].decode()[14:] + '+00:00'
        epoch = dt.datetime.fromisoformat(ref_date)
        for sec in dset[data_sel]:
            self.tstamp.append(epoch + dt.timedelta(seconds=sec))

    def convert(self: HKtlm, key: str) -> np.ndarray:
        """Convert telemetry parameter to physical units.

        Parameters
        ----------
        key :  str
           Name of telemetry parameter

        Returns
        -------
        np.ndarray
        """
        if key.upper() not in self.tlm.dtype.names:
            raise KeyError(f'Parameter: {key.upper()} not found'
                           f' in {self.tlm.dtype.names}')

        raw_data = np.array([x[key.upper()] for x in self.tlm])
        return convert_hk(key.upper(), raw_data)


# - class SCItlm ----------------------------
class SCItlm:
    """Class to handle SPEXone Science-telemetry packets."""

    def __init__(self: SCItlm) -> None:
        """Initialize SCItlm object."""
        self.hdr: np.ndarray | None = None
        self.tlm: np.ndarray | None = None
        self.tstamp: np.ndarray | None = None
        self.images: tuple[np.ndarray, ...] | tuple[()] = ()

    def init_attrs(self: SCItlm) -> None:
        """Initialize class attributes."""
        self.hdr = None
        self.tlm = None
        self.tstamp = None
        self.images = ()

    def extract_l0_sci(self: SCItlm, ccsds_sci: tuple,
                       epoch: dt.datetime) -> None:
        """Extract SPEXone level-0 Science-telemetry data.

        Parameters
        ----------
        ccsds_sci :  tuple[np.ndarray, ...]
           SPEXone level-0 Science-telemetry packets
        epoch :  dt.datetime
           Epoch of the telemetry packets (1958 or 1970)
        """
        self.init_attrs()
        if not ccsds_sci:
            return

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

        # do we have any complete detector images (Note ccsds_sci not empty!)?
        if n_frames == 0:
            module_logger.warning('no valid Science package found')
            return

        # allocate memory
        self.hdr = np.empty(n_frames, dtype=hdr_dtype)
        self.tlm = np.empty(n_frames, dtype=hk_dtype)
        self.tstamp = np.empty(n_frames, dtype=TSTAMP_TYPE)

        # extract data from ccsds_sci
        ii = 0
        img = None
        found_start_first = False
        for buf in ccsds_sci:
            ccsds_hdr = CCSDShdr(buf['hdr'][0])
            if ccsds_hdr.grouping_flag == 1:
                found_start_first = True
                self.hdr[ii] = buf['hdr']
                self.tlm[ii] = buf['hk']
                self.tstamp[ii] = (buf['icu_tm']['tai_sec'][0],
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
                self.images += (np.concatenate(img),)
                ii += 1
                if ii == n_frames:
                    break

    def extract_l1a_sci(self: SCItlm, fid: h5py.File,
                        mps_id: int | None) -> None:
        """Extract data from SPEXone level-1a Science-telemetry packets.

        Parameters
        ----------
        fid :  h5py.File
           File pointer to a SPEXone level-1a product
        mps_id : int, optional
           Select data performed with MPS equals 'mps_id'
        """
        # pylint: disable=no-member
        self.init_attrs()

        # read science telemetry
        dset = fid['/science_data/detector_telemetry']
        if mps_id is None:
            data_sel = np.s_[:]
        else:
            data_sel = mask2slice(dset.fields('MPS_ID')[:] == mps_id)
            if data_sel is None:
                return
        self.tlm = fid['/science_data/detector_telemetry'][data_sel]

        # determine time-stamps
        dset = fid['/image_attributes/icu_time_sec']
        seconds = dset[data_sel]
        try:
            _ = dset.attrs['units'].index(b'1958')
        except ValueError:
            epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
        else:
            epoch = dt.datetime(1958, 1, 1, tzinfo=dt.timezone.utc)
            epoch -= dt.timedelta(seconds=get_leap_seconds(seconds[0]))
        subsec = fid['/image_attributes/icu_time_subsec'][data_sel]

        _dt = []
        for ii, sec in enumerate(seconds):
            msec_offs = self.readout_offset(ii)
            _dt.append(epoch + dt.timedelta(
                seconds=int(sec), milliseconds=-msec_offs,
                microseconds=subsec2musec(subsec[ii])))

        self.tstamp = np.empty(len(seconds), dtype=TSTAMP_TYPE)
        self.tstamp['tai_sec'] = seconds
        self.tstamp['sub_sec'] = subsec
        self.tstamp['dt'] = _dt

        # read image data
        self.images = fid['/science_data/detector_images'][data_sel, :]

    def exposure_time(self: SCItlm, indx: int | None = None) -> np.ndarray:
        """Return exposure time [ms]."""
        if indx is None:
            indx = np.s_[:]
        return 129e-4 * (0.43 * self.tlm['DET_FOTLEN'][indx]
                         + self.tlm['DET_EXPTIME'][indx])

    def frame_period(self: SCItlm, indx: int) -> float:
        """Return frame period of detector measurement [ms]."""
        n_coad = self.tlm['REG_NCOADDFRAMES'][indx]
        # binning mode
        if self.tlm['REG_FULL_FRAME'][indx] == 2:
            return float(n_coad * DET_CONSTS['FTI_science'])

        # full-frame mode
        return float(n_coad * np.clip(DET_CONSTS['FTI_margin']
                                      + DET_CONSTS['overheadTime']
                                      + self.exposure_time(indx),
                                      a_min=DET_CONSTS['FTI_diagnostic'],
                                      a_max=None))

    def readout_offset(self: SCItlm, indx: int) -> float:
        """Return offset wrt start-of-integration [ms]."""
        n_coad = self.tlm['REG_NCOADDFRAMES'][indx]
        n_frm = n_coad + 3 if self.tlm['IMRLEN'][indx] == FULLFRAME_BYTES \
            else 2 * n_coad + 2

        return n_frm * self.frame_period(indx)

    def binning_table(self: SCItlm) -> np.ndarray:
        """Return binning table identifier (zero for full-frame images)."""
        bin_tbl = np.zeros(len(self.tlm), dtype='i1')
        _mm = self.tlm['IMRLEN'] == FULLFRAME_BYTES
        if np.sum(_mm) == len(self.tlm):
            return bin_tbl

        bin_tbl_start = self.tlm['REG_BINNING_TABLE_START']
        bin_tbl[~_mm] = 1 + (bin_tbl_start[~_mm] - 0x80000000) // 0x400000
        return bin_tbl

    def digital_offset(self: SCItlm) -> np.ndarray:
        """Return digital offset including ADC offset [count]."""
        buff = self.tlm['DET_OFFSET'].astype('i4')
        buff[buff >= 8192] -= 16384

        return buff + 70

    def convert(self: SCItlm, key: str) -> np.ndarray:
        """Convert telemetry parameter to physical units.

        Parameters
        ----------
        key :  str
           Name of telemetry parameter

        Returns
        -------
        np.ndarray
        """
        if key.upper() not in self.tlm.dtype.names:
            raise KeyError(f'Parameter: {key.upper()} not found'
                           f' in {self.tlm.dtype.names}')

        raw_data = np.array([x[key.upper()] for x in self.tlm])
        return convert_hk(key.upper(), raw_data)


# - class SPXtlm ----------------------------
class SPXtlm:
    """Access/convert parameters of SPEXone Science telemetry data.

    Notes
    -----
    This class has the following methods::

     - set_coverage(coverage: tuple[datetime, datetime] | None) -> None
     - reference_date() -> datetime
     - time_coverage_start() -> datetime
     - time_coverage_end() -> datetime
     - from_hkt(flnames: Path | list[Path], *,
                instrument: str | None = None, dump: bool = False) -> None
     - from_lv0(flnames: Path | list[Path], *,
                file_format: str, tlm_type: str | None = None,
                debug: bool = False, dump: bool = False) -> None
     - from_l1a(flname: Path, *, tlm_type: str | None = None,
                mps_id: int | None = None) -> None
     - get_selection(mode: str) -> dict
     - gen_l1a(config: dataclass, mode: str) -> None


    Examples
    --------
    Read data from SPEXone level-0 products

    >>> from pyspex.tlm import SPXtlm
    >>> spx = SPXtlm()
    >>> spx.from_l0(list_l0_products, tlm_type='hk')
    # returns list of TcAccept, TcReject, TcExecute, TcFail and EventRp
    >>> spx.nomhk.events
    # return detector images
    >>> spx.from_l0(list_l0_products, tlm_type='sci')
    >>> spx.science.images

    Read data from a SPEXone level-1a product

    >>> from pyspex.tlm import SPXtlm
    >>> spx = SPXtlm()
    >>> spx.from_l1a(l1a_product, tlm_type='sci', mps_is=47)
    >>> spx.science.images

    Read data from a PACE-HKT product

    >>> from pyspex.tlm import SPXtlm
    >>> spx = SPXtlm()
    >>> spx.from_hkt(hkt_product)
    # returns list of TcAccept, TcReject, TcExecute, TcFail and EventRp
    >>> spx.nomhk.events
    """

    def __init__(self: SPXtlm) -> None:
        """Initialize SPXtlm object."""
        self.logger = logging.getLogger(__name__)
        self.file_list: list | None = None
        self._coverage: tuple[dt.datetime, dt.datetime] | None = None
        self.nomhk = HKtlm()
        self.science = SCItlm()

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
    def reference_date(self: SPXtlm) -> dt.datetime:
        """Return date of reference day (tzone aware)."""
        if self._coverage is None:
            raise ValueError('no valid timestamps found')

        return dt.datetime.combine(
                self._coverage[0].date(), dt.time(0), dt.timezone.utc)

    @property
    def time_coverage_start(self: SPXtlm) -> dt.datetime | None:
        """Return time_coverage_start."""
        if self._coverage is None:
            return None

        return self._coverage[0]

    @property
    def time_coverage_end(self: SPXtlm) -> dt.datetime | None:
        """Return time_coverage_end."""
        if self._coverage is None:
            return None

        return self._coverage[1]

    def from_hkt(self: SPXtlm, flnames: Path | list[Path], *,
                 instrument: str | None = None, dump: bool = False) -> None:
        """Read telemetry data from PACE HKT product(s).

        Parameters
        ----------
        flnames :  Path | list[Path]
           sorted list of PACE_HKT filenames (netCDF4 format)
        instrument :  {'spx', 'sc', 'oci', 'harp'}, default='spx'
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
        self.set_coverage(None)
        ccsds_hk: tuple[np.ndarray, ...] | tuple[()] = ()
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
        leap_sec = get_leap_seconds(float(ccsds_hk[ii]['hdr']['tai_sec'][0]))
        epoch -= dt.timedelta(seconds=leap_sec)
        self.nomhk.extract_l0_hk(ccsds_hk, epoch)

    def from_lv0(self: SPXtlm, flnames: Path | list[Path], *,
                 file_format: str, tlm_type: str | None = None,
                 debug: bool = False, dump: bool = False) -> None:
        """Read telemetry data from SPEXone level-0 product(s).

        Parameters
        ----------
        flnames :  Path | list[Path]
           sorted list of CCSDS filenames
        file_format : {'raw', 'st3', 'dsb'}
           type of CCSDS data
        tlm_type :  {'hk', 'sci', 'all'}, default='all'
           select type of telemetry packages
           Note that we allways read the complete level-0 products.
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
        ccsds_sci, ccsds_hk = read_lv0_data(flnames, file_format, debug=debug)
        if dump:
            dump_hkt(flnames[0].stem + '_hkt.dump', ccsds_hk)
            dump_science(flnames[0].stem + '_sci.dump', ccsds_sci)
        if debug or dump:
            return

        # set epoch
        if file_format == 'dsb':
            epoch = dt.datetime(1958, 1, 1, tzinfo=dt.timezone.utc)
            ii = len(ccsds_hk) // 2
            leap_sec = get_leap_seconds(ccsds_hk[ii]['hdr']['tai_sec'][0])
            epoch -= dt.timedelta(seconds=leap_sec)
        else:
            epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)

        # collect Science telemetry data
        tstamp = None
        self.set_coverage(None)
        if tlm_type != 'hk':
            self.science.extract_l0_sci(ccsds_sci, epoch)
            _mm = self.science.tstamp['tai_sec'] > TSTAMP_MIN
            if np.any(_mm):
                tstamp = self.science.tstamp['dt'][_mm]
                ii = int(np.nonzero(_mm)[0][-1])
                intg = dt.timedelta(milliseconds=self.science.frame_period(ii))
                self.set_coverage((tstamp[0], tstamp[-1] + intg))
        del ccsds_sci

        # collected NomHK telemetry data
        if tlm_type != 'sci':
            dt_min = dt.datetime(2020, 1, 1, 1, tzinfo=dt.timezone.utc)
            self.nomhk.extract_l0_hk(ccsds_hk, epoch)
            if tstamp is None:
                tstamp = [x for x in self.nomhk.tstamp if x > dt_min]
                self.set_coverage(
                    (tstamp[0], tstamp[-1] + dt.timedelta(seconds=1)))

    def from_l1a(self: SPXtlm, flname: Path, *,
                 tlm_type: str | None = None,
                 mps_id: int | None = None) -> None:
        """Read telemetry data from SPEXone level-1A product.

        Parameters
        ----------
        flname :  Path
           name of one SPEXone level-1A product
        tlm_type :  {'hk', 'sci', 'all'}, default='all'
           select type of telemetry packages
        mps_id :  int, optional
           select on MPS ID
        """
        if tlm_type is None:
            tlm_type = 'all'
        elif tlm_type not in ['hk', 'sci', 'all']:
            raise KeyError("tlm_type not in ['hk', 'sci', 'all']")

        tstamp = None
        self.file_list = [flname]
        self.set_coverage(None)
        with h5py.File(flname) as fid:
            # collect Science telemetry data
            if tlm_type != 'hk':
                self.science.extract_l1a_sci(fid, mps_id)
                _mm = self.science.tstamp['tai_sec'] > TSTAMP_MIN
                if np.any(_mm):
                    tstamp = self.science.tstamp['dt'][_mm]
                    ii = int(np.nonzero(_mm)[0][-1])
                    intg = dt.timedelta(
                        milliseconds=self.science.frame_period(ii))
                    self.set_coverage((tstamp[0], tstamp[-1] + intg))

            # collected NomHk telemetry data
            if tlm_type != 'sci':
                dt_min = dt.datetime(2020, 1, 1, 1, tzinfo=dt.timezone.utc)
                self.nomhk.extract_l1a_hk(fid, mps_id)
                if tstamp is None:
                    tstamp = [x for x in self.nomhk.tstamp if x > dt_min]
                    self.set_coverage(
                        (tstamp[0], tstamp[-1] + dt.timedelta(seconds=1)))

    def get_selection(self: SPXtlm, mode: str) -> dict | None:
        """Obtain image and housekeeping dimensions given data-mode.

        Parameters
        ----------
        mode :  {'all', 'binned', 'full'}
           Select Science packages with full-frame images or binned images

        Returns
        -------
        dict {'sci_mask': np.ndarray | [],
              'hk_mask': np.ndarray,
              'dims': {'number_of_images': int,
                       'samples_per_image': int,
                       'hk_packets': int}
        }
        """
        if mode == 'full':
            if self.science.tlm is None \
               or np.sum(self.science.tlm['IMRLEN'] == FULLFRAME_BYTES) == 0:
                return None

            sci_mask = self.science.tlm['IMRLEN'] == FULLFRAME_BYTES
            mps_list = np.unique(self.science.tlm['MPS_ID'][sci_mask])
            self.logger.debug('unique Diagnostic MPS: %s', mps_list)
            hk_mask = np.in1d(self.nomhk.tlm['MPS_ID'], mps_list)

            return {'sci_mask': sci_mask,
                    'hk_mask': hk_mask,
                    'dims': {
                        'number_of_images': np.sum(sci_mask),
                        'samples_per_image': DET_CONSTS['dimFullFrame'],
                        'hk_packets': np.sum(hk_mask)}
                    }

        if mode == 'binned':
            if self.science.tlm is None \
               or np.sum(self.science.tlm['IMRLEN'] < FULLFRAME_BYTES) == 0:
                return None

            sci_mask = self.science.tlm['IMRLEN'] < FULLFRAME_BYTES
            mps_list = np.unique(self.science.tlm['MPS_ID'][sci_mask])
            self.logger.debug('unique Science MPS: %s', mps_list)
            hk_mask = np.in1d(self.nomhk.tlm['MPS_ID'], mps_list)
            return {'sci_mask': sci_mask,
                    'hk_mask': hk_mask,
                    'dims': {
                        'number_of_images': np.sum(sci_mask),
                        'samples_per_image': np.max(
                            [self.science.images[ii].size
                             for ii in sci_mask.nonzero()[0]]),
                        'hk_packets': np.sum(hk_mask)}
                    }

        # mode == 'all':
        nr_hk = 0 if self.nomhk.hdr is None else len(self.nomhk.hdr)
        nr_sci = 0 if self.science.hdr is None else len(self.science.hdr)
        return {'hk_mask': np.full(nr_hk, True),
                'sci_mask': np.full(nr_sci, True),
                'dims': {
                    'number_of_images': nr_sci,
                    'samples_per_image': (
                        DET_CONSTS['dimRow'] if nr_sci == 0
                        else np.max([x.size for x in self.science.images])),
                    'hk_packets': nr_hk}
                }

    def l1a_file(self: SPXtlm, config: dataclass, mode: str) -> Path:
        """Return filename of level-1A product.

        Parameters
        ----------
        config :  dataclass
           Settings for the L0->l1A processing
        mode :  {'all', 'binned', 'full'}
           Select Science packages with full-frame images or binned images

        Returns
        -------
        Path
           Filename of level-1A product.

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

        # +++++ in-flight product-name convention +++++
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

        # +++++ OCAL product-name convention +++++
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
        """Generate a SPEXone level-1A product.

        Parameters
        ----------
        config :  dataclass
           Settings for the L0->L1A processing
        mode :  {'all', 'binned', 'full'}
           Select Science packages with full-frame images or binned images
        """
        selection = self.get_selection(mode)
        if selection is None:
            return

        # set time-coverage range
        coverage = None
        if np.sum(selection['sci_mask']) > 0:
            _mm = (selection['sci_mask']
                   & self.science.tstamp['tai_sec'] > TSTAMP_MIN)
            if np.any(_mm):
                tstamp = self.science.tstamp['dt'][_mm]
                ii = int(np.nonzero(_mm)[0][-1])
                intg = dt.timedelta(milliseconds=self.science.frame_period(ii))
                coverage = (tstamp[0], tstamp[-1] + intg)

        if coverage is None:
            dt_min = dt.datetime(2020, 1, 1, 1, tzinfo=dt.timezone.utc)
            tstamp = [self.nomhk.tstamp[ii]
                      for ii in np.nonzero(selection['hk_mask'])[0]]
            tstamp = [x for x in tstamp if x > dt_min]
            coverage = (tstamp[0], tstamp[-1] + dt.timedelta(seconds=1))

        l1a_file = self.l1a_file(config, mode)
        ref_date = self.reference_date
        with L1Aio(l1a_file, ref_date, selection['dims'],
                   compression=config.compression) as l1a:
            l1a.fill_global_attrs(inflight=config.l0_format != 'raw')
            l1a.set_attr('icu_sw_version',
                         f'0x{self.nomhk.tlm["ICUSWVER"][0]:x}')
            l1a.set_attr('time_coverage_start',
                         coverage[0].isoformat(timespec='milliseconds'))
            l1a.set_attr('time_coverage_end',
                         coverage[1].isoformat(timespec='milliseconds'))
            l1a.set_attr('input_files', [x.name for x in config.l0_list])
            self.logger.debug('(1) initialized level-1A product')

            self._fill_engineering(l1a, selection['hk_mask'])
            self.logger.debug('(2) added engineering data')
            self._fill_science(l1a, selection['sci_mask'])
            self.logger.debug('(3) added science data')
            self._fill_image_attrs(l1a, config.l0_format, selection['sci_mask'])
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

    def _fill_engineering(self: SPXtlm, l1a: L1Aio,
                          hk_mask: np.ndarray) -> None:
        """Fill datasets in group '/engineering_data'."""
        if np.sum(hk_mask) == 0:
            return

        l1a.set_dset('/engineering_data/NomHK_telemetry',
                     self.nomhk.tlm[hk_mask])
        ref_date = self.reference_date
        l1a.set_dset('/engineering_data/HK_tlm_time',
                     [(x - ref_date).total_seconds()
                      for ii, x in enumerate(self.nomhk.tstamp)
                      if hk_mask[ii]])
        l1a.set_dset('/engineering_data/temp_detector',
                     self.nomhk.convert('TS1_DEM_N_T')[hk_mask])
        l1a.set_dset('/engineering_data/temp_housing',
                     self.nomhk.convert('TS2_HOUSING_N_T')[hk_mask])
        l1a.set_dset('/engineering_data/temp_radiator',
                     self.nomhk.convert('TS3_RADIATOR_N_T')[hk_mask])

    def _fill_science(self: SPXtlm, l1a: L1Aio,
                      sci_mask: np.ndarray) -> None:
        """Fill datasets in group '/science_data'."""
        if np.sum(sci_mask) == 0:
            return

        img_list = [img for ii, img in enumerate(self.science.images)
                    if sci_mask[ii]]
        img_sz = [img.size for img in img_list]
        if len(np.unique(img_sz)) != 1:
            images = np.zeros((len(img_sz), np.max(img_sz)), dtype='u2')
            for ii, img in enumerate(img_list):
                images[ii, :len(img)] = img
        else:
            images = np.vstack(img_list)
        l1a.set_dset('/science_data/detector_images', images)
        l1a.set_dset('/science_data/detector_telemetry',
                     self.science.tlm[sci_mask])

    def _fill_image_attrs(self: SPXtlm, l1a: L1Aio,
                          lv0_format: str, sci_mask: np.ndarray) -> None:
        """Fill datasets in group '/image_attributes'."""
        if np.sum(sci_mask) == 0:
            return

        l1a.set_dset('/image_attributes/icu_time_sec',
                     self.science.tstamp['tai_sec'][sci_mask])
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
                     self.science.tstamp['sub_sec'][sci_mask])
        ref_date = self.reference_date
        l1a.set_dset('/image_attributes/image_time',
                     [(x - ref_date).total_seconds()
                      for x in self.science.tstamp['dt'][sci_mask]])
        l1a.set_dset('/image_attributes/image_ID',
                     np.bitwise_and(self.science.hdr['sequence'][sci_mask],
                                    0x3fff))
        l1a.set_dset('/image_attributes/binning_table',
                     self.science.binning_table()[sci_mask])
        l1a.set_dset('/image_attributes/digital_offset',
                     self.science.digital_offset()[sci_mask])
        l1a.set_dset('/image_attributes/exposure_time',
                     self.science.exposure_time()[sci_mask] / 1000)
        l1a.set_dset('/image_attributes/nr_coadditions',
                     self.science.tlm['REG_NCOADDFRAMES'][sci_mask])
