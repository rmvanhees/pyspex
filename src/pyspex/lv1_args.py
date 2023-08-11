#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Handle command-line parameters and settings from the YAML file
for L1A generation.
"""
from __future__ import annotations
__all__ = ['get_l1a_settings']

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .version import pyspex_version

# - global parameters ------------------------------
ARG_INPUT_HELP = """provide one or more input files:
- raw: CCSDS packages generated by the SPEXone ICU -- provide name
       of one file with extension '.H'. The files with science and
       housekeeping data are collected using Unix filename pattern
       matching.
- st3: CCSDS packages with ITOS and spacewire headers -- provide name
       of one file with extension '.ST3'.
- dsb: CCSDS packages with PACE headers -- provide list of filenames
       with extension '.spx'.
"""

ARG_YAML_HELP = """provide settings file in YAML format (in-flight example):

# define output directory, default is current working directory
outdir: .
# define name of output file, will be generated automatically when empty
outfile: ''
# compress the dataset /science_data/detector_images
compression: True
# define file-version as nn, neglected when outfile not empty
file_version: 1
# flag to indicate measurements taken in eclipse or day-side
eclipse: True
# provide list, directory, file-glob or empty
hkt_list: <PATH>/PACE.20220617T011*.HKT.nc
# must be a list, directory or glob. Fails when empty
l0_list:
- <PATH>/SPX022000010.spx
- <PATH>/SPX022000011.spx
- <PATH>/SPX022000012.spx
- <PATH>/SPX022000013.spx

"""

_prog_name_ = Path(sys.argv[0]).name
EPILOG_HELP = f"""Usage:
  Generate L1A from OCAL level-0 data directly from the SPEXone instrument:

    {_prog_name_} <Path>/NomSciCal1_20220123T121801.676167.H

    Note that OCAL science & telemetry data is read from the files:
      <Path>/NomSciCal1_20220123T121801.676167.?
      <Path>/NomSciCal1_20220123T121801.676167.??
      <Path>/NomSciCal1_20220123T121801.676167_hk.?

  Generate L1A from OCAL level-0 data via ITOS from the PACE platform:

    {_prog_name_} <Path>/DIAG_20220124_175458_073.ST3

  Generate L1A from in-flight level-0 data, store product in directory L1A:

    {_prog_name_} --outdir L1A <Path>/SPX*.spx

  Generate L1A from in-flight level-0 data read settings from a YAML file:

    {_prog_name_} --yaml config_l1a_gen.yaml

    An example YAML file:
       outdir: L1A
       outfile: ''
       file_version: 1
       compression: False
       eclipse: False
       hkt_list: HKT/PACE.20220617T011*.HKT.nc
       l0_list: <PATH>/SPX0220000??.spx

  Dry-run, be extra verbose without generating data:

    {_prog_name_} --debug <Path>/NomSciCal1_20220123T121801.676167.H

  Read level-0 data and dump CCSDS packet headers in ASCII:

    {_prog_name_} --dump <Path>/NomSciCal1_20220123T121801.676167.H

Return codes:
  2      Failed to parse command-line parameters.
  110    One (or more) SPEXone level-0 files not found.
  115    Failed to generate output directry due to permission error.
  121    Input file not recognized as a SPEXone level-0 product.
  122    Corrupted SPEXone level-0 data.
  131    Failed to generate output file due to netCDF4 library issues.
  132    Incomplete set of navigation data detected
  135    Failed to generate output file due to permission error.

Environment:
   'OCVARROOT'
       The number of leap seconds for the TAI to UTC conversion are determined
       using the file 'tai-utc.dat'. A copy of this file is included in the
       package `pyspex`. The latest version can be obtained from
         `https://maia.usno.navy.mil/ser7/tai-utc.dat`.
       When OCVARROOT is set the path should be '$OCVARROOT/common/tai-utc.dat'.
"""


# - local functions --------------------------------
# pylint: disable=too-many-instance-attributes
@dataclass()
class Config:
    """Initiate class to hold settings for L0->L1a processing."""

    debug: bool = False
    dump: bool = False
    verbose: bool = False
    compression: bool = False
    outdir: Path = Path('.').resolve()
    outfile: str = ''
    file_version: int = 1
    eclipse: bool | None = None
    yaml_fl: Path = None
    hkt_list: list[Path] = field(default_factory=list)
    l0_format: str = ''
    l0_list: list[Path] = field(default_factory=list)


def __commandline_settings():
    """Parse command-line parameters."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Generate PACE level-1A product from SPEXone level-0 data.',
        epilog=EPILOG_HELP)
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s v' + pyspex_version())
    parser.add_argument('--debug', action='store_true',
                        help='be extra verbose, no output files generated')
    parser.add_argument('--dump', action='store_true',
                        help='dump CCSDS packet headers in ASCII')
    parser.add_argument('--verbose', action='store_true', help='be verbose')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--eclipse', action='store_true', default=None)
    group.add_argument('--no_eclipse', dest='eclipse', action='store_false')
    parser.add_argument('--outdir', type=Path, default=None,
                        help=('directory to store the generated'
                              ' level-1A product(s)'))
    # group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--yaml', type=Path, default=None, help=ARG_YAML_HELP)
    parser.add_argument('lv0_list', nargs='*', help=ARG_INPUT_HELP)
    args = parser.parse_args()

    config = Config()
    if args.debug:
        config.debug = True
    if args.dump:
        config.dump = True
    if args.verbose:
        config.verbose = True
    if args.eclipse is not None:
        config.eclipse = args.eclipse
    if args.outdir is not None:
        config.outdir = args.outdir
    if args.yaml:
        config.yaml_fl = args.yaml
    elif args.lv0_list:
        config.l0_list = [Path(x) for x in args.lv0_list]
    else:
        parser.error('You should provide a YAML file or names of L0 products')

    return config


# pylint: disable=too-many-branches
def __yaml_settings(config: dataclass) -> dataclass:
    """Read YAML configuration file."""
    with open(config.yaml_fl, encoding='ascii') as fid:
        config_yaml = yaml.safe_load(fid)

    if 'outdir' in config_yaml and config_yaml['outdir'] is not None:
        config.outdir = Path(config_yaml['outdir'])
    if 'outfile' in config_yaml and config_yaml['outfile']:
        config.outfile = config_yaml['outfile']
    if 'compression' in config_yaml and config_yaml['compression']:
        config.compression = True
    if 'file_version' in config_yaml and config_yaml['file_version'] != 1:
        config.file_version = config_yaml['file_version']
    if 'eclipse' in config_yaml and config_yaml['eclipse'] is not None:
        config.eclipse = config_yaml['eclipse']
    if 'hkt_list' in config_yaml and config_yaml['hkt_list']:
        if isinstance(config_yaml['hkt_list'], list):
            config.hkt_list = [Path(x) for x in config_yaml['hkt_list']]
        else:
            mypath = Path(config_yaml['hkt_list'])
            if mypath.is_dir():
                config.hkt_list = sorted(Path(mypath).glob('*'))
            else:
                config.hkt_list = sorted(Path(mypath.parent).glob(mypath.name))
    if 'l0_list' in config_yaml and config_yaml['l0_list']:
        if isinstance(config_yaml['l0_list'], list):
            config.l0_list = [Path(x) for x in config_yaml['l0_list']]
        else:
            mypath = Path(config_yaml['l0_list'])
            if mypath.is_dir():
                config.l0_list = sorted(Path(mypath).glob('*'))
            else:
                config.l0_list = sorted(Path(mypath.parent).glob(mypath.name))

    return config


def check_input_files(config: dataclass) -> dataclass:
    """
    Check level-0 files on existence and format.

    Parameters
    ----------
    config :  dataclass

    Returns
    -------
    dataclass
       fields 'l0_format' {'raw', 'st3', 'dsb'} and 'l0_list' are updated.

    Raises
    ------
    FileNotFoundError
       If files are not found on the system.
    TypeError
       If determined file type differs from value supplied by user.
    """
    file_list = config.l0_list
    if file_list[0].suffix == '.H':
        if not file_list[0].is_file():
            raise FileNotFoundError(file_list[0])
        data_dir = file_list[0].parent
        file_stem = file_list[0].stem
        file_list = (sorted(data_dir.glob(file_stem + '.[0-9]'))
                     + sorted(data_dir.glob(file_stem + '.?[0-9]'))
                     + sorted(data_dir.glob(file_stem + '_hk.[0-9]')))
        if not file_list:
            raise FileNotFoundError('No measurement or housekeeping data found')

        config.l0_format = 'raw'
        config.l0_list = file_list
    elif file_list[0].suffix == '.ST3':
        if not file_list[0].is_file():
            raise FileNotFoundError(file_list[0])
        config.l0_format = 'st3'
        config.l0_list = [file_list[0]]
    elif file_list[0].suffix == '.spx':
        file_list_out = []
        for flname in file_list:
            if not flname.is_file():
                raise FileNotFoundError(flname)

            if flname.suffix == '.spx':
                file_list_out.append(flname)

        if not file_list:
            raise FileNotFoundError('No measurement or housekeeping data found')
        config.l0_format = 'dsb'
        config.l0_list = file_list_out
    else:
        raise TypeError('File not recognized as SPEXone level-0 data')

    return config


# - main function ----------------------------------
def get_l1a_settings() -> dataclass:
    """Obtain settings from both command-line and YAML file (when provided).

    Returns
    -------
    dataclass
       settings from both command-line arguments and YAML config-file
    """
    config = __commandline_settings()
    if config.yaml_fl is not None:
        if not config.yaml_fl.is_file():
            raise FileNotFoundError('settings file not found')

        config = __yaml_settings(config)

    return check_input_files(config)
