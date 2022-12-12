#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Handle command-line parameters and settings from the YAML file
for L1A generation.
"""
__all__ = ['get_l1a_settings']

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import yaml

# - global parameters ------------------------------
ARG_INPUT_HELP = """Provide one or more input files:
- raw: `OCAL ambient measurement` -- provide name of one file with extension
       '.H'. The files with science and house-keeping data are collected using
       Unix filename pattern matching.
- st3: CCSDS packages with ITOS and spacewire headers -- provide name of
       one file with extension '.ST3'.
- dsb: CCSDS packages with PACE headers -- provide list of filenames with
       extension '.spx'.
"""

ARG_YAML_HELP = """Provide settings file in YAML format as:

 # define output directory, CWD when empty
 outdir: CWD
 # define name of output file, will be generated automatically when empty
 outfile: ''
 # define file-version as nn, neglected when outfile not empty
 file_version: 1
 # flag to indicate measurements taken in eclipse or day-side
 eclipse: True
 # provide list, directory, file-glob or empty
 hkt_list: ''
 # must be a list, directory or glob. Fails when empty
 l0_list: L0/SPX0000000??.spx

"""

EPILOG_HELP = """Usage:
  Read inflight Level-0 data and write Level-1A product in directory L1A:

    spx1_level01a.py --outdir L1A <Path>/SPX*.spx

  If the Level-0 data contains Science and diagnostig measurements then use:

    spx1_level01a.py --outdir L1A <Path>/SPX*.spx --select binned
  or
    spx1_level01a.py --outdir L1A <Path>/SPX*.spx --select fullFrame

  Same call but now we add navigation data from HKT products:

    spx1_level01a.py --outdir L1A <Path>/SPX*.spx --pace_hkt <Path>/PACE.20220621T14*.HKT.nc

  Read OCAL Level-0 data and write Level-1A product in directory L1A:

    spx1_level01a.py --outdir L1A <Path>/NomSciCal1_20220123T121801.676167.H

    Note that OCAL science & telemetry data is read from the files:
      <Path>/NomSciCal1_20220123T121801.676167.?
      <Path>/NomSciCal1_20220123T121801.676167.??
      <Path>/NomSciCal1_20220123T121801.676167_hk.?

  Same call but now we are verbose during the data read (no output generated):

    spx1_level01a.py --debug <Path>/NomSciCal1_20220123T121801.676167.H

  Read ST3 Level-0 file and write Level-1A product in directory L1A:

    spx1_level01a.py --outdir L1A <Path>/SCI_20220124_174737_419.ST3

  Same call but now we dump packet header information in ASCII

    spx1_level01a.py --outdir L1A <Path>/SCI_20220124_174737_419.ST3 --dump
"""


# - local functions --------------------------------
# pylint: disable=too-many-instance-attributes
@dataclass(slots=True)
class Config:
    """Initiate class to hold settings for L0->L1a processing."""
    debug: bool = False
    dump: bool = False
    verbose: bool = False
    outdir: Path = Path('.').resolve()
    outfile: str = ''
    file_version: int = 1
    eclipse: bool = None
    yaml_fl: Path = None
    hkt_list: list[Path] = field(default_factory=list)
    l0_format: str = ''
    l0_list: list[Path] = field(default_factory=list)


def __commandline_settings():
    """Parse command-line parameters."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Store SPEXone Level-0 data in a Level-1A product',
        epilog=EPILOG_HELP)
    parser.add_argument('--debug', action='store_true', help='be more verbose')
    parser.add_argument('--dump', action='store_true',
                        help='dump CCSDS packet headers in ASCII')
    parser.add_argument('--verbose', action='store_true', help='be verbose')
    parser.add_argument('--outdir', type=Path, default=None,
                        help=('Directory to store the generated'
                              ' level-1A product(s)'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--yaml', type=Path, default=None, help=ARG_YAML_HELP)
    group.add_argument('--spex_lv0', nargs='+', help=ARG_INPUT_HELP)
    args = parser.parse_args()

    config = Config()
    if args.debug:
        config.debug = True
    if args.dump:
        config.dump = True
    if args.verbose:
        config.verbose = True
    if args.outdir is not None:
        config.outdir = args.outdir
    if args.spex_lv0:
        config.l0_list = [Path(x) for x in args.spex_lv0]
    else:
        config.yaml_fl = args.yaml

    return config


def __yaml_settings(config):
    """Read YAML configuration file."""
    with open(config.yaml_fl, encoding='ascii') as fid:
        config_yaml = yaml.safe_load(fid)

    if 'outdir' in config_yaml and config_yaml['outdir'] is not None:
        config.outdir = Path(config_yaml['outdir'])
    if 'outfile' in config_yaml and config_yaml['outfile']:
        config.outfile = config_yaml['outfile']
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
                config.hkt_list = list(Path(mypath).glob('*'))
            else:
                config.hkt_list = list(Path(mypath.parent).glob(mypath.name))
    if 'l0_list' in config_yaml and config_yaml['l0_list']:
        if isinstance(config_yaml['l0_list'], list):
            config.l0_list = [Path(x) for x in config_yaml['l0_list']]
        else:
            mypath = Path(config_yaml['l0_list'])
            if mypath.is_dir():
                config.l0_list = list(Path(mypath).glob('*'))
            else:
                config.l0_list = list(Path(mypath.parent).glob(mypath.name))

    return config


def check_input_files(config):
    """
    Check level-0 files on existence and format

    Parameters
    ----------
    config :  dataclass

    Returns
    -------
    dataclass
       fields 'file_format' {'raw', 'st3', 'dsb'} and 'file_list' are updated

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
        config.l0_list = [file_list[0]]
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
