#!/usr/bin/env python3
#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Command-line implementation of spx1_level01a."""

from ..lv0_io import dump_lv0_data, read_lv0_data
from ..lv1_io import write_l1a

from ..lv1_args import get_l1a_settings


def main() -> int:
    """Execute the main bit of the application."""

    # parse command-line parameters and YAML file for settings
    try:
        config = get_l1a_settings()
    except FileNotFoundError as exc:
        print(f'[FATAL]: FileNotFoundError exception raised with "{exc}".')
        return 100
    except TypeError as exc:
        print(f'[FATAL]: TypeError exception raised with "{exc}".')
        return 101

    # show the user command-line steeings after calling `check_input_files`
    if config.verbose:
        print(config)

    # read level 0 data as Science and TMTC packages
    try:
        res = read_lv0_data(config.l0_list, config.l0_format,
                            config.debug, config.verbose)
    except ValueError as exc:
        print(f'[FATAL]: ValueError exception raised with "{exc}".')
        return 110
    if config.debug:
        return 0

    # perform an ASCII dump of level 0 headers parameters
    if config.dump:
        try:
            dump_lv0_data(config.l0_list, config.outdir, *res)
        except FileNotFoundError as exc:
            print(f'[FATAL]: FileNotFoundError exception raised with "{exc}".')
            return 132

        if config.verbose:
            print(f'Wrote ASCII dump in directory: {config.outdir}')
        return 0

    # we will not create a Level-1A product without Science data.
    if not res[0]:
        # inform the caller with a warning message and exit status
        print('[WARNING]: no science data found in L0 data, exit')
        return 110

    # Write Level-1A product.
    try:
        if not config.outdir.is_dir():
            config.outdir.mkdir(mode=0o755, parents=True)
        write_l1a(config, res[0], res[1])
    except (KeyError, RuntimeError) as exc:
        print(f'[FATAL]: RuntimeError with "{exc}"')
        return 131
    except Exception as exc:
        print(f'[FATAL]: PermissionError with "{exc}"')
        return 130

    return 0
