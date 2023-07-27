#!/usr/bin/env python3
#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2023 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Command-line implementation of spx1_level01a.
"""
from ..lv1_args import get_l1a_settings
from ..tlm import SPXtlm


def main() -> int:
    """Execute the main bit of the application."""
    err_code = 0

    # parse command-line parameters and YAML file for settings
    try:
        config = get_l1a_settings()
        if config.verbose:
            print(config)
        if not config.outdir.is_dir():
            config.outdir.mkdir(mode=0o755, parents=True)

        # read level 0 data
        tlm = SPXtlm(config.verbose)
        tlm.from_lv0(config.l0_list,
                     file_format=config.l0_format,
                     debug=config.debug,
                     dump=config.dump)

    except FileNotFoundError as exc:
        print(f'[FATAL]: FileNotFoundError exception raised with "{exc}".')
        err_code = 110
    except OSError as exc:
        print(f'[FATAL]: PermissionError with "{exc}"')
        return 115
    except TypeError as exc:
        print(f'[FATAL]: TypeError exception raised with "{exc}".')
        err_code = 121
    except ValueError as exc:
        print(f'[FATAL]: ValueError exception raised with "{exc}".')
        err_code = 122

    if err_code != 0 or config.debug or config.dump:
        return err_code

    # Write Level-1A product.
    try:
        if config.eclipse is None:
            tlm.gen_l1a(config, 'all')
        elif config.eclipse:
            tlm.gen_l1a(config, 'binned')
            tlm.gen_l1a(config, 'full')
        else:
            tlm.gen_l1a(config, 'binned')
    except (KeyError, RuntimeError) as exc:
        print(f'[FATAL]: RuntimeError with "{exc}"')
        err_code = 131
    except UserWarning as exc:
        print('[WARNING]: navigation data is incomplete,'
              f' original message: "{exc}"')
        err_code = 132
    except Exception as exc:
        print(f'[FATAL]: Error with "{exc}"')
        err_code = 135

    return err_code
