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
"""Command-line implementation of spx1_level01a."""
from __future__ import annotations

import logging

from pyspex.argparse_gen_l1a import argparse_gen_l1a
from pyspex.lib.check_input_files import check_input_files
from pyspex.lib.logger import start_logger
from pyspex.tlm import SPXtlm


def main() -> int:
    """Execute the main bit of the application."""
    err_code = 0

    # initialize logger
    start_logger()

    # parse command-line parameters and YAML file for settings
    config = argparse_gen_l1a()
    logger = logging.getLogger('pyspex.gen_l1a')
    logger.setLevel(config.verbose)
    logger.debug('%s', config)

    # check input files (SEPXone level-0)
    try:
        check_input_files(config)
    except FileNotFoundError as exc:
        logger.fatal('File "%s" not found on system.', exc)
        return 110
    except TypeError as exc:
        logger.fatal('%s', exc)
        return 121

    # read level 0 data
    tlm = None
    try:
        tlm = SPXtlm()
        tlm.from_lv0(config.l0_list,
                     file_format=config.l0_format,
                     debug=config.debug,
                     dump=config.dump)
    except FileNotFoundError as exc:
        logger.fatal('FileNotFoundError exception raised for "%s".', exc)
        err_code = 110
    except TypeError as exc:
        logger.fatal('TypeError exception raised with "%s".', exc)
        err_code = 121
    except ValueError as exc:
        logger.fatal('ValueError exception raised with "%s".', exc)
        err_code = 122

    if err_code != 0 or config.debug or config.dump:
        return err_code

    # Write Level-1A product.
    if not config.outdir.is_dir():
        config.outdir.mkdir(mode=0o755, parents=True)
    try:
        if config.eclipse is None:
            tlm.gen_l1a(config, 'all')
        elif config.eclipse:
            tlm.gen_l1a(config, 'binned')
            tlm.gen_l1a(config, 'full')
        else:
            tlm.gen_l1a(config, 'binned')
    except (KeyError, RuntimeError) as exc:
        logger.fatal('RuntimeError with "%s"', exc)
        err_code = 131
    except UserWarning as exc:
        logger.warning('navigation data is incomplete: "%s".', exc)
        err_code = 132
    except Exception as exc:
        logger.fatal('Unexpected exception occurred with "%s".', exc)
        err_code = 135

    return err_code


if __name__ == '__main__':
    main()
