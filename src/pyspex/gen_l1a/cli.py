#!/usr/bin/env python3
#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2024 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Command-line implementation of spx1_level01a."""
from __future__ import annotations

import logging
import warnings

from pyspex.argparse_gen_l1a import argparse_gen_l1a
from pyspex.lib.check_input_files import check_input_files
from pyspex.lib.logger import start_logger
from pyspex.lv0_lib import CorruptPacketWarning
from pyspex.tlm import SPXtlm


def main() -> int:
    """Execute the main bit of the application."""
    error_code = 0
    warn_code = 0

    # initialize logger
    start_logger()
    logging.captureWarnings(True)

    # parse command-line parameters and YAML file for settings
    config = argparse_gen_l1a()
    logging.getLogger().setLevel(config.verbose)  # first, set the root logger
    logger = logging.getLogger("pyspex.gen_l1a")  # then initiate a descendant
    logger.debug("%s", config)

    # check input files (SEPXone level-0)
    try:
        check_input_files(config)
    except FileNotFoundError as exc:
        logger.fatal('File "%s" not found on system.', exc)
        return 110
    except TypeError as exc:
        logger.fatal("%s", exc)
        return 121

    # read level 0 data
    # Note that we read as much as possible packages from a file, but stop at
    # the first occurence of a corrupted data-packet, then the remainder of
    # the file is neglected.
    tlm = None
    with warnings.catch_warnings(record=True) as wrec_list:
        warnings.simplefilter("always", category=CorruptPacketWarning)
        try:
            tlm = SPXtlm()
            tlm.from_lv0(
                config.l0_list,
                file_format=config.l0_format,
                debug=config.debug,
                dump=config.dump,
            )
        except FileNotFoundError as exc:
            logger.fatal('FileNotFoundError exception raised for "%s".', exc)
            error_code = 110
        except TypeError as exc:
            logger.fatal('TypeError exception raised with "%s".', exc)
            error_code = 121

        for wrec in wrec_list:
            logger.warning('CorruptPacketWarning raised with "%s".', str(wrec.message))
            warn_code = 122

    if error_code != 0 or config.debug or config.dump:
        return error_code

    # Write Level-1A product.
    if not config.outdir.is_dir():
        config.outdir.mkdir(mode=0o755, parents=True)
    try:
        if config.eclipse is None:
            tlm.gen_l1a(config, "all")
        elif config.eclipse:
            tlm.gen_l1a(config, "full")
            tlm.gen_l1a(config, "binned")
        else:
            tlm.gen_l1a(config, "binned")
    except (KeyError, RuntimeError) as exc:
        # raise RuntimeError from exc
        logger.fatal('RuntimeError with "%s"', exc)
        error_code = 131
    except UserWarning as exc:
        logger.warning('navigation data is incomplete: "%s".', exc)
        error_code = 132
    except Exception as exc:
        # raise RuntimeError from exc
        logger.fatal('Unexpected exception occurred with "%s".', exc)
        error_code = 135

    return warn_code if error_code == 0 else error_code


if __name__ == "__main__":
    main()
