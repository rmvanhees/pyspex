#
# This file is part of pyspex:
#    https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Generate a SPEXone level-1A product (netCDF4 format)."""

from __future__ import annotations

import logging
import warnings

import numpy as np

from pyspex.hkt_io import HKTio
from pyspex.lib.logger import start_logger
from pyspex.lv0_lib import CorruptPacketWarning
from pyspex.tlm import SPXtlm

from .argparse_gen_l1a import argparse_gen_l1a
from .l1a import check_input_files, create_l1a


# - main function ----------------------------------
def main() -> int:
    """Execute the main bit of the application."""
    error_code = 0
    warn_code = 0

    # (1) initialize logger
    start_logger()
    logging.captureWarnings(True)

    # (2) parse command-line parameters and YAML file for settings
    config = argparse_gen_l1a()
    logging.getLogger().setLevel(config.verbose)  # first, set the root logger
    logger = logging.getLogger("pyspex.gen_l1a")  # then initiate a descendant
    logger.debug("%s", config)

    # (3) check input files (SEPXone level-0)
    try:
        check_input_files(config)
    except FileNotFoundError as exc:
        logger.fatal('File "%s" not found on system.', exc)
        return 110
    except TypeError as exc:
        logger.fatal("%s", exc)
        return 121

    # (4) read level 0 data
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

    if tlm.science.size > 0:
        mps_list = np.unique(tlm.science.tlm["MPS_ID"])
        logger.debug("unique Science MPS: %s", mps_list)
        tlm.nomhk = tlm.nomhk.sel(np.isin(tlm.nomhk.tlm["MPS_ID"], mps_list))

    # (5) read navigation data from PACE_HKT products
    nav_dict = None
    if config.hkt_list:
        coverage_spx = (
            tlm.coverage[0].replace(tzinfo=None),
            tlm.coverage[1].replace(tzinfo=None),
        )
        hkt = HKTio(config.hkt_list)
        nav_dict = hkt.navigation()
        nav_dict = hkt.nav_coverage_adjust(nav_dict, coverage_spx)
        nav_dict["coverage_quality"] = hkt.nav_coverage_flag(coverage_spx)

    # (6) write Level-1A product.
    try:
        if config.eclipse is None or tlm.science.size == 0:
            create_l1a(config, tlm, nav_dict)
        elif config.eclipse:
            # binned measurements
            create_l1a(config, tlm.binned(), nav_dict, "binned")

            # full-frame measurements
            create_l1a(config, tlm.full(), nav_dict, "full")
        else:
            # binned measurements
            create_l1a(config, tlm.binned(), nav_dict, "binned")
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
