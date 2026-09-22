#
# This file is part of pyspex:
#    https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2026 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
#
"""Generate a SPEXone level-1A product (netCDF4 format)."""

from __future__ import annotations

import logging
import traceback
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

    # (1) parse command-line parameters and YAML file for settings
    config = argparse_gen_l1a()
    if config.verbose == "debug":
        print(config)

    # (2) initialize logger
    logging.captureWarnings(True)
    start_logger(config.verbose.upper())
    logger = logging.getLogger("pyspex.spx1_level01a")

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

    # (4.1) keep nomhk packages with a MPS_ID present in the science packages
    hk_mps_list = np.unique(tlm.nomhk.tlm["MPS_ID"])
    sci_mps_list = np.unique(tlm.science.tlm["MPS_ID"])
    if not np.array_equal(hk_mps_list, sci_mps_list):
        logger.debug("Science vs nomhk MPS: %s - %s", sci_mps_list, hk_mps_list)
        tlm.nomhk = tlm.nomhk.sel(np.isin(tlm.nomhk.tlm["MPS_ID"], sci_mps_list))

    # (5) read navigation data from PACE_HKT products
    hkt = HKTio(config.hkt_list) if config.hkt_list else None

    def get_hkt_nav() -> dict | None:
        """..."""
        if hkt is None:
            return None

        nav_dict = hkt.navigation()
        nav_dict = hkt.nav_coverage_adjust(nav_dict, coverage_spx)
        nav_dict["coverage_quality"] = hkt.nav_coverage_flag(coverage_spx)
        return nav_dict

    # (6) write Level-1A product.
    try:
        if config.eclipse is None or tlm.science.size == 0:
            coverage_spx = (
                tlm.coverage[0].replace(tzinfo=None),
                tlm.coverage[1].replace(tzinfo=None),
            )
            create_l1a(config, tlm, get_hkt_nav())
        elif config.eclipse:
            # binned measurements
            tlm0 = tlm.binned()
            coverage_spx = (
                tlm0.coverage[0].replace(tzinfo=None),
                tlm0.coverage[1].replace(tzinfo=None),
            )
            create_l1a(config, tlm0, get_hkt_nav())
            del tlm0

            # full-frame measurements
            config.outfile = ""
            tlm = tlm.full()
            coverage_spx = (
                tlm.coverage[0].replace(tzinfo=None),
                tlm.coverage[1].replace(tzinfo=None),
            )
            create_l1a(config, tlm, get_hkt_nav())
        else:
            # binned measurements
            tlm = tlm.binned()
            coverage_spx = (
                tlm.coverage[0].replace(tzinfo=None),
                tlm.coverage[1].replace(tzinfo=None),
            )
            create_l1a(config, tlm, get_hkt_nav())
    except (KeyError, OSError, RuntimeError) as exc:
        # raise RuntimeError from exc
        logger.fatal('RuntimeError with "%s"', exc)
        error_code = 131
    except UserWarning as exc:
        logger.warning('navigation data is incomplete: "%s".', exc)
        error_code = 132
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        logger.fatal('Unexpected exception occurred with "%s".', exc)
        error_code = 135

    return warn_code if error_code == 0 else error_code
