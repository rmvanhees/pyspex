#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2025 SRON
#    All Rights Reserved
#
# Author: Sean Bailey - DAAC Manager, OB.DAAC, NASA/GSFC Code 619
# License:  BSD-3-Clause
#
"""Python script to determine the number of leap seconds for given timestamp.

The source for the latest version of tai-utc.dat is the US Naval Observatory:

   https://maia.usno.navy.mil/ser7/tai-utc.dat
"""

from __future__ import annotations

__all__ = ["get_leap_seconds"]

import datetime as dt
import logging
from importlib.resources import files
from os import environ
from pathlib import Path

import julian

# - global parameters -----------------------
module_logger = logging.getLogger("pyspex.lib.leap_sec")


def get_leap_seconds(taitime: float, epochyear: int = 1958) -> float:
    """Return the number of elapsed leap seconds given a TAI time in seconds.

    Requires the file tai-utc.dat.
    """
    # determine location of the file 'tai-utc.dat'
    ocvarroot = environ.get("OCVARROOT")
    if ocvarroot is None:
        taiutc = files("pyspex.Data").joinpath("tai-utc.dat")
    else:
        taiutc = Path(ocvarroot) / "common" / "tai-utc.dat"

    epochsecs = dt.datetime(epochyear, 1, 1, tzinfo=dt.UTC).timestamp()
    taidt = dt.datetime.fromtimestamp(taitime + epochsecs, dt.UTC)
    leapsec: float = 0
    with taiutc.open("r", encoding="ascii") as fp:
        for line in fp:
            rec = line.rstrip().split(None, 7)
            if julian.from_jd(float(rec[4])).replace(tzinfo=dt.UTC) < taidt:
                leapsec = float(rec[6])
    module_logger.debug("leap seconds: %f", leapsec)

    return leapsec
