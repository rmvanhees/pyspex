#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# Author: Sean Bailey - DAAC Manager, OB.DAAC, NASA/GSFC Code 619
# License:  BSD-3-Clause
"""
Python script to determine the number of leap seconds for given timestamp.

The source for the latest version of tai-utc.dat is the US Naval Observatory:

   https://maia.usno.navy.mil/ser7/tai-utc.dat
"""
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from os import environ

import julian


def get_leap_seconds(taitime: float, epochyear: int = 1958) -> float:
    """
    Return the number of elapsed leap seconds given a TAI time in seconds
    Requires tai-utc.dat
    """
    # determine location of the file 'tai-utc.dat'
    ocvarroot = environ['OCVARROOT'] if 'OCVARROOT' in environ else None
    if ocvarroot is None:
        taiutc = resources.files('pyspex.data').joinpath('tai-utc.dat')
    else:
        taiutc = Path(ocvarroot) / 'common' / 'tai-utc.dat'

    epochsecs = (datetime(epochyear, 1, 1, tzinfo=timezone.utc)
                 - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()
    taidt = datetime.utcfromtimestamp(taitime + epochsecs)
    leapsec = 0
    with taiutc.open("r", encoding='ascii') as fp:
        for line in fp:
            rec = line.rstrip().split(None, 7)
            if julian.from_jd(float(rec[4])) < taidt:
                leapsec = float(rec[6])

    return leapsec
