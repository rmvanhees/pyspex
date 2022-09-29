#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause

from datetime import datetime, timezone

# The on-ground calibration campaigns are expected to be finished long
# before the expected launch date of PACE in 2023
LAUNCH_DATE = datetime(2023, 1, 1, tzinfo=timezone.utc)


def before_launch(timestamp: int) -> bool:
    """
    Return True when timestamp is before launch of PACE
    """
    return timestamp < LAUNCH_DATE
