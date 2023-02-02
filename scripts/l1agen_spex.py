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
"""Python script to store SPEXone Level-0 data in a Level-1A product.
"""
import sys
from datetime import datetime
from pathlib import Path

from pyspex.gen_l1a.cli import main
from pyspex.version import pyspex_version

if __name__ == '__main__':
    mtime_str = datetime.fromtimestamp(
        Path(__file__).stat().st_mtime).isoformat(sep=' ', timespec='seconds')
    print(f'l1agen_spex.py {pyspex_version()} ({mtime_str})\n')
    sys.exit(main())
