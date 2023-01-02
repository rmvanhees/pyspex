#!/usr/bin/env python3
#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Python script to store SPEXone Level-0 data in a Level-1A product."""

import sys

from pyspex.gen_l1a.cli import main


if __name__ == '__main__':
    sys.exit(main())
