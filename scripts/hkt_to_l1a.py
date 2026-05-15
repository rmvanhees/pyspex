#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2022-2025 SRON
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Illustrate how to write PACE_HKT data to a SPEXone Level-1A product."""

from __future__ import annotations

from pathlib import Path

from pyspex.gen_l1a.argparse_gen_l1a import Config
from pyspex.gen_l1a.l1a import SpexL1A
from pyspex.hkt_io import HKTio
from pyspex.tlm import SPXtlm, get_l1a_filename

HKT_DIR = Path("/nfs/SPEXone/ical/archives/pace_hkt/V1.0/2024/07/01")
HKT_LIST = [
    HKT_DIR / "PACE.20240701T055223.HKT.nc",
    HKT_DIR / "PACE.20240701T055757.HKT.nc",
    HKT_DIR / "PACE.20240701T060332.HKT.nc",
    HKT_DIR / "PACE.20240701T060908.HKT.nc",
    HKT_DIR / "PACE.20240701T061445.HKT.nc",
    HKT_DIR / "PACE.20240701T062022.HKT.nc",
    HKT_DIR / "PACE.20240701T062600.HKT.nc",
    HKT_DIR / "PACE.20240701T063137.HKT.nc",
    HKT_DIR / "PACE.20240701T063714.HKT.nc",
    HKT_DIR / "PACE.20240701T064252.HKT.nc",
]


def main() -> None:
    """..."""
    # define settings
    config = Config(outdir=Path("."), hkt_list=HKT_LIST)

    # read SPEXone telemetry data from HKT file(s)
    tlm = SPXtlm()
    tlm.from_hkt(HKT_LIST)

    # read navigation data from HKT file(s)
    hkt = HKTio(HKT_LIST)
    nav_dict = hkt.navigation()
    nav_dict["coverage_quality"] = 0

    # write HKT data to SPEXone Level-1A product
    with SpexL1A(
        get_l1a_filename(config, tlm.coverage, None),
        tlm.coverage,
        dims={
            "hk_packets": tlm.nomhk.size,
            "number_of_images": 0,
            "samples_per_image": 1,
            "/navigation_data/att_time": nav_dict["att_time"].size,
            "/navigation_data/orb_time": nav_dict["orb_time"].size,
            "/navigation_data/tilt_time": nav_dict["tilt_time"].size,
        },
    ) as l1a:
        l1a.write_config(config)
        # l1a.write_img_vars(tlm.science)  # before write_hk_vars
        l1a.write_hk_vars(tlm.nomhk)
        l1a.write_nav_vars(nav_dict)


if __name__ == "__main__":
    main()
