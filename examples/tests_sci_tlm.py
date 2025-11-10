"""
Run tests on SCItlm methods 'sel' and 'append'.

A SPEXone L0 product may contain a mix of Science and Diagnostic measurements,
while a SPEXone L1A product contains Science or Diagnostic measurements. Where:
 - SPEXone detector images are alway stored in a 1-D array
 - Diagnostic images are always 4194304 (= 2048 ** 2) in size
 - Science images are determined by the size of the binning table.

SPXtlm reads the images of SPEXone L0 products as a tuple of 1-D arrays, while the
images of the SPEXone L1A product can be stored as 2-D arrays.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyspex.tlm import SPXtlm

SPX_L0_DIR = Path("/media/richardh/T9-RvH/SPEXone/ical/archives/spx1_l0_diag/0x12d")
if not SPX_L0_DIR.is_dir():
    SPX_L0_DIR = Path("/nfs/SPEXone/ical/archives/spx1_l0_diag/0x12d")
SPX_L1A_DIR = Path("/media/richardh/T9-RvH/SPEXone/ical/archives/spx1_l1a_diag/V1.4")
if not SPX_L1A_DIR.is_dir():
    SPX_L1A_DIR = Path("/nfs/SPEXone/ical/archives/spx1_l1a_diag/V1.4")


# ------------------------- unit-tests -------------------------
def check_lv0_sci_sel(lv0_prod: Path, mps: int | None = None) -> None:
    """..."""
    spx = SPXtlm()
    spx.from_lv0(lv0_prod, tlm_type="sci")
    if mps is None:
        spx = spx.binned()
    else:
        mask = spx.science.tlm["MPS_ID"] == mps
        spx = spx.sel(mask)

    print(f"MPS: {spx.science.tlm['MPS_ID']}")
    print(
        "check_lv0_sci_sel :",
        type(spx.science.images),
        len(spx.science.images),
        spx.science.images[0].dtype,
        spx.science.images[0].shape,
    )


def check_lv0_diag_sel(lv0_prod: Path, mps: int | None = None) -> None:
    """..."""
    spx = SPXtlm()
    spx.from_lv0(lv0_prod, tlm_type="sci")
    if mps is None:
        spx = spx.full()
    else:
        mask = spx.science.tlm["MPS_ID"] == mps
        spx = spx.sel(mask)

    print(f"MPS: {spx.science.tlm['MPS_ID']}")
    print(
        "check_lv0_diag_sel :",
        type(spx.science.images),
        len(spx.science.images),
        spx.science.images[0].dtype,
        spx.science.images[0].shape,
    )


def check_l1a_sci_sel(
    prod_list: tuple[Path], do_sel: bool = False, mps: int | None = None
) -> None:
    """..."""
    sci_all = SPXtlm().science  # empty SCItlm object
    for l1a_prod in prod_list:
        spx = SPXtlm()
        spx.from_l1a(l1a_prod, tlm_type="sci", mps_id=mps)
        if spx.science.size == 0:
            continue

        sci_all.append(spx.science)

    if do_sel:
        mask = (np.arange(sci_all.images.shape[0]) % 2).astype(bool)

    print(
        "check_l1a_sci_sel :",
        type(sci_all.images),
        len(sci_all.images),
        sci_all.images.dtype,
        sci_all.images.shape,
        sci_all.sel(mask).images.shape if do_sel else "",
    )


def check_l1a_diag_sel(
    prod_list: tuple[Path], do_sel: bool = False, mps: int | None = None
) -> None:
    """..."""
    sci_all = SPXtlm().science  # empty SCItlm object
    for l1a_prod in prod_list:
        spx = SPXtlm()
        spx.from_l1a(l1a_prod, tlm_type="sci", mps_id=mps)
        if spx.science.size == 0:
            continue

        sci_all.append(spx.science)

    if do_sel:
        mask = (np.arange(sci_all.images.shape[0]) % 2).astype(bool)

    print(
        "check_l1a_diag_sel :",
        type(sci_all.images),
        len(sci_all.images),
        sci_all.images.dtype,
        sci_all.images.shape,
        sci_all.sel(mask).images.shape if do_sel else "",
    )


def check_lv0() -> None:
    """..."""
    date_dir = Path("2024") / "07" / "07"

    # SPEXone L1A science measurements
    prod_list = sorted(SPX_L0_DIR.glob(date_dir / "SPX0000?????.spx"))
    print(prod_list[0])
    check_lv0_sci_sel([prod_list[0]])
    check_lv0_sci_sel([prod_list[0]], mps=46)

    check_lv0_diag_sel([prod_list[1]])
    check_lv0_diag_sel([prod_list[1]], mps=162)


def check_l1a() -> None:
    """..."""
    date_dir = Path("2024") / "07" / "03"

    # SPEXone L1A science measurements
    prod_list = sorted(SPX_L1A_DIR.glob(date_dir / "PACE_SPEXONE_DARK.*.L1A.nc"))
    print(prod_list[0])
    check_l1a_sci_sel([prod_list[0]])
    check_l1a_sci_sel(prod_list, mps=46)
    check_l1a_sci_sel(prod_list, do_sel=True, mps=46)

    # SPEXone L1A diagnostic measurements
    prod_list = sorted((SPX_L1A_DIR / date_dir).glob("PACE_SPEXONE_CAL.*.L1A.nc"))
    print(prod_list[0])
    check_l1a_diag_sel([prod_list[0]])
    check_l1a_diag_sel(prod_list, mps=162)
    check_l1a_diag_sel(prod_list, do_sel=True, mps=162)


if __name__ == "__main__":
    check_lv0()
    # check_l1a()
