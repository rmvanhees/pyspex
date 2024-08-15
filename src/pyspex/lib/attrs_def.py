#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2024 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""Provides the common global-attributes for SPEXone Level-1 products."""

from __future__ import annotations

__all__ = ["attrs_def"]

import datetime as dt
import sys
from pathlib import Path

from . import pyspex_version


# - main functions --------------------------------
def attrs_def(inflight: bool = True, origin: str | None = None) -> dict:
    """Define global attributes of a SPEXone Level-1A product.

    Parameters
    ----------
    inflight : bool, default=True
       Flag for in-flight or on-ground products
    origin : str
       Product origin: 'SRON' or 'NASA'

    Returns
    -------
    dict
       Global attributes for a Level-1A product

    """
    if origin is None:
        origin = "NASA" if inflight else "SRON"

    res = {
        "creator_name": "NASA/GSFC",
        "creator_email": "data@oceancolor.gsfc.nasa.gov",
        "creator_url": "http://oceancolor.gsfc.nasa.gov",
        "institution": (
            "NASA Goddard Space Flight Center, Ocean Biology Processing Group"
        ),
        "publisher_name": "NASA/GSFC",
        "publisher_email": "data@oceancolor.gsfc.nasa.gov",
        "publisher_url": "http://oceancolor.gsfc.nasa.gov",
        "standard_name_vocabulary": "CF Standard Name Table v79",
        "keyword_vocabulary": (
            "NASA Global Change Master Directory (GCMD) Science Keywords"
        ),
        "license": (
            "http://science.nasa.gov/earth-science/"
            "earth-science-data/data-information-policy/"
        ),
        "naming_authority": "gov.nasa.gsfc.sci.oceancolor",
        "project": "PACE Project",
        "conventions": "CF-1.8 ACDD-1.3",
        "title": "PACE SPEXone Level-1A Data",
        "instrument": "SPEXone",
        "platform": "PACE",
        "stdname_vocabulary": "NetCDF Climate and Forecast (CF) Metadata Convention",
        "processing_level": "L1A",
        "cdm_data_type": "swath" if inflight else "granule",
        "product_name": None,
        "start_direction": "Ascending" if inflight else None,
        "end_direction": "Ascending" if inflight else None,
        "time_coverage_start": "yyyy-mm-ddTHH:MM:DD",
        "time_coverage_end": "yyyy-mm-ddTHH:MM:DD",
        "processing_version": 1,
        "identifier_product_doi_authority": "http://dx.doi.org/",
        "identifier_product_doi": "10.5067/PACE/SPEXONE/L1A/SCI/2",
        "date_created": dt.datetime.now(dt.UTC)
        .replace(tzinfo=None)
        .isoformat(timespec="milliseconds"),
        "software_name": f"{Path(sys.argv[0]).name}",
        "software_description": "SPEXone L0-L1A processor (SRON)",
        "software_url": "https://github.com/rmvanhees/pyspex",
        "software_version": pyspex_version(),
        "software_doi": "https://doi.org/10.5281/zenodo.5705691",
        "history": " ".join(sys.argv),
    }

    if origin == "SRON":
        res["institution"] = "SRON Netherlands Institute for Space Research"
        res["creator_name"] = "SRON/Earth"
        res["creator_email"] = "SPEXone-MPC@sron.nl"
        res["creator_url"] = "https://www.sron.nl/missions-earth/pace-spexone"
        res["publisher_name"] = "SRON/Earth"
        res["publisher_email"] = "SPEXone-MPC@sron.nl"
        res["publisher_url"] = "https://www.sron.nl/missions-earth/pace-spexone"

    return res
