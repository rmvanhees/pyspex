#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Provides the common global-attributes for SPEXone Level-1 products.
"""
__all__ = ['attrs_def']

from datetime import datetime, timezone

from pyspex import version


# - main functions --------------------------------
def attrs_def(level: str, inflight=True, origin=None) -> dict:
    """
    Defines all global attributes for SPEXone Level-1 products.

    Parameters
    ----------
    level : str
       Product processing level 'L1A', 'L1B' or 'L1C'
    inflight : bool
       Flag for in-flight or on-ground products
    origin : str
       Product origin: 'SRON' or 'NASA'

    Returns
    -------
    dict
       Global attributes for a Level-1A product
    """
    if origin is None:
        origin = 'NASA' if inflight else 'SRON'

    res = {
        "title": f"PACE SPEX Level-{level[1:]} data",
        "instrument": "SPEX",
        "processing_version": "V1.0",
        "conventions": "CF-1.6",
        "institution": ("NASA Goddard Space Flight Center,"
                        " Ocean Biology Processing Group"),
        "license": ("http://science.nasa.gov/earth-science/"
                    "earth-science-data/data-information-policy/"),
        "naming_authority": "gov.nasa.gsfc.sci.oceancolor",
        "keyword_vocabulary": ("NASA Global Change Master Directory (GCMD)"
                               " Science Keywords"),
        "stdname_vocabulary": ("NetCDF Climate and Forecast (CF)"
                               " Metadata Convention"),
        "creator_name": "NASA/GSFC",
        "creator_email": "data@oceancolor.gsfc.nasa.gov",
        "creator_url": "http://oceancolor.gsfc.nasa.gov",
        "project": "PACE Project",
        "publisher_name": "NASA/GSFC",
        "publisher_email": "data@oceancolor.gsfc.nasa.gov",
        "publisher_url": "http://oceancolor.gsfc.nasa.gov",
        "processing_level": level,
        "CDM_data_type": "swath" if inflight else "on-ground calibration",
        "orbit_number": -1,
        "history": None,
        "CDL_version_date": "2020-02-20",
        "product_name": None,
        "start_direction": "Ascending" if inflight else None,
        "end_direction": "Ascending" if inflight else None,
        "time_coverage_start": "yyyy-mm-ddTHH:MM:DD",
        "time_coverage_end": "yyyy-mm-ddTHH:MM:DD",
        "date_created": datetime.now(timezone.utc).isoformat(
            timespec='milliseconds'),
        "software_name": 'https://github.com/rmvanhees/pyspex',
        "software_version": version.get(),
        "sun_earth_distance": None,
        "terrain_data_source": None,
        "spectral_response_function": None,
        "systematic_uncertainty_model": None,
        "nadir_bin": None,
        "bin_size_at_nadir": None
    }

    if origin == 'SRON':
        res['title'] = f"SPEXone Level-{level[1:]} data"
        res['instrument'] = "SPEXone"
        res['institution'] = "SRON Netherlands Institute for Space Research"
        res['creator_name'] = "SRON/Earth"
        res['creator_email'] = "SPEXone-MPC@sron.nl"
        res['creator_url'] = "https://www.sron.nl/missions-earth/pace-spexone"
        res['publisher_name'] = "SRON/Earth"
        res['publisher_email'] = "SPEXone-MPC@sron.nl"
        res['publisher_url'] = "https://www.sron.nl/missions-earth/pace-spexone"

    return res
