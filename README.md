# pySpex
[![image](https://img.shields.io/pypi/v/pyspex.svg?label=release)](https://github.com/rmvanhees/pyspex/)
[![image](https://img.shields.io/pypi/l/pyspex.svg)](https://github.com/rmvanhees/pyspex/LICENSE)
[![image](https://img.shields.io/pypi/dm/pyspex.svg)](https://pypi.org/project/pyspex/)
[![image](https://img.shields.io/pypi/status/pyspex.svg?label=status)](https://pypi.org/project/pyspex/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5705691.svg)](https://doi.org/10.5281/zenodo.5705691)

Python package `pyspex` contains software to access SPEXone data.
This package contains software scripts and classes to:
* Read data in CCSDS format from the SPEXone instrument and write the data in Level-1A format.
* Read Level-0 data from the SPEXone after integration on the PACE platform and write the data in Level-1A in-flight science and calibration products.
* Read SPEXone CKD product
* Read in-flight PACE engineering and housekeeping (HKT) products.
* Handle SPEXone binning tables (read, write, visualize).

For more information on PACE mission visit:
* https://pace.gsfc.nasa.gov
* https://pace.oceansciences.org/mission.htm

For more information on SPEXone instrument visit:
* https://www.sron.nl/earth-instrument-development/spex/spexone
* https://pace.oceansciences.org/spexone.htm

## Documentation
Online documentation is available from [Read the Docs](https://pyspex.readthedocs.io).

## Installation
The `pyspex` package requires Python3.8+ and the Python packages: h5py, netCDF4, numpy and xarray.

Installation instructions are provided on [Read the Docs](https://pyspex.readthedocs.io/en/latest/build.html) or in the INSTALL file.
