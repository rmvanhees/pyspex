# pyspex
[![PyPI Latest Release](https://img.shields.io/pypi/v/pyspex.svg)](https://pypi.org/project/pyspex/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5705691.svg)](https://doi.org/10.5281/zenodo.5705691)
[![Package Status](https://img.shields.io/pypi/status/pyspex.svg)](https://pypi.org/project/pyspex/)
[![License](https://img.shields.io/pypi/l/pyspex.svg)](https://github.com/rmvanhees/pyspex/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/pyspex?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/pyspex/)

Python package pyspex contains software to access SPEXone data.
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


## Installation
The module pyspex requires Python3.8+ and Python modules: h5py, netCDF4, numpy and xarray.

Installation instructions are provided in the INSTALL file.
