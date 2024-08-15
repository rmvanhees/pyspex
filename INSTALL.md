Installing pyspex
=================


Wheels
------
I you have an existing Python (v3.8+) installation, pyspex can be installed
using pip from PyPI:

    pip3 install pyspex [--user]


Python Distributions
--------------------
If you use a Python Distribution, the installation of pyspex can be done on
the command line via:

    conda install pytiff [optional]
    conda install moniplot
    conda install pyspex

for [Anaconda](https://www.anaconda.com/)/[MiniConda](http://conda.pydata.org/miniconda.html).


Install from source
-------------------
The latest release of pyspex is available from
[gitHub](https://github.com/rmvanhees/pyspex).
Where you can download the source code as a tar-file or zipped archive.
Or you can use git do download the repository:

    git clone https://github.com/rmvanhees/pyspex.git

Before you can install pyspex, you need:

 * Python version 3.8+ with development headers
 * HDF5, installed with development headers
 * netCDF4, installed with development headers

And have the following Python modules available:

 * setuptools v57+
 * setuptools-scm v6+
 * numpy v1.19+ or v2+
 * h5py v3.8+
 * netCDF4 v1.5+
 * xarray v0.20+
 * moniplot v0.2+
 * pytiff v0.8+ (optional), requires libtiff5 with development headers

The software is known to work using:

 * HDF5 v1.14+, netCDF4 v4.9+ (preferred configuration)
 * HDF5 v1.12+, netCDF4 v4.8+
 * HDF5 v1.10+, netCDF4 v4.7.3
 * HDF5 v1.8.21, netCDF4 v4.7.3

You can install pyspex once you have satisfied the requirements listed above.
Run at the top of the source tree:

	pip install [--user] .
	
or

    python3 -m build
    pip3 install [--user] dist/pyspex-<version>.whl

The Python scripts can be found under `/usr/local/bin` or `$USER/.local/bin`.
