Installing pyspex
=================


Wheels
------
I you have an existing Python (v3.7+) installation, pyspex can be installed
using pip3 from PyPI:

    pip3 install pyspex [--user]


Python Distributions
--------------------
If you use a Python Distribution, the installation of pyspex can be done on
the command line via:

    conda install pytiff [optional]  
    conda install pys5p  
    conda install pyspex

for [Anaconda](https://www.anaconda.com/)/[MiniConda](http://conda.pydata.org/miniconda.html).


Source installation
-------------------
The latest release of pyspex is available from
[gitHub](https://github.com/rmvanhees/pyspex).
Where you can download the source code as a tar-file or zipped archive.
Or you can use git do download the repository:

    git clone https://github.com/rmvanhees/pyspex.git

Before you can install pyspex, you need:

 * Python version 3.7+ with development headers
 * HDF5, installed with development headers
 * netCDF4, installed with development headers

And the following Python modules:

 * setuptools v57+
 * setuptools-scm v6+
 * numpy v1.20+
 * h5py v3.40+
 * netCDF4 v1.5+
 * xarray v0.16+
 * pytiff v0.8+ (optional), requires libtiff5 with development headers
 * pys5p v1.1+

The software is known to work using:

 * HDF5 v1.8.21, netCDF4 v4.7.3 and python-netCDF4 v1.5+
 * HDF5 v1.10+, netCDF4 v4.7.3 or v4.8+ and python-netCDF4 v1.5+
 * HDF5 v1.12+, netCDF4 v4.8+ and python-netCDF4 v1.5+

The actual installation of pyspex should be done via:

    python3 -m build  
    pip3 install dist/pyspex-<version>.whl [--user]

The scripts to convert raw SPEXone data to L1A can be found under:
`/usr/local/bin` or `$USER/.local/bin`.
