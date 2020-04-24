Installing pyspex
=================

You can only install pyspex from source.

Download
--------
The latest release of pyspex is available from
[gitHub](https://github.com/rmvanhees/pyspex).
Here you can download the source code as an tar-file or zipped archive.
Alternativelly, you can use git do download the repository:

 * `git clone https://github.com/rmvanhees/pyspex.git`

Before you can install pyspex, you need:

 * Python version 3.7+ with development headers
 * HDF5, installed with development headers
 * netCDF4, installed with development headers

And the following Python modules (using pip3):

 * setuptools-scm v3+
 * numpy v1.17+
 * h5py v2.10
 * netCDF4 v1.5.3+
 * pytiff v0.8+ (optional), requires libtiff5 with development headers

The software is known to work using:

 * HDF5 v1.8.21, netCDF4 v4.7.3 and python-netCDF4 v1.5.3
 * HDF5 v1.10.6, netCDF4 v4.7.3 and python-netCDF4 v1.5.3
 * HDF5 v1.12.0, netCDF4 v4.7.4 and python-netCDF4 v1.5.4


Install
-------
Once you have satisfied the requirements, simply run as administrator:

 * `python3 setup.py install`

or as user:

 * `python3 setup.py install --user`

The scripts to convert raw SPEXone data to L1A are installed under
`/usr/local/bin` or `$USER/.local/bin`.

