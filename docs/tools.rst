Tools
=====

The package pyspex comes with several tools, these are desccribed below.

Script: spx1_level01a.py
------------------------
Write raw data (Level-0) from the SPEXone instrument to a Level-1A product.

Usage::

  spx1_level01a  [-h] [--debug] [--dump] [--verbose]
		 [--outdir OUTDIR]
		 [[--spex_lv0 SPEX_LV0 [SPEX_LV0 ...]]
		  [--yaml_fl YAML_FL]]


Positional arguments::
  
   SPX1_LV0   Provide one or more input files:
         - raw: if you provide only the file with extension '.H'
	        then all files of the same measurement with science and
		house-keeping data are collected, else you have to provide
		these files yourself;
         - st3: in general all measurement data are collected in one file;
         - dsb: please provide all files with the data of one measurement.

Options::

  -h, --help            show this help message and exit
  --debug               be more verbose and do not write any output products
  --dump                dump CCSDS packet headers in ASCII
  --verbose             be verbose
  --outdir OUTDIR       Directory to store the Level-1A product
  --spex_lv0 SPEX_LV0 [SPEX_LV0 ...]
  --yaml_fl YAML_FL


Return values::

  2      Failed to parse command-line parameters.
  100    Input file not found error.
  101    Input file not recognized as a SPEXone level-0 product.
  110    Corrupted SPEXone level-0 data.
  130    Output file permission error.
  131    Output file write error from netCDF/HDF5 library.

Content of YAML file::

  # define output directory, CWD when empty
  outdir: CWD
  # define name of output file, will be generated automatically when empty
  outfile: ''
  # define file-version as nn, neglected when outfile not empty
  file_version: 1
  # flag to indicate measurements taken in eclipse or day-side
  eclipse: True
  # provide list, directory, file-glob or empty
  hkt_list: ''
  # must be a list, directory or glob. Fails when empty
  l0_list: L0/SPX0000000??.spx

Examples::
  
  
Script: spx1_add_egse2l1a.py
----------------------------


Script: spx1_add_ogse2l1a.py
----------------------------


