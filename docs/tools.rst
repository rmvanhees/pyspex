Tools
=====

The package pyspex comes with several tools, these are desccribed below.

Script: spx1_level01a.py
------------------------
Write raw data (Level-0) from the SPEXone instrument to a Level-1A product.

Usage::

  spx1_level01a.py [-h] [--verbose] [--debug] [--dump] 
		   [--datapath DATAPATH]
                   [--file_version FILE_VERSION]
		   [--file_format {raw,st3,dsb}]
                   [--select {binned,fullFrame}]
                   [--pace_hkt PACE_HKT [PACE_HKT ...]]
		   SPX1_LV0 [SPX1_LV0 ...]

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
  --verbose             be verbose
  --debug               be more verbose
  --dump                dump CCSDS packet headers in ASCII
  --select {binned,fullFrame}
                        Select "binned" or "fullFrame" detector-readouts
  --datapath DATAPATH   Directory to store the Level-1A product
  --file_version FILE_VERSION
                        Provide file version number of level-1A product
  --file_format {raw,st3,dsb}
                        Provide data format of the input file(s):
                        - raw: CCSDS packages (a.o. ambient calibration);
                        - st3: CCSDS packages with ITOS and spacewire headers;
                        - dsb: files recorded on the observatory data storage
			       board;
                        - default: determine file format from input files.
  --pace_hkt PACE_HKT [PACE_HKT ...]
                        names of PACE HKT products with navigation data

  
Script: spx1_add_egse2l1a.py
----------------------------


Script: spx1_add_ogse2l1a.py
----------------------------


