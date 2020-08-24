Version 1.1.5
=============
 * spx1_ccsds2l1a.py: you can now combine multiple files with CCSDS packages from the same measurement into one Level-1A product
 * CCSDS: improved verbosity and ignore incomplete images
 
Version 1.1.4
=============
  * Method DEMio::read_hdr can now return the raw detector header register data, or return this info written into a MPS record
  * Fixed: replaced depreciated pyspex.lib.sw_version with pyspex.version

Version 1.1.3
=============

 * Fixed several bugs introduced with the new LV1io classes
 * Removed obsolete modules l1a_h5.py and mps_def.py
 * Renamed lib.sw_version.py to version.py
 * Rearranged source tree to comply with PIP 517, 518 (requires: setuptools 42 or later)
 * More work on package setup, many bugs and typo's fixed
 * INSTALL.md: updated installation instructions
 * Updated Copyright line in all modules
 * Updated LICENSE file
 * Added ChangeLog.md
