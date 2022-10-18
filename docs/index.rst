.. pyspex documentation master file, created by
   sphinx-quickstart on Thu Sep 29 17:24:52 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Package pyspex User Manual
==========================

Python package pyspex contains software to access SPEXone data.
This package contains software scripts and classes to:

  * Read data in CCSDS format from the SPEXone instrument and write the data
    in Level-1A format.
    
  * Read Level-0 data from the SPEXone after integration on the PACE platform
    and write the data in Level-1A in-flight products (science or calibration).

  * Read the SPEXone CKD product

  * Read in-flight PACE engineering and housekeeping (HKT) products.

  * Handle SPEXone binning tables (read, write, visualize).

For more information on PACE mission visit:

  * https://pace.gsfc.nasa.gov

  * https://pace.oceansciences.org/mission.htm

For more information on SPEXone instrument visit:
  
  * https://www.sron.nl/earth-instrument-development/spex/spexone

  * https://pace.oceansciences.org/spexone.htm


Quick-start
-----------

.. toctree::
    :maxdepth: 1

    quick
    build

Tools
-----

.. autosummary::
   
.. toctree::

   tools

Module Documentation
--------------------

.. toctree::
   :maxdepth: 2

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
