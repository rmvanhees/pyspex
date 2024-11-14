# Release Notes for pyspex v1.4.15

## API changes

 * Extended flagging of HKT data [monitoring, only]

## File format changes

 * The global and dataset attributes of the L1A products have been modified to comform
   to the NASA/PACE conventions

# Release Notes for pyspex v1.4.14

## API changes

 * Code compatible with numpy v2+
 * Improved annotations of functions and variables
 * class L1Aio:
   * Replaced method `fill_global_attrs()` by private method: `_fill_attr()`
   * Added dataset `/image_attributes/timedelta_centre`
 * class HKTio:
   * Added keyword `add_coverage_quality` to method `add_nav()`
 * class HKtlm:
   * Fixed bugs in method `sel()`
 * class SCItlm:
   * Fixed bugs in methods `copy()` and `sel()`
   * Improved implementation to calculate timestamps of the data packages
   * Fixed roundoff error in method `subsec2musec()`
   * Methods `exposure_time()` and `frame_period()` can return scalar or ndarray
 * class SPXtlm:
   * Removed method `l1a_file()`
   * Improved timestamps of the science data

## L1A processing configuration

 * L0->L1A configuration file:
   * usage of keyword `file_version` is obsolete 
   * the data type of keyword `processing_version` is changed from str to int

## File format changes

 * Added dataset `timedelta_centre` which contains the offset from the start of the
   integration time to the center of the integration time
 * Instrument temperatures are provided in degree Celsius, instead of Kelvin
 * Added field `MPS_ACT_STATUS` in the telemetry data
 * Global attributes updated
   * dropped the timezone offset from the date/time fields, since UTC is implicit
   * added DOI of the SPEXone L1A product 
 * Added group `/processing_control` providing `input_parameters`
