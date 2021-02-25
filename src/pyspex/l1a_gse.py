"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python class add EGSE/OGSE data to a SPEXone Level-1A product

Copyright (c) 2021 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import numpy as np

from netCDF4 import Dataset


class L1Agse:
    """
    Python class add EGSE/OGSE data to a SPEXone Level-1A product
    """
    def __init__(self, l1a_file: str) -> None:
        """
        Initialize netCDF4 group 'gse_data' in a SPEXone Level-1 product
        """
        self.fid = Dataset(l1a_file, 'r+')

        if "gse_data" not in self.fid.groups.values():
            gid = self.fid.createGroup('/gse_data')

            dset = gid.createVariable('viewport', 'u1')
            dset.long_name = "viewport status"
            dset.standard_name = "status_flag"
            dset.valid_range = np.array([0, 16], dtype='u1')
            dset.flag_values = np.array([0, 1, 2, 4, 8, 16], dtype='u1')
            dset.flag_meanings = "ALL -50deg -20deg 0deg +20deg +50deg"
            dset[:] = 0   # initialize to default value: all viewports used

            gid.FOV_begin = np.nan
            gid.FOV_end = np.nan
            gid.ACT_rotationAngle = np.nan
            gid.ALT_rotationAngle = np.nan
            gid.ACT_illumination = np.nan
            gid.ALT_illumination = np.nan
            gid.DoLP = 0.
            gid.AoLP = 0.

    def __iter__(self) -> None:
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __enter__(self):
        """
        method called to initiate the context manager
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """
        method called when exiting the context manager
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self) -> None:
        """
        Close all resources (currently a placeholder function)
        """
        if self.fid is None:
            return

        self.fid.close()
        self.fid = None

    def write_viewport(self, viewport: int) -> None:
        """
        Parameters
        ----------
        viewport :  int
        """
        self.fid['/gse_data/viewport'][:] = viewport

    def write_data_stimulus(self, wavelength, signal, units) -> None:
        """
        Add wavelength and signal of data stimulus

        Parameters
        ----------
        wavelength :  ndarray
        signal :  ndarray
        units :  list
        """
        gid = self.fid['/gse_data']
        _ = gid.createDimension('wavelength', len(wavelength))
        dset = gid.createVariable('wavelength', 'f8', ('wavelength',))
        dset.long_name = 'wavelength of stimulus'
        dset.units = units[0]
        dset[:] = wavelength

        dset = gid.createVariable('signal', 'f8', ('wavelength',))
        dset.long_name = 'signal of stimulus'
        dset.units = units[1]
        dset[:] = signal

    def write_egse(self, egse_time, egse_data, egse_attrs: dict) -> None:
        """
        Parameters
        ----------
        egse_time :  ndarray
        egse_data :  ndarray
        egse_attrs :  dict
        """
        gid = self.fid['/gse_data']
        _ = gid.createDimension('time', len(egse_data))
        dset = gid.createVariable('time', 'f8', ('time',))
        dset[:] = egse_time

        egse_t = gid.createCompoundType(egse_data.dtype, 'egse_dtype')
        dset = gid.createVariable('egse', egse_t, ('time',))
        dset.setncatts(egse_attrs)
        dset[:] = egse_data

    def write_reference_diode(self, ref_time, ref_data,
                              ref_attrs: dict) -> None:
        """
        Parameters
        ----------
        ref_time :  ndarray
        ref_data :  ndarray
        ref_attrs :  dict
           fid['/ReferenceDiode/ref_diode'].__dict__
        """
        gid = self.fid.createGroup('/gse_data/ReferenceDiode')
        _ = gid.createDimension('time', len(ref_data))
        dset = gid.createVariable('time', 'f8', ('time',))
        dset[:] = ref_time

        ref_t = gid.createCompoundType(ref_data.dtype, 'ref_dtype')
        dset = gid.createVariable('ref_diode', ref_t, ('time',))
        dset.setncatts(ref_attrs)
        dset[:] = ref_data

    def write_wavelength_monitor(self, wav_time, wav_intg, wav_avg_num,
                                 wav_wv, wav_signal) -> None:
        """
        Parameters
        ----------
        wav_time :  ndarray
        wav_intg :  ndarray
        wav_avg_num :  ndarray
        wav_wv :  ndarray
        wav_signal :  ndarray
        """
        gid = self.fid.createGroup('/gse_data/WaveMonitor')

        _ = gid.createDimension('time', len(wav_time))
        dset = gid.createVariable('time', 'f8', ('time',))
        dset.units = 'seconds since 1970-01-01T00:00:00+00:00'
        dset.comment = 'Generated on SRON clean-room tablet'
        dset[:] = wav_time

        _ = gid.createDimension('wavelength', len(wav_wv))
        dset = gid.createVariable('wavelength', 'f4', ('wavelength',))
        dset.longname = 'wavelength grid'
        dset.comment = 'wavelength annotation of fibre spectrometer'
        dset[:] = wav_wv

        dset = gid.createVariable('t_intg', 'i2', ('time',))
        dset.long_name = 'Integration time'
        dset.units = 'ms'
        dset[:] = wav_intg

        dset = gid.createVariable('n_avg', 'i2', ('time',))
        dset.long_name = 'Averaging number'
        dset[:] = wav_avg_num

        dset = gid.createVariable('wav_mon', 'f4', ('time', 'wavelength'))
        dset.long_name = 'wavelength-monitor spectra'
        dset.comment = 'Avantes fibre spectrometer'
        dset[:] = wav_signal

    def write_attr_fov(self, val_begin: float, val_end: float) -> None:
        """
        Parameters
        ----------
        val_begin :  float
        val_end :  float
        """
        self.fid['/gse_data'].FOV_begin = val_begin
        self.fid['/gse_data'].FOV_end = val_end

    def write_attr_act(self, angle: float, illumination=None) -> None:
        """
        Parameters
        ----------
        angle :  float
        illumination :  float
        """
        self.fid['/gse_data'].ACT_rotationAngle = angle
        self.fid['/gse_data'].ACT_illumination = illumination

    def write_attr_alt(self, angle: float, illumination=None) -> None:
        """
        Parameters
        ----------
        angle :  float
        illumination :  float
        """
        self.fid['/gse_data'].ALT_rotationAngle = angle
        self.fid['/gse_data'].ALT_illumination = illumination

    def write_attr_polarization(self, aolp: float, dolp: float) -> None:
        """
        Parameters
        ----------
        AoLP :  float
        DoLP :  float
        """
        if aolp is not None:
            self.fid['/gse_data'].AoLP = aolp
        if dolp is not None:
            self.fid['/gse_data'].DoLP = dolp


#    def fill_gse(self, reference=None) -> None:
#        """
#        Write EGSE/OGSE data to L1A product
#
#        Parameters
#        ----------
#        reference : dict, optional
#           biweight value and spread of the signal measured during the
#           measurement by a reference detector.
#           Expected dictionary keys: 'value', 'error'
#        """
#        if reference is not None:
#            dset = self.fid.createVariable('/gse_data/reference_signal',
#                                           'f8', ())
#            dset.long_name = "biweight median of reference-detector signal"
#            dset.comment = "t_sat = min(2.28e-9 / S_reference, 30)"
#            dset.units = 'A'
#            dset[:] = reference['value']
#            self.set_attr('Illumination_level',
#                          reference['value'] * 5e9 / 1.602176634,
#                          ds_name='gse_data')
#
#            dset = self.fid.createVariable('/gse_data/reference_error',
#                                           'f8', ())
#            dset.long_name = "biweight spread of reference-detector signal"
#            dset.units = 'A'
#            dset[:] = reference['error']
