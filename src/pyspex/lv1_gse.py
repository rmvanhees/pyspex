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


class LV1gse:
    """
    Python class add EGSE/OGSE data to a SPEXone Level-1A product
    """
    def __init__(self, l1a_file: str) -> None:
        """
        Initialize netCDF4 group 'gse_data' in a SPEXone Level-1 product
        """
        self.fid = Dataset(l1a_file, 'r+')

        if self.fid.groups.get("gse_data"):
            return

        # investigate filename
        parts = l1a_file.split('_')
        act_angle = [float(x.replace('act', ''))
                     for x in parts if x.startswith('act')]
        alt_angle = [float(x.replace('alt', ''))
                     for x in parts if x.startswith('alt')]
        pol_angle = [float(x.replace('pol', ''))
                     for x in parts if x.startswith('pol')]

        # determine viewport: default 0, when all viewports are illuminated
        if alt_angle:
            vp_dict = {'-50.0': 1, '-20.0': 2, '0.0': 4, '20.0': 8, '50.0': 16}

            viewport = vp_dict.get('{:.1f}'.format(alt_angle[0]), 0)
        else:
            vp_dict = {'M50DEG': 1, 'M20DEG': 2, '0DEG': 4, 'P20DEG': 8,
                       'P50DEG': 16}
            vp_parts = parts[2].split('-')
            viewport = vp_dict.get(vp_parts[min(2, len(vp_parts))], 0)

        gid = self.fid.createGroup('/gse_data')
        dset = gid.createVariable('viewport', 'u1')
        dset.long_name = "viewport status"
        dset.standard_name = "status_flag"
        dset.valid_range = np.array([0, 16], dtype='u1')
        dset.flag_values = np.array([0, 1, 2, 4, 8, 16], dtype='u1')
        dset.flag_meanings = "ALL -50deg -20deg 0deg +20deg +50deg"
        dset[:] = viewport

        # gid.FOV_begin = np.nan
        # gid.FOV_end = np.nan
        gid.ACT_rotationAngle = np.nan if not act_angle else act_angle[0]
        gid.ALT_rotationAngle = np.nan if not alt_angle else alt_angle[0]
        # gid.ACT_illumination = np.nan
        # gid.ALT_illumination = np.nan
        gid.AoLP = 0. if not pol_angle else pol_angle[0]
        fields = parts[0].split('-')
        if fields[0] in ('POLARIZED', 'POLARIMETRIC') and fields[1] != 'BKG':
            gid.DoLP = 1.
        else:
            gid.DoLP = 0.

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
        Add/update which viewports are illuminated

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
        Add EGSE parameters

        Parameters
        ----------
        egse_time :  ndarray
        egse_data :  ndarray
        egse_attrs :  dict
        """
        # perform sanity check on EGSE parameters
        self.check_egse(egse_data)

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
        Add data from the reference diode

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
        Add wavelength monitoring data of the Avantas fibre-spectrometer

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

    # def write_attr_fov(self, val_begin: float, val_end: float) -> None:
    #    """
    #    Add field-of-view range as an group attribute
    #
    #    Parameters
    #    ----------
    #    val_begin :  float
    #    val_end :  float
    #    """
    #    self.fid['/gse_data'].FOV_begin = val_begin
    #    self.fid['/gse_data'].FOV_end = val_end

    def write_attr_act(self, angle: float, illumination=None) -> None:
        """
        Add act rotation angle as an group attribute

        Parameters
        ----------
        angle :  float
        illumination :  float
        """
        self.fid['/gse_data'].ACT_rotationAngle = angle
        self.fid['/gse_data'].ACT_illumination = illumination

    def write_attr_alt(self, angle: float, illumination=None) -> None:
        """
        Add altitude rotation angle as an group attribute

        Parameters
        ----------
        angle :  float
        illumination :  float
        """
        self.fid['/gse_data'].ALT_rotationAngle = angle
        self.fid['/gse_data'].ALT_illumination = illumination

    def write_attr_polarization(self, aolp: float, dolp: float) -> None:
        """
        Add polarization parameters AoLP & DoLP as an group attribute

        Parameters
        ----------
        AoLP :  float
        DoLP :  float
        """
        if aolp is not None:
            self.fid['/gse_data'].AoLP = aolp
        if dolp is not None:
            self.fid['/gse_data'].DoLP = dolp

    def check_egse(self, egse_data):
        """
        Check consistency of OGSE/EGSE information during measurement
        """
        for key, fmt in egse_data.dtype.fields.items():
            if fmt[0] == np.uint8:
                res_sanity = (egse_data[key] == egse_data[key][0]).all()
                if not res_sanity:
                    print('[WARNING] ', key, egse_data[key])

        act_angle = self.fid['/gse_data'].ACT_rotationAngle
        if np.isfinite(act_angle):
            if not np.allclose(egse_data['ACT_ANGLE'], act_angle, 1e-2):
                print('[WARNING] ', 'ACT_ANGLE', egse_data['ACT_ANGLE'])

        alt_angle = self.fid['/gse_data'].ALT_rotationAngle
        if alt_angle:
            if not np.allclose(egse_data['ALT_ANGLE'], alt_angle, 1e-2):
                print('[WARNING] ', 'ALT_ANGLE', egse_data['ALT_ANGLE'])


#    def fill_gse(self, reference=None) -> None:
#        """
#        Write EGSE/OGSE data to a SPEXone Level-1A product
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
