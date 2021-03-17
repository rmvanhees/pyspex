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


# --------------------------------------------------
class LV1gse:
    """
    Python class add EGSE/OGSE data to a SPEXone Level-1A product
    """
    def __init__(self, l1a_file: str) -> None:
        """
        Initialize netCDF4 group 'gse_data' in a SPEXone Level-1 product

        Attributes
        ----------
        fid : netCDF4::Dataset
            netCDF4 Dataset instance.

        Methods
        -------
        check_egse(egse_data)
           Check consistency of OGSE/EGSE information during measurement.
        set_attr(name, value)
           Add/update an attribute of the group 'gse_data'
        write_attr_act(angle: float, illumination)
           Add ACT rotation angle as an group attribute.
        write_attr_alt(angle: float, illumination)
           Add altitude rotation angle as an group attribute.
        write_attr_polarization(aolp: float, dolp: float)
           Add polarization parameters AoLP & DoLP as group attributes.
        write_viewport(viewport)
           Add/update which viewports are illuminated.
        write_egse(egse_time, egse_data, egse_attrs)
           Add EGSE parameters.
        write_data_stimulus(xds_signal)
           Add wavelength and signal of data stimulus.
        write_reference_diode(ref_time, ref_data, ref_attrs)
           Add data measured by the reference diode during the measurement.
        write_reference_signal(xds_signal)
           Add reference signal and its error.
        write_wavelength_monitor(xds_wavelength)
           Add wavelength monitoring data of the Avantas fibre-spectrometer.
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
                     for x in parts if x.startswith('pol') and x != 'polcal']

        # determine viewport: default 0, when all viewports are illuminated
        if alt_angle:
            vp_dict = {'-50.0': 1, '-20.0': 2, '0.0': 4, '20.0': 8, '50.0': 16}

            viewport = vp_dict.get('{:.1f}'.format(alt_angle[0]), 0)
        else:
            vp_dict = {'M50DEG': 1, 'M20DEG': 2, '0DEG': 4, 'P20DEG': 8,
                       'P50DEG': 16}
            vp_str = [x for x in parts[2].split('-') if x.endswith('DEG')]

            viewport = vp_dict.get(vp_str[0], 0) if vp_str else 0

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

    def check_egse(self, egse_data) -> None:
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
        if np.isfinite(alt_angle):
            if not np.allclose(egse_data['ALT_ANGLE'], alt_angle, 1e-2):
                print('[WARNING] ', 'ALT_ANGLE', egse_data['ALT_ANGLE'])

    def set_attr(self, name: str, value) -> None:
        """
        Add attribute to group 'gse_data'

        Parameters
        ----------
        name: str
        value: anything(?)
        """
        self.fid['/gse_data'].setncattr(name, value)

    def write_attr_act(self, angle: float, illumination=None) -> None:
        """
        Add act rotation angle as an group attribute

        Parameters
        ----------
        angle :  float
        illumination :  float
        """
        self.fid['/gse_data'].ACT_rotationAngle = angle
        if illumination is not None:
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
        if illumination is not None:
            self.fid['/gse_data'].ALT_illumination = illumination

    def write_attr_polarization(self, aolp: float, dolp: float) -> None:
        """
        Add polarization parameters AoLP & DoLP as group attributes

        Parameters
        ----------
        AoLP :  float
        DoLP :  float
        """
        if aolp is not None:
            self.fid['/gse_data'].AoLP = aolp
        if dolp is not None:
            self.fid['/gse_data'].DoLP = dolp

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

    def write_viewport(self, viewport: int) -> None:
        """
        Add/update which viewports are illuminated

        Parameters
        ----------
        viewport :  int
        """
        self.fid['/gse_data/viewport'][:] = viewport

    def write_reference_signal(self, signal, spread) -> None:
        """
        Write reference detector signal and variance

        Parameters
        ----------
        signal :  float
           Median signal level measured during the measurement
        spread :  float
           Variance of the signal level measured during the measurement

        Notes
        -----
        Used for non-linearity measurements.
        """
        gid = self.fid['/gse_data']
        gid.Illumination_level = 5e9 / 1.602176634 * signal

        dset = gid.createVariable('reference_signal', 'f8', ())
        dset.long_name = "biweight median of reference-detector signal"
        dset.comment = "t_sat = min(2.28e-9 / S_reference, 30)"
        dset.units = 'A'
        dset[:] = signal

        dset = gid.createVariable('reference_error', 'f8', ())
        dset.long_name = "biweight spread of reference-detector signal"
        dset.units = 'A'
        dset[:] = spread

    def write_reference_diode(self, ref_time, ref_data, ref_attrs) -> None:
        """
        Add data measured by the reference diode during the measurement

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
        dset.units = 'seconds since 1970-01-01T00:00:00+00:00'
        dset.comment = 'Generated on SRON clean-room tablet'
        dset[:] = ref_time

        ref_t = gid.createCompoundType(ref_data.dtype, 'ref_dtype')
        dset = gid.createVariable('ref_diode', ref_t, ('time',))
        dset.setncatts(ref_attrs)
        dset[:] = ref_data

    def write_data_stimulus(self, xds_stimulus) -> None:
        """
        Add wavelength and signal of data stimulus

        Parameters
        ----------
        xds_stimulus :  xarray::Dataset
           Contains xarray::DataArrays 'wavelength' and 'signal'
        """
        xds_stimulus.to_netcdf(self.fid.filepath(), mode='a', group='gse_data')

    def write_wavelength_monitor(self, xds_wav_mon) -> None:
        """
        Add wavelength monitoring data of the Avantas fibre-spectrometer

        Parameters
        ----------
        xds_wav_mon :  xarray::Dataset
           Contains xarray::DataArrays 'signal', 'wavelength', 'wav_time'
        """
        xds_wav_mon.to_netcdf(self.fid.filepath(), mode='a',
                              group='/gse_data/WaveMonitor')

        # gid = self.fid.createGroup('/gse_data/WaveMonitor')
        # _ = gid.createDimension('time', len(wav_time))
        # dset = gid.createVariable('time', 'f8', ('time',))
        # dset.units = 'seconds since 1970-01-01T00:00:00+00:00'
        # dset.comment = 'Generated on SRON clean-room tablet'
        # dset[:] = wav_time
        # _ = gid.createDimension('wavelength', len(wav_wv))
        # dset = gid.createVariable('wavelength', 'f4', ('wavelength',))
        # dset.longname = 'wavelength grid'
        # dset.comment = 'wavelength annotation of fibre spectrometer'
        # dset[:] = wav_wv
        # dset = gid.createVariable('t_intg', 'i2', ('time',))
        # dset.long_name = 'Integration time'
        # dset.units = 'ms'
        # dset[:] = wav_intg
        # dset = gid.createVariable('n_avg', 'i2', ('time',))
        # dset.long_name = 'Averaging number'
        # dset[:] = wav_avg_num
        # dset = gid.createVariable('wav_mon', 'f4', ('time', 'wavelength'))
        # dset.long_name = 'wavelength-monitor spectra'
        # dset.comment = 'Avantes fibre spectrometer'
        # dset[:] = wav_signal
