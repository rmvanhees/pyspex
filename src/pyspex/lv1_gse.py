#
# This file is part of pyspex
#
# https://github.com/rmvanhees/pyspex.git
#
# Copyright (c) 2019-2022 SRON - Netherlands Institute for Space Research
#    All Rights Reserved
#
# License:  BSD-3-Clause
"""
Contains the class `LV1gse` which checks EGSE/OGSE data and/or writes it to
a SPEXone Level-1A product.
"""
__all__ = ['LV1gse']

import numpy as np

from netCDF4 import Dataset


# --------------------------------------------------
class LV1gse:
    """Adds EGSE/OGSE data to a SPEXone Level-1A product.

    Parameters
    ----------
    l1a_file: str
        Name of the Level-1A product
    """
    def __init__(self, l1a_file: str) -> None:
        """Initialize netCDF4 group 'gse_data' in a SPEXone Level-1 product.
        """
        self.fid = Dataset(l1a_file, 'r+')

        if self.fid.groups.get("gse_data"):
            return

        # investigate filename
        parts = l1a_file.split('_')
        if len(parts) > 2 and parts[0] == 'SPX1':
            parts = parts[2:]
        msmt_fields = parts[0].split('-')
        background = 'BKG' in msmt_fields
        act_angle = [float(x.replace('act', ''))
                     for x in parts if x.startswith('act')]
        alt_angle = [float(x.replace('alt', ''))
                     for x in parts if x.startswith('alt')]
        pol_angle = [float(x.replace('pol', ''))
                     for x in parts if x.startswith('pol') and x != 'polcal']
        gp1_angle = [float(x.replace('glass', ''))
                     for x in parts if x.startswith('glass')]
        gp1_offs = 5.634375
        # gp2_offs = 5.09625

        # determine viewport: default 0, when all viewports are illuminated
        if alt_angle:
            vp_angle = np.array([-50., -20, 0, 20, 50])
            vp_diff = np.abs(vp_angle - alt_angle[0])
            if vp_diff.min() > 6:
                viewport = 0
            else:
                viewport = 2 ** np.argmin(vp_diff)
        else:
            viewport = 0

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
        if not background and msmt_fields[0] in ('POLARIZED', 'POLARIMETRIC'):
            gid.DoLP = 1.
            if gp1_angle:
                gid.GP1_angle = gp1_angle[0] + gp1_offs
        else:
            gid.DoLP = 0.

    def __iter__(self) -> None:
        for attr in sorted(self.__dict__):
            if not attr.startswith("__"):
                yield attr

    def __enter__(self):
        """Method called to initiate the context manager.
        """
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Method called when exiting the context manager.
        """
        self.close()
        return False  # any exception is raised by the with statement.

    def close(self) -> None:
        """Close all resources (currently a placeholder function).
        """
        if self.fid is None:
            return

        self.fid.close()
        self.fid = None

    def check_egse(self, egse_data) -> None:
        """Check consistency of OGSE/EGSE information during measurement.
        """
        for key, fmt in egse_data.dtype.fields.items():
            if fmt[0] == np.uint8:
                res_sanity = (egse_data[key] == egse_data[key][0]).all()
                if not res_sanity:
                    print(f'[WARNING]: {key}={egse_data[key]}')

        act_angle = self.fid['/gse_data'].ACT_rotationAngle
        if np.isfinite(act_angle):
            if not np.allclose(egse_data['ACT_ANGLE'], act_angle, 1e-2):
                print(f'[WARNING]: ACT_ANGLE={egse_data["ACT_ANGLE"]}')

        alt_angle = self.fid['/gse_data'].ALT_rotationAngle
        if np.isfinite(alt_angle):
            if not np.allclose(egse_data['ALT_ANGLE'], alt_angle, 1e-2):
                print(f'[WARNING]: ALT_ANGLE={egse_data["ALT_ANGLE"]}')

    def set_attr(self, name: str, value) -> None:
        """Add attribute to group 'gse_data'.

        Parameters
        ----------
        name: str
        value: anything(?)
        """
        self.fid['/gse_data'].setncattr(name, value)

    def write_attr_act(self, angle: float, illumination=None) -> None:
        """Add act rotation angle as an group attribute.

        Parameters
        ----------
        angle :  float
        illumination :  float
        """
        self.fid['/gse_data'].ACT_rotationAngle = angle
        if illumination is not None:
            self.fid['/gse_data'].ACT_illumination = illumination

    def write_attr_alt(self, angle: float, illumination=None) -> None:
        """Add altitude rotation angle as an group attribute.

        Parameters
        ----------
        angle :  float
        illumination :  float
        """
        self.fid['/gse_data'].ALT_rotationAngle = angle
        if illumination is not None:
            self.fid['/gse_data'].ALT_illumination = illumination

    def write_attr_polarization(self, aolp: float, dolp: float) -> None:
        """Add polarization parameters AoLP & DoLP as group attributes.

        Parameters
        ----------
        AoLP :  float
           Angle of linear polarization
        DoLP :  float
           Degree of linear polarization
        """
        if aolp is not None:
            self.fid['/gse_data'].AoLP = aolp
        if dolp is not None:
            self.fid['/gse_data'].DoLP = dolp

    def write_egse(self, egse_time, egse_data, egse_attrs: dict) -> None:
        """Add EGSE parameters.

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
        dset.long_name = egse_attrs['long_name']
        dset.fields = egse_attrs['fields']
        dset.units = egse_attrs['units']
        dset.comment = egse_attrs['comment']
        dset[:] = egse_data

    def write_viewport(self, viewport: int) -> None:
        """Add/update which viewports are illuminated.

        Parameters
        ----------
        viewport :  int
        """
        self.fid['/gse_data/viewport'][:] = viewport

    def write_reference_signal(self, signal, stdev) -> None:
        """Write reference detector signal and variance.

        Parameters
        ----------
        signal :  float
           Mean of signal level measured during the measurement
        stdev :  float
           standard deviation of signal level measured during the measurement

        Notes
        -----
        Used for non-linearity measurements.
        """
        gid = self.fid['/gse_data']
        gid.Illumination_level = 5e9 / 1.602176634 * signal

        dset = gid.createVariable('reference_signal', 'f8', ())
        dset.long_name = "mean of reference-diode signal"
        dset.comment = "t_sat = min(2.28e-9 / S_reference, 30)"
        dset.units = 'A'
        dset[:] = signal

        dset = gid.createVariable('reference_signal_std', 'f8', ())
        dset.long_name = "standard deviation of reference-diode signal"
        dset.units = 'A'
        dset[:] = stdev

    def write_reference_diode(self, ref_time, ref_data, ref_attrs) -> None:
        """Add data measured by the reference diode during the measurement.

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

    def write_wavelength_monitor(self, xds_wav_mon) -> None:
        """Add wavelength monitoring data of the Avantas fibre-spectrometer.

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
