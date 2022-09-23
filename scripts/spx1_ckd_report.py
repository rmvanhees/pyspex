#!/usr/bin/env python3
"""
This file is part of pyspex

https://github.com/rmvanhees/pyspex.git

Python script to generate a report from the CKD in a CKD product.

Copyright (c) 2022 SRON - Netherlands Institute for Space Research
   All Rights Reserved

License:  BSD-3-Clause
"""
import argparse

from moniplot.lib.fig_info import FIGinfo
from moniplot.mon_plot import MONplot

from pyspex.ckd_io import CKDio


# - global parameters ------------------------------
# - local functions --------------------------------
def add_dark_figs(plot, fig_info_in, ckd, ckd_ref=None):
    """
    Add figures of Dark CKD
    """
    dark_ckd = ckd.dark()
    if dark_ckd is None:
        return
    plot.set_caption('SPEXone Dark CKD')
    plot.draw_signal(dark_ckd['offset'], title='dark offset',
                     fig_info=fig_info_in.copy())
    plot.draw_signal(dark_ckd['current'], title='dark current',
                     fig_info=fig_info_in.copy())

    ref_ckd = ckd_ref.dark() if ckd_ref is not None else None
    if ref_ckd is not None:
        plot.draw_signal(dark_ckd['offset'] - ref_ckd['offset'],
                         title='dark offset - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
        plot.draw_signal(dark_ckd['current'] - ref_ckd['current'],
                         title='dark current - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')


def add_noise_figs(plot, fig_info_in, ckd, ckd_ref=None):
    """
    Add figures of Noise CKD
    """
    noise_ckd = ckd.noise()
    if noise_ckd is None:
        return

    plot.set_caption('SPEXone Noise CKD')
    g_str = 'conversion gain'
    plot.draw_signal(noise_ckd['g'], fig_info=fig_info_in.copy(),
                     title=g_str)
    n_str = 'read noise'
    plot.draw_signal(noise_ckd['n'], fig_info=fig_info_in.copy(),
                     title=n_str)

    ref_ckd = ckd_ref.noise() if ckd_ref is not None else None
    if ref_ckd is not None:
        plot.draw_signal(noise_ckd['g'] - ref_ckd['g'],
                         title=g_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
        plot.draw_signal(noise_ckd['n'] - ref_ckd['n'],
                         title=n_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')


def add_nlin_figs(plot, fig_info_in, ckd, ckd_ref=None):
    """
    Add figures of non-linearity CKD
    """
    nlin_ckd = ckd.nlin()
    if nlin_ckd is None:
        return

    plot.set_caption('SPEXone non-linearity CKD')
    # plot commands...

    ref_ckd = ckd_ref.nlin() if ckd_ref is not None else None
    # plot commands...


def add_prnu_figs(plot, fig_info_in, ckd, ckd_ref=None):
    """
    Add figures of PRNU CKD
    """
    prnu_ckd = ckd.prnu()
    if prnu_ckd is None:
        return

    plot.set_caption('SPEXone PRNU CKD')
    prnu_str = 'Pixel Response Non-Uniformity'
    plot.draw_signal(prnu_ckd, fig_info=fig_info_in.copy(),
                     title=prnu_str)

    ref_ckd = ckd_ref.prnu() if ckd_ref is not None else None
    if ref_ckd is not None:
        plot.draw_signal(prnu_ckd - ref_ckd,
                         title=prnu_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')


def add_fov_figs(plot, fig_info_in, ckd, ckd_ref=None):
    """
    Add figures of Field-of-View CKD
    """
    fov_ckd = ckd.fov()
    if fov_ckd is None:
        return

    plot.set_caption('SPEXone field-of-view CKD')
    # plot commands...

    ref_ckd = ckd_ref.fov() if ckd_ref is not None else None
    # plot commands...


def add_swath_figs(plot, fig_info_in, ckd, ckd_ref=None):
    """
    Add figures of Swath CKD
    """
    swath_ckd = ckd.swath()
    if swath_ckd is None:
        return

    plot.set_caption('SPEXone Swath CKD')
    # plot commands...

    ref_ckd = ckd_ref.swath() if ckd_ref is not None else None
    # plot commands...


def add_wave_figs(plot, fig_info_in, ckd, ckd_ref=None):
    """
    Add figures of Wavelength CKD
    """
    wave_ckd = ckd.wavelength()
    if wave_ckd is None:
        return

    plot.set_caption('SPEXone Wavelength CKD')
    wave_str = wave_ckd['common'].attrs['long_name']
    plot.draw_signal(wave_ckd['common'], fig_info=fig_info_in.copy(),
                     title=wave_str)
    wave_s_str = 'wavelengths of S spectra'
    plot.draw_signal(wave_ckd['full'][0, ...], fig_info=fig_info_in.copy(),
                     title=wave_s_str)
    wave_p_str = 'wavelengths of P spectra'
    plot.draw_signal(wave_ckd['full'][1, ...], fig_info=fig_info_in.copy(),
                     title=wave_p_str)

    ref_ckd = ckd_ref.wavelength() if ckd_ref is not None else None
    if ref_ckd is not None:
        plot.draw_signal(wave_ckd['common'] - ref_ckd['common'],
                         title=wave_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
        plot.draw_signal(wave_ckd['full'][0, ...] - ref_ckd['full'][0, ...],
                         title=wave_s_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
        plot.draw_signal(wave_ckd['full'][1, ...] - ref_ckd['full'][1, ...],
                         title=wave_p_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')


def add_rad_figs(plot, fig_info_in, ckd, ckd_ref=None):
    """
    Add figures of Radiometric CKD
    """
    rad_ckd = ckd.radiometric()
    if rad_ckd is None:
        return

    plot.set_caption('SPEXone Radiometric CKD')
    rad_s_str = rad_ckd.attrs['long_name'] + ' (S)'
    plot.draw_signal(rad_ckd[:, 0, :], fig_info=fig_info_in.copy(),
                     title=rad_s_str)
    rad_p_str = rad_ckd.attrs['long_name'] + ' (P)'
    plot.draw_signal(rad_ckd[:, 1, :], fig_info=fig_info_in.copy(),
                     title=rad_p_str)

    ref_ckd = ckd_ref.radiometric() if ckd_ref is not None else None
    if ref_ckd is not None:
        plot.draw_signal(rad_ckd[:, 0, :] - ref_ckd[:, 0, :],
                         title=rad_s_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
        plot.draw_signal(rad_ckd[:, 1, :] - ref_ckd[:, 1, :],
                         title=rad_p_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')


def add_pol_figs(plot, fig_info_in, ckd, ckd_ref=None):
    """
    Add figures of Polarimetric CKD
    """
    pol_ckd = ckd.polarimetric()
    if pol_ckd is None:
        return

    plot.set_caption('SPEXone Polarimetric CKD')
    pol_q_str = pol_ckd['pol_m_q'].attrs['long_name']
    plot.draw_signal(pol_ckd['pol_m_q'], fig_info=fig_info_in.copy(),
                     title=pol_q_str)
    pol_u_str = pol_ckd['pol_m_u'].attrs['long_name']
    plot.draw_signal(pol_ckd['pol_m_u'], fig_info=fig_info_in.copy(),
                     title=pol_u_str)
    pol_t_str = pol_ckd['pol_m_t'].attrs['long_name']
    plot.draw_signal(pol_ckd['pol_m_t'], fig_info=fig_info_in.copy(),
                     title=pol_t_str)

    ref_ckd = ckd_ref.polarimetric() if ckd_ref is not None else None
    if ref_ckd is not None:
        plot.draw_signal(pol_ckd['pol_m_q'] - ref_ckd['pol_m_q'],
                         title=pol_q_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
        plot.draw_signal(pol_ckd['pol_m_u'] - ref_ckd['pol_m_u'],
                         title=pol_u_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
        plot.draw_signal(pol_ckd['pol_m_t'] - ref_ckd['pol_m_t'],
                         title=pol_t_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')


# - main function ----------------------------------
def main() -> None:
    """
    Main function
    """
    # parse command-line parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description='Generate a PDF report from CKD in a SPEXone CKD product')

    parser.add_argument('--ref_ckd_file', default=None,
                        help='name of reference CKD product')
    parser.add_argument('ckd_file', help='name of CKD product')
    args = parser.parse_args()

    # open CKD report
    plot = MONplot('spx1_ckd_report.pdf')
    plot.set_institute('SRON')
    fig_info_in = FIGinfo()

    # open CKD product
    ckd = CKDio(args.ckd_file)
    fig_info_in.add('processor_version',
                    (ckd.processor_version,
                     ckd.git_commit), fmt='{} ({})')
    fig_info_in.add('processing_date', ckd.date_created)

    # open CKD product to be used as reference
    ckd_ref = None
    if args.ref_ckd_file is not None:
        ckd_ref = CKDio(args.ref_ckd_file)
        fig_info_in.add('reference_version',
                        (ckd_ref.processor_version,
                         ckd_ref.git_commit), fmt='{} ({})')

    # add CKD's to report
    add_dark_figs(plot, fig_info_in, ckd, ckd_ref)
    add_noise_figs(plot, fig_info_in, ckd, ckd_ref)
    # add_nlin_figs(plot, fig_info_in, ckd, ckd_ref)
    add_prnu_figs(plot, fig_info_in, ckd, ckd_ref)
    # add_fov_figs(plot, fig_info_in, ckd, ckd_ref)
    # add_swath_figs(plot, fig_info_in, ckd, ckd_ref)
    add_wave_figs(plot, fig_info_in, ckd, ckd_ref)
    add_rad_figs(plot, fig_info_in, ckd, ckd_ref)
    add_pol_figs(plot, fig_info_in, ckd, ckd_ref)

    plot.close()
    ckd.close()
    ckd_ref.close()


# --------------------------------------------------
if __name__ == '__main__':
    main()
