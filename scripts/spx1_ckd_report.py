#!/usr/bin/env python3
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
Python script to generate a report from the CKD in a CKD product.
"""
import argparse

from moniplot.lib.fig_info import FIGinfo
from moniplot.mon_plot import MONplot

from pyspex.ckd_io import CKDio

# - global parameters ------------------------------


# - local functions --------------------------------
def init_plot_file(key: str, ckd, ckd_ref=None):
    """
    Initialize CKD report
    """
    fig_info_in = FIGinfo()
    fig_info_in.add('processor_version',
                    (ckd.processor_version,
                     ckd.git_commit), fmt='{} ({})')
    fig_info_in.add('processing_date', ckd.date_created())

    # open CKD product to be used as reference
    if ckd_ref is not None:
        fig_info_in.add('reference_version',
                        (ckd_ref.processor_version,
                         ckd_ref.git_commit), fmt='{} ({})')

    # open CKD report
    ckd_date = ckd.date_created(compact=True)
    plot = MONplot(f'spx1_ckd_report_{key}_{ckd_date}.pdf')
    plot.set_institute('SRON')

    return (plot, fig_info_in)


def add_dark_figs(ckd, ckd_ref=None):
    """Generate figures of Dark CKD.
    """
    dark_ckd = ckd.dark()
    if dark_ckd is None:
        return

    ckd_version = 'v1' if 'dark_offset' in dark_ckd.data_vars else 'v2'

    plot, fig_info_in = init_plot_file('dark', ckd, ckd_ref)
    plot.set_caption('SPEXone Dark CKD')
    if ckd_version == 'v1':
        plot.draw_signal(dark_ckd['dark_offset'], title='dark offset',
                         fig_info=fig_info_in.copy())
        plot.draw_hist(dark_ckd['dark_offset'], bins=161, vrange=[-5.5, 2.5],
                       title='dark offset', fig_info=fig_info_in.copy())
    else:
        plot.draw_signal(dark_ckd['offset_long'], title='offset (long)',
                         fig_info=fig_info_in.copy())
        plot.draw_hist(dark_ckd['offset_long'], bins=161, vrange=[-5.5, 2.5],
                       title='offset (long)', fig_info=fig_info_in.copy())
        plot.draw_signal(dark_ckd['offset_short'], title='offset (short)',
                         fig_info=fig_info_in.copy())
        plot.draw_hist(dark_ckd['offset_short'], bins=161, vrange=[-5.5, 2.5],
                       title='offset (short)', fig_info=fig_info_in.copy())
    plot.draw_signal(dark_ckd['dark_current'], title='dark current',
                     fig_info=fig_info_in.copy())
    plot.draw_hist(dark_ckd['dark_current'], bins=101, vrange=[1.5, 6.5],
                   title='dark current', fig_info=fig_info_in.copy())

    ref_ckd = ckd_ref.dark() if ckd_ref is not None else None
    if ref_ckd is None:
        plot.close()
        return

    ref_version = 'v1' if 'dark_offset' in ref_ckd.data_vars else 'v2'

    if ckd_version == 'v1':
        if ref_version == 'v1':
            plot.draw_signal(dark_ckd['dark_offset'] - ref_ckd['dark_offset'],
                             title='dark offset - reference',
                             fig_info=fig_info_in.copy(), zscale='diff')
        else:
            plot.draw_signal(dark_ckd['dark_offset'] - ref_ckd['offset_long'],
                             title='dark offset - reference',
                             fig_info=fig_info_in.copy(), zscale='diff')
    else:
        if ref_version == 'v1':
            plot.draw_signal(dark_ckd['offset_long'] - ref_ckd['dark_offset'],
                             title='offset (long) - reference',
                             fig_info=fig_info_in.copy(), zscale='diff')
        else:
            plot.draw_signal(dark_ckd['offset_short'] - ref_ckd['offset_short'],
                             title='offset (short) - reference',
                             fig_info=fig_info_in.copy(), zscale='diff')
            plot.draw_signal(dark_ckd['offset_long'] - ref_ckd['offset_long'],
                             title='offset (long) - reference',
                             fig_info=fig_info_in.copy(), zscale='diff')

    plot.draw_signal(dark_ckd['dark_current'] - ref_ckd['dark_current'],
                     title='dark current - reference',
                     fig_info=fig_info_in.copy(), zscale='diff')
    plot.close()


def add_noise_figs(ckd, ckd_ref=None):
    """Generate figures of Noise CKD.
    """
    noise_ckd = ckd.noise()
    if noise_ckd is None:
        return

    plot, fig_info_in = init_plot_file('noise', ckd, ckd_ref)
    plot.set_caption('SPEXone Noise CKD')
    g_str = 'inverse conversion gain'
    plot.draw_signal(1 / noise_ckd['g'], vrange=[11.5, 15], title=g_str,
                     fig_info=fig_info_in.copy())
    plot.draw_hist(1 / noise_ckd['g'], bins=161, vrange=[9.5, 17.5],
                   title=g_str, fig_info=fig_info_in.copy())
    n_str = 'read noise'
    plot.draw_signal(noise_ckd['n'], title=n_str,
                     fig_info=fig_info_in.copy())
    plot.draw_hist(noise_ckd['n'], bins=161, vrange=[0, 8],
                   title=n_str, fig_info=fig_info_in.copy())

    ref_ckd = ckd_ref.noise() if ckd_ref is not None else None
    if ref_ckd is not None:
        plot.draw_signal(noise_ckd['g'] - ref_ckd['g'],
                         title=g_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
        plot.draw_signal(noise_ckd['n'] - ref_ckd['n'],
                         title=n_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
    plot.close()


def add_nlin_figs(ckd, ckd_ref=None):
    """Generate figures of non-linearity CKD.
    """
    nlin_ckd = ckd.nlin()
    if nlin_ckd is None:
        return

    plot, fig_info_in = init_plot_file('nlin', ckd, ckd_ref)
    plot.set_caption('SPEXone non-linearity CKD')
    # if 'A' in nlin_ckd:
    #    plot.draw_signal(nlin_ckd['A'], title='parameter A',
    #                     fig_info=fig_info_in.copy())
    #    plot.draw_signal(nlin_ckd['B'], title='parameter B',
    #                     fig_info=fig_info_in.copy())
    #    plot.draw_signal(nlin_ckd['C'], title='parameter C',
    #                     fig_info=fig_info_in.copy())
    if 'c' in nlin_ckd:
        plot.draw_signal(nlin_ckd['f1'], title='parameter f1',
                         fig_info=fig_info_in.copy())
        plot.draw_signal(nlin_ckd['f2'], title='parameter f2',
                         fig_info=fig_info_in.copy())
        plot.draw_signal(nlin_ckd['c'], title='sigmoid parameter c',
                         fig_info=fig_info_in.copy())
        plot.draw_signal(nlin_ckd['r0'], title='sigmoid parameter r0',
                         fig_info=fig_info_in.copy())
        plot.draw_signal(nlin_ckd['r1'], title='sigmoid parameter r1',
                         fig_info=fig_info_in.copy())
        plot.draw_signal(nlin_ckd['r2'], title='sigmoid parameter r2',
                         fig_info=fig_info_in.copy())
        plot.draw_signal(nlin_ckd['r3'], title='sigmoid parameter r3',
                         fig_info=fig_info_in.copy())
        plot.draw_signal(nlin_ckd['r4'], title='sigmoid parameter r4',
                         fig_info=fig_info_in.copy())
        plot.draw_signal(nlin_ckd['m0'], title='sigmoid parameter m0',
                         fig_info=fig_info_in.copy())
        # plot.draw_signal(nlin_ckd['m1'], title='sigmoid parameter m1',
        #                 fig_info=fig_info_in.copy())
        # plot.draw_signal(nlin_ckd['m2'], title='sigmoid parameter m2',
        #                 fig_info=fig_info_in.copy())

    ref_ckd = ckd_ref.nlin() if ckd_ref is not None else None
    # plot commands...
    plot.close()


def add_prnu_figs(ckd, ckd_ref=None):
    """Generate figures of PRNU CKD.
    """
    prnu_ckd = ckd.prnu()
    if prnu_ckd is None:
        return

    ref_ckd = ckd_ref.prnu() if ckd_ref is not None else None

    plot, fig_info_in = init_plot_file('prnu', ckd, ckd_ref)
    plot.set_caption('SPEXone PRNU CKD')

    prnu_str = 'Pixel Response Non-Uniformity'
    plot.draw_signal(prnu_ckd, fig_info=fig_info_in.copy(),
                     title=prnu_str)
    if ref_ckd is not None:
        prnu_str = 'Pixel Response Non-Uniformity (reference)'
        plot.draw_signal(ref_ckd, fig_info=fig_info_in.copy(),
                         title=prnu_str)
    plot.draw_hist(prnu_ckd, bins=201, vrange=[0.95, 1.05],
                   title=prnu_str, fig_info=fig_info_in.copy())
    if ref_ckd is not None:
        prnu_str = 'Pixel Response Non-Uniformity (reference)'
        plot.draw_hist(ref_ckd, bins=201, vrange=[0.95, 1.05],
                       title=prnu_str, fig_info=fig_info_in.copy())
        prnu_str = 'Pixel Response Non-Uniformity'
        plot.draw_signal(prnu_ckd - ref_ckd,
                         title=prnu_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
    plot.close()


def add_fov_figs(ckd, ckd_ref=None):
    """Generate figures of Field-of-View CKD.
    """
    fov_ckd = ckd.fov()
    if fov_ckd is None:
        return

    plot, fig_info_in = init_plot_file('fov', ckd, ckd_ref)
    plot.set_caption('SPEXone field-of-view CKD')
    # plot commands...

    ref_ckd = ckd_ref.fov() if ckd_ref is not None else None
    # plot commands...
    plot.close()


def add_swath_figs(ckd, ckd_ref=None):
    """Generate figures of Swath CKD.
    """
    swath_ckd = ckd.swath()
    if swath_ckd is None:
        return

    plot, fig_info_in = init_plot_file('swath', ckd, ckd_ref)
    plot.set_caption('SPEXone Swath CKD')
    # plot commands...

    ref_ckd = ckd_ref.swath() if ckd_ref is not None else None
    # plot commands...
    plot.close()


def add_wave_figs(ckd, ckd_ref=None):
    """Generate figures of Wavelength CKD.
    """
    wave_ckd = ckd.wavelength()
    if wave_ckd is None:
        return

    plot, fig_info_in = init_plot_file('wave', ckd, ckd_ref)
    plot.set_caption('SPEXone Wavelength CKD')
    wave_s_str = 'wavelength grid of the S spectra'
    plot.draw_signal(wave_ckd['wave_full'][0, ...], fig_info=fig_info_in.copy(),
                     title=wave_s_str)
    wave_p_str = 'wavelength grid of the P spectra'
    plot.draw_signal(wave_ckd['wave_full'][1, ...], fig_info=fig_info_in.copy(),
                     title=wave_p_str)
    wave_str = 'common wavelength grid for S and P'
    plot.draw_signal(wave_ckd['wave_common'], fig_info=fig_info_in.copy(),
                     title=wave_str)

    ref_ckd = ckd_ref.wavelength() if ckd_ref is not None else None
    if ref_ckd is not None:
        plot.draw_signal(wave_ckd['wave_common'] - ref_ckd['wave_common'],
                         title=wave_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
        plot.draw_signal(wave_ckd['wave_full'][0, ...] - ref_ckd['wave_full'][0, ...],
                         title=wave_s_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
        plot.draw_signal(wave_ckd['wave_full'][1, ...] - ref_ckd['wave_full'][1, ...],
                         title=wave_p_str + ' - reference',
                         fig_info=fig_info_in.copy(), zscale='diff')
    plot.close()


def add_rad_figs(ckd, ckd_ref=None):
    """Generate figures of Radiometric CKD.
    """
    rad_ckd = ckd.radiometric()
    if rad_ckd is None:
        return

    plot, fig_info_in = init_plot_file('rad', ckd, ckd_ref)
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
    plot.close()


def add_pol_figs(ckd, ckd_ref=None):
    """Generate figures of Polarimetric CKD.
    """
    pol_ckd = ckd.polarimetric()
    if pol_ckd is None:
        return

    plot, fig_info_in = init_plot_file('pol', ckd, ckd_ref)
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
    plot.close()


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

    # open CKD product
    ckd = CKDio(args.ckd_file)

    # open CKD product to be used as reference
    ckd_ref = None
    if args.ref_ckd_file is not None:
        ckd_ref = CKDio(args.ref_ckd_file)

    # add CKD's to report
    add_dark_figs(ckd, ckd_ref)
    add_noise_figs(ckd, ckd_ref)
    add_nlin_figs(ckd, ckd_ref)
    add_prnu_figs(ckd, ckd_ref)
    # add_fov_figs(ckd, ckd_ref)
    # add_swath_figs(ckd, ckd_ref)
    add_wave_figs(ckd, ckd_ref)
    add_rad_figs(ckd, ckd_ref)
    add_pol_figs(ckd, ckd_ref)


# --------------------------------------------------
if __name__ == '__main__':
    main()
