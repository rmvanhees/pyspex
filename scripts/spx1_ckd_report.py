

from pyspex.ckd_io import CKDio

from moniplot.lib.fig_info import FIGinfo
from moniplot.mon_plot import MONplot



def main() -> None:
    """
    Main function
    """
    ckd_dir = '/data/richardh/SPEXone/share/ckd/'
    ckd_file = 'CKD_reference_all_20220719_145838.nc'

    plot = MONplot('spx1_ckd_report.pdf')
    plot.set_institute('SRON')
    fig_info_in = FIGinfo()
    
    with CKDio(ckd_dir + ckd_file) as ckd:
        fig_info_in.add('processor_version',
                        (ckd.processor_version,
                         ckd.git_commit), fmt='{} ({})')
        fig_info_in.add('processing_date', ckd.date_created)
        dark_ckd = ckd.dark()
        noise_ckd = ckd.noise()
        prnu_ckd = ckd.prnu()
        wave_ckd = ckd.wavelength()
        rad_ckd = ckd.radiometric()
        pol_ckd = ckd.polarimetric()

    if dark_ckd is not None:
        plot.set_caption('SPEXone Dark CKD')
        plot.draw_signal(dark_ckd['offset'], title='dark offset',
                         fig_info=fig_info_in.copy())
        plot.draw_signal(dark_ckd['current'], title='dark current',
                         fig_info=fig_info_in.copy())
    if noise_ckd is not None:
        plot.set_caption('SPEXone Noise CKD')
        plot.draw_signal(noise_ckd['g'], fig_info=fig_info_in.copy(),
                         title=noise_ckd['g'].attrs['long_name'])
        plot.draw_signal(noise_ckd['n'], fig_info=fig_info_in.copy(),
                         title=noise_ckd['n'].attrs['long_name'])
    if prnu_ckd is not None:
        plot.set_caption('SPEXone PRNU CKD')
        plot.draw_signal(prnu_ckd, fig_info=fig_info_in.copy(),
                         title=prnu_ckd.attrs['long_name'])
    if wave_ckd is not None:
        plot.set_caption('SPEXone Wavelength CKD')
        plot.draw_signal(wave_ckd['common'], fig_info=fig_info_in.copy(),
                         title=wave_ckd['common'].attrs['long_name'])
        plot.draw_signal(wave_ckd['full'][0, ...], fig_info=fig_info_in.copy(),
                         title='wavelengths of S spectra')
        plot.draw_signal(wave_ckd['full'][1, ...], fig_info=fig_info_in.copy(),
                         title='wavelengths of P spectra')
    if rad_ckd is not None:
        plot.set_caption('SPEXone Radiometric CKD')
        plot.draw_signal(rad_ckd[:, 0, :], fig_info=fig_info_in.copy(),
                         title=rad_ckd.attrs['long_name'] + ' (S)')
        plot.draw_signal(rad_ckd[:, 1, :], fig_info=fig_info_in.copy(),
                         title=rad_ckd.attrs['long_name'] + ' (P)')
    if pol_ckd is not None:
        plot.set_caption('SPEXone Polarimetric CKD')
        plot.draw_signal(pol_ckd['pol_m_q'], fig_info=fig_info_in.copy(),
                         title=pol_ckd['pol_m_q'].attrs['long_name'])
        plot.draw_signal(pol_ckd['pol_m_u'], fig_info=fig_info_in.copy(),
                         title=pol_ckd['pol_m_u'].attrs['long_name'])
        plot.draw_signal(pol_ckd['pol_m_t'], fig_info=fig_info_in.copy(),
                         title=pol_ckd['pol_m_t'].attrs['long_name'])

    plot.close()


# --------------------------------------------------
if __name__ == '__main__':
    main()
