"""
This script runs fullsky, white sims to calculate the fullsky kspace filter in harmonic space.
"""
from pspipe_utils import log, kspace
from pspy import so_dict, so_map, pspy_utils

from mnms import utils
from pixell import enmap, curvedsky, wcsutils
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import savgol_filter

import os
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

filters_dir = d['filters_dir']
plot_dir = os.path.join(d['plot_dir'], 'filters')
pspy_utils.create_directory(filters_dir)
pspy_utils.create_directory(plot_dir)

surveys = d['surveys']
arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}

apply_kspace_filter = d["apply_kspace_filter"]
lmax = d['lmax']
ainfo = curvedsky.alm_info(lmax=lmax)
ainfo_rect = curvedsky.alm_info(lmax=lmax, layout='rect')

num_flm_sims = d['num_flm_sims']

savgol_w = d['savgol_w']
savgol_k = d['savgol_k']

# Get fullsky, white sims to calculate the fullsky kspace filter in harmonic space.
# To do so, we may need to upgrade the filter resolution in the y-direction to support 
# SHT to our lmax. To make sure the filter upgrades cleanly, we can only upgrade in
# integer amounts.
if apply_kspace_filter:

    for sv1 in surveys:
        fn = f'{filters_dir}/{sv1}_flm_fullsky.npy'
        
        if not os.path.isfile(fn):
            # Check to make sure only one geometry
            i = 0
            for ar1 in arrays[sv1]:
                for chan1 in arrays[sv1][ar1]:
                    for pol1 in ['T', 'pol']:
                        win_fn = d[f'window_{pol1}_{sv1}_{ar1}_{chan1}']
                        if i == 0:
                            template = so_map.read_map(win_fn)
                            template_shape, template_wcs = template.data.geometry
                            assert len(template_shape) == 2, 'Template must have 2 dimensions'
                        else:
                            _template_shape, _template_wcs = enmap.read_map_geometry(win_fn)
                            assert _template_shape == template_shape, \
                                f'Only one geometry allowed for {sv1=}, mismatched shapes'
                            assert wcsutils.equal(_template_wcs, template_wcs), \
                                f'Only one geometry allowed for {sv1=}, mismatched wcs'
                            i += 1
        
            # get the fullsky geometry and filter, upgraded in y as necessary
            
            # lmax here assumes fejer1 variant (c.f. lmax+1 for cc), but that's ok
            # since we are doing fullsky (unmasked) sims, so any geometry works
            nup = int(np.ceil(lmax / template_shape[0]))
            log.info(f'Upgrading y-direction shape {template_shape[0]} by a factor of {nup} to support {lmax=}')

            full_shape, full_wcs = enmap.fullsky_geometry(
                shape=(template_shape[0]*nup, template_shape[1]), variant='fejer1'
                )

            ks_f = d[f"k_filter_{sv1}"]
            filter = kspace.get_kspace_filter(template, ks_f)
            filter = filter[..., :template_shape[1]//2 + 1] # prepare for rfft
            filter = np.repeat(filter, nup, axis=0)

            # now we can accumulate the fullsky white noise sims
            flm = 0
            for i in range(num_flm_sims):
                eta = utils.concurrent_normal(size=ainfo.nelem, scale=1/np.sqrt(2), seed=[0, i], dtype=np.float64, complex=True)
                eta[..., :ainfo.lmax + 1] *= np.sqrt(2) # respect reality condition of m=0 
                eta = curvedsky.alm2map(eta, enmap.zeros(full_shape, full_wcs), ainfo=ainfo, method='cyl')
                eta = utils.rfft(eta, normalize='backward')
                eta = utils.concurrent_op(np.multiply, filter, eta, flatten_axes=[-2, -1])
                eta = enmap.ndmap(utils.irfft(eta, normalize='backward'), full_wcs)
                flm += np.abs(curvedsky.map2alm(eta, ainfo=ainfo, method='cyl'))**2
                if (i + 1) % int(num_flm_sims * 0.1) == 0:
                    log.info(f'Done {i+1} out of {num_flm_sims}')
            flm /= num_flm_sims
            flm **= 0.5
            np.save(fn, flm)
        else:
            flm = np.load(fn)

        _fl1 = utils.alm2cl(flm**0.5*(1+0j), flm**0.5*(1+0j))
        fl1 = np.zeros_like(_fl1)
        fl1[_fl1 > 1e-3] = savgol_filter(_fl1[_fl1 > 1e-3], savgol_w, savgol_k)
        fl1[fl1 < 0] = 0
        np.save(f'{filters_dir}/{sv1}_fl1_fullsky.npy', fl1)
        
        _fl2 = utils.alm2cl(flm*(1+0j), flm*(1+0j))
        fl2 = np.zeros_like(_fl2)
        fl2[_fl2 > 1e-3] = savgol_filter(_fl2[_fl2 > 1e-3], savgol_w, savgol_k)
        fl2[fl2 < 0] = 0
        np.save(f'{filters_dir}/{sv1}_fl2_fullsky.npy', fl2)

        _fl4 = utils.alm2cl(flm**2*(1+0j), flm**2*(1+0j))
        fl4 = np.zeros_like(_fl4)
        fl4[_fl4 > 1e-3] = savgol_filter(_fl4[_fl4 > 1e-3], savgol_w, savgol_k)
        fl4[fl4 < 0] = 0
        np.save(f'{filters_dir}/{sv1}_fl4_fullsky.npy', fl4)

        flm_rect = curvedsky.transfer_alm(ainfo, flm, ainfo_rect).reshape(lmax+1, -1)
        plt.figure(figsize=(8, 8))
        plt.imshow(flm_rect, vmin=0, vmax=1, origin='lower')
        plt.colorbar()
        plt.ylabel('m')
        plt.xlabel('l')
        plt.title('flm')
        plt.savefig(f'{plot_dir}/{sv1}_flm_fullsky.png')

        plt.figure(figsize=(8, 8))
        plt.imshow(flm_rect[:300, :300], vmin=0, vmax=1, origin='lower')
        plt.colorbar()
        plt.ylabel('m')
        plt.xlabel('l')
        plt.title('flm')        
        plt.savefig(f'{plot_dir}/{sv1}_flm_fullsky_zoom.png')

        for p, filt in enumerate((fl1, fl2, fl4)):
            edges = [np.argmin(abs(filt - i)) for i in (.2, .4, .6, .8)]
            min_edge = 80
            for sel in [np.s_[min_edge:lmax+1], np.s_[min_edge:edges[0]]] + [np.s_[edges[i]:edges[i+1]] for i in range(0, len(edges)-1)] + [np.s_[edges[-1]:lmax+1]]:
                plt.figure(figsize=(16, 8))
                plt.semilogx(np.arange(sel.start, sel.stop), curvedsky.alm2cl(flm**(2**(p-1))*(1+0j))[sel], alpha=.3, label='sims')
                plt.semilogx(np.arange(sel.start, sel.stop), filt[sel], label='fit')
                plt.xlim(sel.start, sel.stop-1)
                plt.xlabel('l')
                plt.ylabel('tf')
                plt.legend()
                plt.title(f'fl{2**p}')
                plt.grid()
                plt.savefig(f'{plot_dir}/{sv1}_fl{2**p}_fullsky_{sel.start}_{sel.stop}.png')
else:
    log.info(f'WARNING: no kspace filter, so running {__name__} is unnecessary')