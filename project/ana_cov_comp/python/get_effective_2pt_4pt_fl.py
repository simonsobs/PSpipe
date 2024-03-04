"""
The anisotropy of the kspace filter and/or the noise breaks the standard
master toolkit, both at the 2pt (mode-coupling) and 4pt (covariance) level.
This is the first of 3 scripts that develop an ansatz correction for this.

The ansatz essentially consists of a diagonal transfer function template 
in ell raised to a power, where the power is a function of the filter, mask,
and power spectra. In this first script, we get the transfer function template.
In order to do this, we want to ask "what does the kspace filter look like in
harmonic space?" We answer that by running fullsky white noise sims, filtering
them, and then recording their effect as a function of ell.

It's important that the template corresponds to the fullsky filter: the filter
couples to the mask, so we don't want to mix effects here. Thus, we get the
closest filter acting on the fullsky to the actual filter that acts on the 
data footprint (which is also masked).
"""
from pspipe_utils import log, kspace, pspipe_list
from pspy import so_dict, so_map, pspy_utils

from mnms import utils
from pixell import enmap, curvedsky, wcsutils
import matplotlib.pyplot as plt

import numpy as np
from scipy.signal import savgol_filter

import os
import sys

# FIXME: allow array over channels/pols

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

filters_dir = d['filters_dir']
plot_dir = os.path.join(d['plot_dir'], 'filters')
pspy_utils.create_directory(filters_dir)
pspy_utils.create_directory(plot_dir)

sv2arrs2chans = pspipe_list.get_survey_array_channel_map(d)

apply_kspace_filter = d["apply_kspace_filter"]
lmax = d['lmax']
ainfo = curvedsky.alm_info(lmax=lmax)
ainfo_rect = curvedsky.alm_info(lmax=lmax, layout='rect')

num_flm_sims = d['num_flm_sims']

savgol_w = d['savgol_w']
savgol_k = d['savgol_k']

# get fullsky, white sims to calculate the fullsky kspace filter in harmonic space
if apply_kspace_filter:

    for sv1 in sv2arrs2chans:
        fn_flm_2pt = f'{filters_dir}/{sv1}_flm_2pt_fullsky.npy'
        fn_flm_4pt = f'{filters_dir}/{sv1}_flm_4pt_fullsky.npy'
        
        if not os.path.isfile(fn_flm_2pt) or not os.path.isfile(fn_flm_4pt):
            # check to make sure only one geometry
            i = 0
            for ar1 in sv2arrs2chans[sv1]:
                for chan1 in sv2arrs2chans[sv1][ar1]:
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

            # check to make sure the x-resolution is at least as high as y,
            # otherwise we can't support mmax = lmax and none of this makes sense.
            # remember, in wcs, the first element is x
            assert np.abs(template_wcs.wcs.cdelt[0]) <= np.abs(template_wcs.wcs.cdelt[1]), \
                'The template x-resolution is lower than y-resolution'
        
            # get the fullsky geometry and filter
            full_shape, full_wcs = enmap.fullsky_geometry(
                res=np.deg2rad(np.abs(template_wcs.wcs.cdelt[::-1])), # res assumes y-major ordering, wcs is opposite
                variant=utils.get_variant(template_shape, template_wcs)
                ) # FIXME: for noise nup -> padding

            # FIXME: for now, assume x-direction is fullsky
            assert full_shape[1] == template_shape[1], \
                'x-direction needs to be fullsky'
            
            # get the filters, both to be used in mock sims (so, the survey
            # geometry) and for the fullsky sims (so, the fullsky geometry)
            fn_fk = f'{filters_dir}/{sv1}_fk.npy'
            fn_fk_fullsky = f'{filters_dir}/{sv1}_fk_fullsky.npy'

            if not os.path.isfile(fn_fk) or not os.path.isfile(fn_fk_fullsky):
                fk = kspace.get_kspace_filter(template, d[f"k_filter_{sv1}"], dtype=np.float32) # single prec sims
                fk = fk[..., :fk.shape[1]//2 + 1] # prepare for rfft

                fk_fullsky = kspace.get_kspace_filter(template, d[f"k_filter_{sv1}"], dtype=np.float32,
                                                          shape_y=full_shape, wcs_y=full_wcs)
                fk_fullsky = fk_fullsky[..., :fk_fullsky.shape[1]//2 + 1] # prepare for rfft

                np.save(fn_fk, fk)
                np.save(fn_fk_fullsky, fk_fullsky)

                utils.eplot(np.fft.fftshift(fk, axes=0)[:, ::-1], fname=f'{plot_dir}/fk',
                            downgrade=16, colorbar=True, grid=False, min=0.5, max=1.5)
                utils.eplot(np.fft.fftshift(fk_fullsky, axes=0)[:, ::-1], fname=f'{plot_dir}/fk_fullsky',
                            downgrade=16, colorbar=True, grid=False, min=0.5, max=1.5)
            else:
                fk = np.load(fn_fk)
                fk_fullsky = np.load(fn_fk_fullsky)

            # now we can accumulate the fullsky white noise sims
            fk_4pt_fullsky = fk_fullsky**2               

            flm_2pt = np.zeros(ainfo.nelem, dtype=np.float64) # double prec result
            flm_4pt = np.zeros(ainfo.nelem, dtype=np.float64)
            for i in range(num_flm_sims):
                eta = utils.concurrent_normal(size=ainfo.nelem, scale=1/np.sqrt(2), seed=[0, i], dtype=np.float32, complex=True) # single prec sims
                eta[..., :ainfo.lmax + 1] *= np.sqrt(2) # respect reality condition of m=0 
                eta = curvedsky.alm2map(eta, enmap.zeros(full_shape, full_wcs), ainfo=ainfo, method='cyl')
                
                eta = utils.rfft(eta, normalize='backward')
                for j, _flm in enumerate([flm_2pt, flm_4pt]):
                    _fk_fullsky = [fk_fullsky, fk_4pt_fullsky][j]
                    _eta = utils.concurrent_op(np.multiply, _fk_fullsky, eta, flatten_axes=[-2, -1])
                    _eta = utils.irfft(_eta, normalize='backward')
                    _eta = enmap.ndmap(_eta, full_wcs)
                    _flm += np.abs(curvedsky.map2alm(_eta, ainfo=ainfo, method='cyl'))**2
                
                if (i + 1) % int(num_flm_sims * 0.1) == 0:
                    log.info(f'Done {i+1} out of {num_flm_sims}')
            
            for j, _flm in enumerate([flm_2pt, flm_4pt]):
                _flm /= num_flm_sims
                _flm **= 1/(2*(j + 1)) # at the "field level" for both 2pt and 4pt

                np.save([fn_flm_2pt, fn_flm_4pt][j], _flm)
        else:
            flm_2pt = np.load(fn_flm_2pt)
            flm_4pt = np.load(fn_flm_4pt)

        fn_fl_2pt = f'{filters_dir}/{sv1}_fl_2pt_fullsky.npy'
        fn_fl_4pt = f'{filters_dir}/{sv1}_fl_4pt_fullsky.npy'

        # now do the the savitzky golay filter to smooth the transfer
        # function template as a function of ell
        if not os.path.isfile(fn_fl_2pt) or not os.path.isfile(fn_fl_4pt):
            fl_2pt = np.zeros(lmax + 1)
            fl_4pt = np.zeros(lmax + 1)
            for j, _flm in enumerate([flm_2pt, flm_4pt]):
                _fl_data = curvedsky.alm2cl(_flm**(j+1)*(1+0j)) # at the "ps level" for 2pt, "covmat level" for 4pt
                _fl = [fl_2pt, fl_4pt][j]
                _fl[_fl_data >= 1e-3] = savgol_filter(_fl_data[_fl_data >= 1e-3], savgol_w, savgol_k)
                _fl[_fl < 1e-3] = 0 # in case savgol goes negative

                np.save([fn_fl_2pt, fn_fl_4pt][j], _fl)
        else:
            fl_2pt = np.load(fn_fl_2pt)
            fl_4pt = np.load(fn_fl_4pt)

        for j, _flm in enumerate([flm_2pt, flm_4pt]):
            _flm_rect = curvedsky.transfer_alm(ainfo, _flm, ainfo_rect).reshape(lmax+1, -1)

            plt.figure(figsize=(8, 8))
            plt.imshow(_flm_rect, vmin=0.5, vmax=1.5, origin='lower')
            plt.colorbar()
            plt.ylabel('m')
            plt.xlabel('l')
            plt.title('flm')
            plt.savefig(f'{plot_dir}/{sv1}_flm_{2*(j+1)}pt_fullsky.png')

            plt.figure(figsize=(8, 8))
            plt.imshow(_flm_rect[:300, :300], vmin=0.5, vmax=1.5, origin='lower')
            plt.colorbar()
            plt.ylabel('m')
            plt.xlabel('l')
            plt.title('flm')        
            plt.savefig(f'{plot_dir}/{sv1}_flm_{2*(j+1)}pt_fullsky_zoom.png')

        for j, _fl in enumerate([fl_2pt, fl_4pt]):
            edges = [np.argmin(abs(_fl - i)) for i in (.2, .4, .6, .8)]
            min_edge = max(min(80, edges[0] - 10), 0)
            _flm = [flm_2pt, flm_4pt][j]
            for sel in [np.s_[min_edge:lmax+1], np.s_[min_edge:edges[0]]] + [np.s_[edges[i]:edges[i+1]] for i in range(0, len(edges)-1)] + [np.s_[edges[-1]:lmax+1]]:
                plt.figure(figsize=(16, 8))
                plt.semilogx(np.arange(sel.start, sel.stop), curvedsky.alm2cl(_flm**(j+1)*(1+0j))[sel], alpha=.3, label='sims')
                plt.semilogx(np.arange(sel.start, sel.stop), _fl[sel], label='fit')
                plt.xlim(sel.start, sel.stop-1)
                plt.xlabel('l')
                plt.ylabel('tf')
                plt.legend()
                plt.title(f'fl{2*(j+1)}')
                plt.grid()
                plt.savefig(f'{plot_dir}/{sv1}_fl{2*(j+1)}_fullsky_{sel.start}_{sel.stop}.png')
else:
    log.info(f'WARNING: no kspace filter, so running {__name__} is unnecessary')