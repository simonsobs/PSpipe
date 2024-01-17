"""
This script runs masked, ps'ed sims for the mock pol and T noise ps and average noise mask.

This script assumes:
1. No cross-survey spectra.
2. All power spectra and masks are similar enough for all fields in a survey.
"""
from pspipe_utils import log, kspace
from pspy import so_dict, so_map, pspy_utils

from mnms import utils
from pixell import enmap, curvedsky

import numpy as np

import os
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

filters_dir = d['filters_dir']
pspy_utils.create_directory(filters_dir)

surveys = d['surveys']
arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}

apply_kspace_filter = d["apply_kspace_filter"]
lmax = d['lmax']
ainfo = curvedsky.alm_info(lmax=lmax)

num_tf_sims = d['num_tf_sims']
start, stop = 0, num_tf_sims
if len(sys.argv) == 4:
    log.info(f'computing only the spectra matrices: ' + 
             f'{int(sys.argv[2])}:{int(sys.argv[3])} of {num_tf_sims}')
    start, stop = int(sys.argv[2]), int(sys.argv[3])

def get_ps(lmax, lknee, lcap, pow):
    """Get a mock power spectrum that is white at high-ell, a power-law at low
    ell, but is capped at a given minimum ell.

    Parameters
    ----------
    lmax : int
        lmax of power spectrum.
    lknee : int
        lknee of power law.
    lcap : int
        minimum ell at which the power law is capped.
    pow : scalar
        exponent of power law.

    Returns
    -------
    np.ndarray (lmax+1,)
        mock power spectrum. 
    """
    ells = np.arange(lmax + 1, dtype=np.float64)
    ps = np.zeros_like(ells)
    ps[lcap:] = (ells[lcap:]/lknee)**pow + 1
    ps[:lcap] = ps[lcap]
    return ps

if apply_kspace_filter:

    for sv1 in surveys:
        # Get our mock power spectra
        log.info(f'Getting mock power spectra for {sv1=}')
        ps_params = d[f'mock_ps_{sv1}']

        pss = {
            k: get_ps(lmax, lknee=ps_params[k]['lknee'], lcap=ps_params[k]['lcap'], pow=ps_params[k]['pow']) for k in ('T', 'pol')
        }

        # Get our average mask (total normalization doesn't matter)
        log.info(f'Getting mock mask for {sv1=}')
        fn_T = f'{filters_dir}/{sv1}_win_T.fits'
        fn_pol = f'{filters_dir}/{sv1}_win_pol.fits'
        if not os.path.isfile(fn_T) or not os.path.isfile(fn_pol):
            w_T = 0
            w_pol = 0
            csig = 0
            for ar1 in arrays[sv1]:
                for chan1 in arrays[sv1][ar1]:
                    w_T += enmap.read_map(d[f'window_T_{sv1}_{ar1}_{chan1}'])
                    w_pol += enmap.read_map(d[f'window_pol_{sv1}_{ar1}_{chan1}'])
                    for split1 in d[f'ivars_{sv1}_{ar1}_{chan1}']: # FIXME: iterate over field_infos in case pol matters
                        csig += enmap.read_map(split1)
            csig = np.sqrt(np.reciprocal(csig, where=csig!=0) * (csig!=0))
            csig *= csig.pixsizemap()**0.5
            w_T *= csig
            w_pol *= csig

            enmap.write_map(fn_T, w_T)
            enmap.write_map(fn_pol, w_pol)
        w_T = so_map.read_map(fn_T)
        w_pol = so_map.read_map(fn_pol)
        assert w_T.data.ndim == 2 and w_pol.data.ndim == 2, 'Masks must have 2 dimensions'
        
        masks = {'T': w_T, 'pol': w_pol}
    
        # get the filter, upgraded in y as necessary
        ks_f = d[f"k_filter_{sv1}"]
        filter = kspace.get_kspace_filter(w_T, ks_f)
        filter = filter[..., :w_T.data.shape[1]//2 + 1] # prepare for rfft

        # now we can accumulate the masked, ps'ed sims
        for p, pol in enumerate(('T', 'pol')):
            ps = pss[pol]
            mask = masks[pol].data
            for i in range(start, stop):
                fn = f'{filters_dir}/{sv1}_{pol}_filt_masked_spec{i}.npy'
                if not os.path.isfile(fn):
                    eta = utils.concurrent_normal(size=ainfo.nelem, scale=1/np.sqrt(2), seed=[p+1, i], dtype=np.float64, complex=True) # different than flm seed
                    eta[..., :ainfo.lmax + 1] *= np.sqrt(2) # respect reality condition of m=0 
                    eta = curvedsky.almxfl(eta, ps**0.5, ainfo)
                    eta = curvedsky.alm2map(eta, enmap.zeros(mask.shape, mask.wcs), ainfo=ainfo, method='cyl')
                    eta = utils.rfft(eta, normalize='backward')
                    eta = utils.concurrent_op(np.multiply, filter, eta, flatten_axes=[-2, -1])
                    eta = utils.irfft(eta, normalize='backward')
                    eta = enmap.ndmap(utils.concurrent_op(np.multiply, mask, eta), mask.wcs)
                    spec = curvedsky.alm2cl(curvedsky.map2alm(eta, ainfo=ainfo, method='cyl'), ainfo=ainfo)
                    if (i + 1) % int((start - stop) * 0.1) == 0:
                        log.info(f'Done {i+1} out of {start} to {stop}')
                    np.save(fn, spec)
                else:
                    continue
else:
    log.info(f'WARNING: no kspace filter, so this {__name__} is unnecessary')