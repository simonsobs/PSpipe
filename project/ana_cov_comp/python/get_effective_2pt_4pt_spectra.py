"""
The anisotropy of the kspace filter and/or the noise breaks the standard
master toolkit, both at the 2pt (mode-coupling) and 4pt (covariance) level.
This is the second  of 3 scripts that develop an ansatz correction for this.

The ansatz essentially consists of a diagonal transfer function template 
in ell raised to a power, where the power is a function of the filter, mask,
and power spectra. In this second script, we get mock/simple simulations that
we will use to fit for the power in the ansatz. The idea is that this power 
is not very sensitive to details of the mask or power spectrum; rather it is
most sensitive to the filter. Thus, we don't need to run accurate or complete
simulations of the survey; instead we can run a small number of simulations
with a mask and power spectrum that are roughly representative.

This script assumes:
1. No cross-survey spectra.
2. All power spectra and masks are similar enough for all fields in a survey.
"""
from pspipe_utils import log, kspace, pspipe_list, covariance as psc
from pspy import so_dict, so_map, pspy_utils

from mnms import utils
from pixell import enmap, curvedsky

import numpy as np

import os
import sys
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

filters_dir = d['filters_dir']
pspy_utils.create_directory(filters_dir)

sv2arrs2chans = pspipe_list.get_survey_array_channel_map(d)

apply_kspace_filter = d["apply_kspace_filter"]
lmax = d['lmax']
ainfo = curvedsky.alm_info(lmax=lmax)

num_tf_sims = d['num_tf_sims']
start, stop = 0, num_tf_sims
if len(sys.argv) == 4:
    log.info(f'computing only the spectra matrices: ' + 
             f'{int(sys.argv[2])}:{int(sys.argv[3])} of {num_tf_sims}')
    start, stop = int(sys.argv[2]), int(sys.argv[3])

if apply_kspace_filter:

    for sv1 in sv2arrs2chans:
        # Get our mock power spectra
        log.info(f'Getting mock power spectra for {sv1=}')
        ps_params = d[f'mock_ps_{sv1}']

        pss = {
            k: psc.get_mock_noise_ps(
                lmax, lknee=ps_params[k]['lknee'], lcap=ps_params[k]['lcap'], pow=ps_params[k]['pow']
                ) for k in ('T', 'pol')
        }

        # Get our average mask (total normalization doesn't matter)
        log.info(f'Getting mock mask for {sv1=}')
        fn_T = f'{filters_dir}/{sv1}_win_T.fits'
        fn_pol = f'{filters_dir}/{sv1}_win_pol.fits'
        fn_k = f'{filters_dir}/{sv1}_win_k.fits'

        # only job that starts at 0 will create masks. this is to avoid 
        # a weird error that occurs in enmap.write_map wherein multiple
        # processes writing to the same filename at the same time can 
        # crash. so here, other jobs will wait for the first job to create
        # the masks
        cond = lambda x, y, z: not os.path.isfile(x) or not os.path.isfile(y) or not os.path.isfile(z)
        if start == 0 and cond(fn_T, fn_pol, fn_k):
            w_T = 0
            w_pol = 0
            w_k = 0
            w_count = 0
            
            csig = 0
            csig_count = 0
            for ar1 in sv2arrs2chans[sv1]:
                for chan1 in sv2arrs2chans[sv1][ar1]:
                    w_T += enmap.read_map(d[f'window_T_{sv1}_{ar1}_{chan1}'])
                    w_pol += enmap.read_map(d[f'window_pol_{sv1}_{ar1}_{chan1}'])
                    w_k += enmap.read_map(d[f'window_kspace_{sv1}_{ar1}_{chan1}'])
                    w_count += 1

                    for split1 in d[f'ivars_{sv1}_{ar1}_{chan1}']: # FIXME: iterate over field_infos in case pol matters
                        csig += enmap.read_map(split1)
                        csig_count += 1
            
            # csig comes from the average ivar map
            csig /= csig_count
            csig = np.sqrt(np.reciprocal(csig, where=csig!=0) * (csig!=0))
            csig *= csig.pixsizemap()**0.5
            
            # average analysis windows
            w_T /= w_count
            w_pol /= w_count
            w_k /= w_count

            w_T *= csig
            w_pol *= csig

            enmap.write_map(fn_T, w_T.astype(np.float32, copy=False)) # single prec sims
            enmap.write_map(fn_pol, w_pol.astype(np.float32, copy=False))
            enmap.write_map(fn_k, w_k.astype(np.float32, copy=False))
        else:
            while cond(fn_T, fn_pol, fn_k):
                time.sleep(1)
            w_T = enmap.read_map(fn_T)
            w_pol = enmap.read_map(fn_pol)
            w_k = enmap.read_map(fn_k)
        
        assert w_T.ndim == 2 and w_pol.ndim == 2 and w_k.ndim == 2, \
            'Masks must have 2 dimensions'
        
        masks = {'T': w_T, 'pol': w_pol}
    
        # get the filter from the previous script
        fk = np.load(f'{filters_dir}/{sv1}_fk.npy')

        # now we can accumulate the masked, ps'ed sims
        for p, pol in enumerate(('T', 'pol')):
            ps = pss[pol]
            mask = masks[pol]
            for i in range(start, stop):
                fn = f'{filters_dir}/{sv1}_{pol}_filt_masked_spec{i}.npy'
                if not os.path.isfile(fn):
                    eta = utils.concurrent_normal(size=ainfo.nelem, scale=1/np.sqrt(2), seed=[p+1, i], dtype=np.float32, complex=True) # different than flm seed. single prec sims
                    eta[..., :ainfo.lmax + 1] *= np.sqrt(2) # respect reality condition of m=0 
                    eta = curvedsky.almxfl(eta, ps**0.5, ainfo)
                    eta = curvedsky.alm2map(eta, enmap.zeros(mask.shape, mask.wcs), ainfo=ainfo, method='cyl')
                    eta = utils.concurrent_op(np.multiply, w_k, eta, flatten_axes=[-2, -1])
                    eta = utils.rfft(eta, normalize='backward')
                    eta = utils.concurrent_op(np.multiply, fk, eta, flatten_axes=[-2, -1])
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