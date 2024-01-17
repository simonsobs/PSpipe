# Given a:
# 1. mask (which may be no mask, i.e. full sky)
# 2. power spectrum (loosely defined, either a proper power spectrum or mnms)
# 3. filter (may be defined in harmonic or fourier space)
# which defines a "scenario"
# 
# We want to be able to:
# 1. Obtain the full sky 2pt and 4pt transfer function
#   - in the case of harmonic filter, these are exactly calculable.
#   - for a fourier space filter, these need to be simulated, which
#   - can be accomplished by including full sky in the scenario, but
#   - with a little help for resolution-matching
# 2. Obtain the spectra under the "scenario"
# 3. Obtain the mode-coupling matrix for the mask and mask^2

from pspy import so_mcm as smc, so_spectra as ssp, pspy_utils as psu
from mnms import noise_models as nm, utils

from pixell import enmap, curvedsky, wcsutils

import numpy as np
from scipy.ndimage import uniform_filter1d as uf1d
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import argparse
from itertools import product
from functools import partial
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--scenario', dest='scenario', type=int, required=True)
parser.add_argument('--num-sims', dest='num_sims', type=int, required=True)
args = parser.parse_args()

masks = ['none', 'mnms', 'pt_src', 'pt_src_sig_sqrt_pixar']
pss = ['white', 'noise_EE_l100_cap', 'noise_TT', 'signal_EE']
filts = ['m_exact', 'l_ish', 'l_ish_m_exact', 'binary_cross']

scenarios = list(product(masks, pss, filts))
scenarios += [
    ('pt_src', 'mnms', 'm_exact'),
    ('pt_src', 'mnms', 'l_ish_m_exact'),
    ('pt_src', 'mnms', 'binary_cross')
    ]

scenario = scenarios[args.scenario]
print(scenario)
num_sims = args.num_sims
lmax = 5400
bin_low, bin_high, bin_cent, _ = psu.read_binning_file('/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/cov_dr6_v4_20231128/binning/BIN_ACTPOL_50_4_SC_large_bin_at_low_ell', lmax)
bin_high += 1 # non-inclusive

ainfo_rect = curvedsky.alm_info(lmax=lmax, layout='rect')
ainfo_tri = curvedsky.alm_info(lmax=lmax, layout='tri')

full_shape, full_wcs = enmap.fullsky_geometry(res=np.deg2rad(1/120), variant='fejer1')
full_shape, full_wcs = enmap.downgrade_geometry(full_shape, full_wcs, 4)

mask_name, ps_name, filt_name = scenario

def bin_spec(specs, bin_low, bin_high, lmin=0):
    good_bins = np.where(bin_high > lmin)
    bin_low = np.maximum(bin_low[good_bins], lmin)
    bin_high = bin_high[good_bins]

    out = np.zeros((*specs.shape[:-1], len(bin_low)))
    for i in range(len(bin_low)):
        out[..., i] = specs[..., bin_low[i]:bin_high[i]].mean(axis=-1) 
    return out

def bin_mat(mats, bin_low, bin_high, lmin=0):
    good_bins = np.where(bin_high > lmin)
    bin_low = np.maximum(bin_low[good_bins], lmin)
    bin_high = bin_high[good_bins]

    out = np.zeros((*mats.shape[:-2], len(bin_low), len(bin_low)))
    for i in range(len(bin_low)):
        for j in range(len(bin_low)):
            out[..., i, j] = mats[..., bin_low[i]:bin_high[i], bin_low[j]:bin_high[j]].mean(axis=(-2, -1))
    return out

def get_expected_spec_func(mcm, tf, p, bin=False, bin_low=None, bin_high=None, lmin=0, pseudo=True):
    if bin:
        if pseudo:
            def f(alpha):
                return bin_spec(mcm @ (tf**alpha * p), bin_low, bin_high, lmin=lmin)
        else:
            def f(alpha):
                return bin_spec(tf**alpha * p, bin_low, bin_high, lmin=lmin)           
    else:
        if pseudo:
            def f(alpha):
                return mcm @ (tf**alpha * p)
        else:
            def f(alpha):
                return tf**alpha * p
    return f

def get_expected_cov_func(mcm, w2, tf, p, c, bin=False, bin_low=None, bin_high=None, lmin=0, pseudo=True, pre_mcm_inv=None, INKA=True):
    if INKA:
        def pseudo_cov(alpha):
            return 0.5 * ((mcm @ (tf**(alpha/2) * p / w2)) + (mcm @ (tf**(alpha/2) * p / w2))[:, None])**2 * c
    else:
        def pseudo_cov(alpha):
            return 0.5 * ((tf**(alpha/2) * p) + (tf**(alpha/2) * p)[:, None])**2 * c

    if bin:
        if pseudo:
            def f(alpha):
                return np.diag(bin_mat(pseudo_cov(alpha), bin_low, bin_high, lmin=lmin))
        else:
            def f(alpha):
                return np.diag(bin_mat(pre_mcm_inv @ pseudo_cov(alpha) @ pre_mcm_inv.T, bin_low, bin_high, lmin=lmin))
    else:
        if pseudo:
            def f(alpha):
                return np.diag(pseudo_cov(alpha))
        else:
            def f(alpha):
                return np.diag(pre_mcm_inv @ pseudo_cov(alpha) @ pre_mcm_inv.T)  
    
    return f

print('building scenario')
# get mask
if mask_name == 'none':
    full_sky = True
    mask = 1

if mask_name == 'mnms':
    full_sky = False
    mask = enmap.read_map('/scratch/gpfs/zatkins/data/simonsobs/mnms/masks/effective_est/pa6_baseline_effective_est_20230816.fits')

if mask_name == 'pt_src':
    full_sky = False
    mask = enmap.read_map('/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/cov_dr6_v4_20231128/windows/window_dr6_pa5_f150_baseline.fits')

if mask_name == 'pt_src_sig_sqrt_pixar':
    full_sky = False
    mask = enmap.read_map('/scratch/gpfs/ACT/dr6v4/maps/dr6v4_20230316/release/cmb_night_pa5_f150_3pass_4way_set0_ivar.fits')
    mask = np.sqrt(np.reciprocal(mask, where=mask!=0) * (mask!=0))
    mask *= enmap.read_map('/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/cov_dr6_v4_20231128/windows/window_dr6_pa5_f150_baseline.fits')
    mask *= mask.pixsizemap()**0.5

if not full_sky:
    mask = mask.downgrade(4)
    assert mask.shape == (2580, 10800)
    assert wcsutils.is_compatible(mask.wcs, full_wcs)

# get ps
if ps_name == 'white':
    isotropic_ps = True
    ps = np.ones(lmax+1)

if ps_name == 'noise_EE_l100_cap':
    isotropic_ps = True
    ps = np.load('/scratch/gpfs/zatkins/data/zatkins2/mnms_client/v3_vs_v4_nl/nl4_uncorrected_set0.npy')[3, 1, :lmax+1] # pa5_f150 EE
    ps[:100] = ps[100]

if ps_name == 'noise_TT':
    isotropic_ps = True
    ps = np.load('/scratch/gpfs/zatkins/data/zatkins2/mnms_client/v3_vs_v4_nl/nl4_uncorrected_set0.npy')[3, 0, :lmax+1] # pa5_f150 TT

if ps_name == 'signal_EE':
    isotropic_ps = True
    lcmb, cmb = ssp.read_ps('/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/cov_dr6_v4_20231128/best_fits/cmb.dat',
                               spectra=['TT', 'TE', 'TB', 'ET', 'BT', 'EE', 'EB', 'BE', 'BB'])
    lcmb = lcmb[:(lmax+1) - 2]

    lfg, fg = ssp.read_ps('/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/cov_dr6_v4_20231128/best_fits/fg_dr6_pa5_f150xdr6_pa5_f150.dat',
                               spectra=['TT', 'TE', 'TB', 'ET', 'BT', 'EE', 'EB', 'BE', 'BB'])
    lfg = lfg[:(lmax+1) - 2]

    assert np.all(lcmb == lfg)
    assert lcmb[0] == 2

    ps = np.zeros(lmax+1)
    ps[2:] = cmb['EE'][:(lmax+1) - 2] + fg['EE'][:(lmax+1) - 2]
    ps[2:] *= 2*np.pi/(lcmb * (lcmb + 1))

if ps_name == 'mnms':
    isotropic_ps = False
    ps = nm.BaseNoiseModel.from_config('act_dr6v4', 'tile_cmbmask', 'pa5a', 'pa5b')
    filt_mask = enmap.read_map('/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/cov_dr6_v4_20231128/windows/kspace_mask_dr6_pa5_f150.fits')
    filt_mask = filt_mask.downgrade(4)
    assert not full_sky
    assert filt_mask.shape == mask.shape
    assert wcsutils.is_compatible(filt_mask.wcs, mask.wcs)

# get filt
if filt_name == 'm_exact':
    harmonic_filt = True

    # to comapre to filt which we build by hand (confirm intuition with kx <-> m on fullsky)
    fn = '/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/flm_m_exact.npy'
    if not os.path.isfile(fn):
        k = enmap.ones((full_shape[0], full_shape[-1]//2 + 1), full_wcs)
        k[:, :180] = 0

        nsim = 300
        _filt = 0
        for i in range(nsim):
            alm = curvedsky.rand_alm(np.ones(lmax + 1), ainfo=ainfo_tri, seed=2_000_000+i)
            imap = curvedsky.alm2map(alm, enmap.zeros(full_shape, full_wcs), ainfo=ainfo_tri, method='cyl')
            imap = utils.irfft(k * utils.rfft(imap, normalize='backward'), normalize='backward')
            _filt += np.abs(curvedsky.map2alm(imap, ainfo=ainfo_rect, method='cyl').reshape(lmax + 1, -1))**2
        _filt /= nsim
        _filt **= 0.5
        np.save(fn, _filt)
    else:
        pass

    filt = np.abs(curvedsky.prepare_alm(ainfo=ainfo_rect)[0].reshape(lmax+1, -1) + 1)
    filt[:180] = 0
    filt = curvedsky.transfer_alm(ainfo_rect, filt.reshape(-1), ainfo_tri)

if filt_name == 'l_ish':
    harmonic_filt = True

    fn = '/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/flm_l_ish.npy'
    if not os.path.isfile(fn):
        k = enmap.ones((full_shape[0], full_shape[-1]//2 + 1), full_wcs)
        k[:90] = 0
        k[-89:] = 0

        nsim = 300
        filt = 0
        for i in range(nsim):
            alm = curvedsky.rand_alm(np.ones(lmax + 1), ainfo=ainfo_tri, seed=1_000_000+i)
            imap = curvedsky.alm2map(alm, enmap.zeros(full_shape, full_wcs), ainfo=ainfo_tri, method='cyl')
            imap = utils.irfft(k * utils.rfft(imap, normalize='backward'), normalize='backward')
            filt += np.abs(curvedsky.map2alm(imap, ainfo=ainfo_rect, method='cyl').reshape(lmax + 1, -1))**2
        filt /= nsim
        filt **= 0.5
        filt[:, :180] = 0 # avoid low ell spurious ringing
        np.save(fn, filt)
    else:
        filt = np.load(fn)
    filt = curvedsky.transfer_alm(ainfo_rect, filt.reshape(-1), ainfo_tri)

if filt_name == 'l_ish_m_exact':
    harmonic_filt = True

    fn = '/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/flm_l_ish_m_exact.npy'
    if not os.path.isfile(fn):
        k = enmap.ones((full_shape[0], full_shape[-1]//2 + 1), full_wcs)
        k[:90] = 0
        k[-89:] = 0
        k[:, :180] = 0

        nsim = 300
        _filt = 0
        for i in range(nsim):
            alm = curvedsky.rand_alm(np.ones(lmax + 1), ainfo=ainfo_tri, seed=2_000_000+i)
            imap = curvedsky.alm2map(alm, enmap.zeros(full_shape, full_wcs), ainfo=ainfo_tri, method='cyl')
            imap = utils.irfft(k * utils.rfft(imap, normalize='backward'), normalize='backward')
            _filt += np.abs(curvedsky.map2alm(imap, ainfo=ainfo_rect, method='cyl').reshape(lmax + 1, -1))**2
        _filt /= nsim
        _filt **= 0.5
        np.save(fn, _filt)
    else:
        pass
    
    fn = '/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/flm_l_ish.npy'
    if not os.path.isfile(fn):
        k = enmap.ones((full_shape[0], full_shape[-1]//2 + 1), full_wcs)
        k[:90] = 0
        k[-89:] = 0

        nsim = 300
        filt = 0
        for i in range(nsim):
            alm = curvedsky.rand_alm(np.ones(lmax + 1), ainfo=ainfo_tri, seed=1_000_000+i)
            imap = curvedsky.alm2map(alm, enmap.zeros(full_shape, full_wcs), ainfo=ainfo_tri, method='cyl')
            imap = utils.irfft(k * utils.rfft(imap, normalize='backward'), normalize='backward')
            filt += np.abs(curvedsky.map2alm(imap, ainfo=ainfo_rect, method='cyl').reshape(lmax + 1, -1))**2
        filt /= nsim
        filt **= 0.5
        filt[:, :180] = 0 # avoid low ell spurious ringing
        np.save(fn, filt)
    else:
        filt = np.load(fn)
    filt[:180] = 0
    filt = curvedsky.transfer_alm(ainfo_rect, filt.reshape(-1), ainfo_tri)

if filt_name == 'binary_cross':
    harmonic_filt = False

    if full_sky:
        filt = enmap.ones((full_shape[0], full_shape[-1]//2 + 1), full_wcs)
        filt[:90] = 0
        filt[-89:] = 0
        filt[:, :180] = 0
    else:
        filt = enmap.ones((mask.shape[0], mask.shape[-1]//2 + 1), mask.wcs)
        filt[:43] = 0
        filt[-42:] = 0
        filt[:, :180] = 0

# get fullsky tf
if harmonic_filt:
    fl2 = utils.alm2cl(filt*(1+0j), filt*(1+0j))
    fl4 = utils.alm2cl(filt**2*(1+0j), filt**2*(1+0j))
else:
    fn = '/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/flm_l_ish.npy'
    if not os.path.isfile(fn):
        k = enmap.ones((full_shape[0], full_shape[-1]//2 + 1), full_wcs)
        k[:90] = 0
        k[-89:] = 0

        nsim = 300
        _filt = 0
        for i in range(nsim):
            alm = curvedsky.rand_alm(np.ones(lmax + 1), ainfo=ainfo_tri, seed=1_000_000+i)
            imap = curvedsky.alm2map(alm, enmap.zeros(full_shape, full_wcs), ainfo=ainfo_tri, method='cyl')
            imap = utils.irfft(k * utils.rfft(imap, normalize='backward'), normalize='backward')
            _filt += np.abs(curvedsky.map2alm(imap, ainfo=ainfo_rect, method='cyl').reshape(lmax + 1, -1))**2
        _filt /= nsim
        _filt **= 0.5
        _filt[:, :180] = 0 # avoid low ell spurious ringing
        np.save(fn, filt)
    else:
        _filt = np.load(fn)
    _filt[:180] = 0
    _filt = curvedsky.transfer_alm(ainfo_rect, _filt.reshape(-1), ainfo_tri)

    _fl2 = utils.alm2cl(_filt*(1+0j), _filt*(1+0j))
    _fl4 = utils.alm2cl(_filt**2*(1+0j), _filt**2*(1+0j))

    fl2 = np.zeros_like(_fl2)
    fl2[np.where(_fl2 > 1e-3)] = savgol_filter(_fl2[np.where(_fl2 > 1e-3)], 75, 4)

    fl4 = np.zeros_like(_fl4)
    fl4[np.where(_fl4 > 1e-3)] = savgol_filter(_fl4[np.where(_fl4 > 1e-3)], 75, 4)

lmin2_fit = np.max(np.where(fl2 <= 0.5)[0]) + 1
bmin2_fit = np.min(np.where(bin_low >= lmin2_fit)[0])
lmin4_fit = np.max(np.where(fl4 <= 0.5)[0]) + 1
bmin4_fit = np.min(np.where(bin_low >= lmin4_fit)[0])

print('getting spectra')
# now we have a mask, a ps, and a filter, so we can obtain the spectra
for i in range(num_sims):
    fn = f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_ps_{ps_name}_filt_{filt_name}_spec{i}.npy'
    if os.path.exists(fn):
        continue
    else:
        if isotropic_ps:
            eta = curvedsky.rand_alm(ps, ainfo=ainfo_tri, seed=i)
            if harmonic_filt:
                eta *= filt
                if full_sky:
                    pass
                else:
                    eta = curvedsky.alm2map(eta, enmap.zeros(mask.shape, mask.wcs), ainfo=ainfo_tri, method='cyl')
                    eta *= mask
                    eta = curvedsky.map2alm(eta, ainfo=ainfo_tri, method='cyl')
            else:
                if full_sky:
                    eta = curvedsky.alm2map(eta, enmap.zeros(full_shape, full_wcs), ainfo=ainfo_tri, method='cyl')
                else:
                    eta = curvedsky.alm2map(eta, enmap.zeros(mask.shape, mask.wcs), ainfo=ainfo_tri, method='cyl')
                eta = utils.irfft(filt * utils.rfft(eta, normalize='backward'), normalize='backward')
                eta *= mask
                eta = curvedsky.map2alm(eta, ainfo=ainfo_tri, method='cyl')
        else:
            eta = ps.get_sim(0, i, 5400, check_on_disk=True)[1, 0, 1] # pa5_f150 Q
            assert eta.shape == mask.shape
            assert not full_sky
            eta = enmap.ndmap(eta, mask.wcs)
            eta *= filt_mask
            if harmonic_filt:
                eta = curvedsky.map2alm(eta, ainfo=ainfo_tri, method='cyl')
                eta *= filt
                eta = curvedsky.alm2map(eta, enmap.zeros(mask.shape, mask.wcs), ainfo=ainfo_tri, method='cyl')
                eta *= mask
                eta = curvedsky.map2alm(eta, ainfo=ainfo_tri, method='cyl')
            else:
                eta = utils.irfft(filt * utils.rfft(eta, normalize='backward'), normalize='backward')
                eta *= mask
                eta = curvedsky.map2alm(eta, ainfo=ainfo_tri, method='cyl')
        
        np.save(fn, utils.alm2cl(eta))

print('getting couplings')
# get the couplings
if not full_sky:
    fn = f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}1_coupling.npy'
    if os.path.exists(fn):
        mcm = np.load(fn)
    else:
        mask_alm = curvedsky.map2alm(mask, ainfo=ainfo_tri, method='cyl')
        mcm = smc.coupling_block('00', mask_alm.astype(np.complex128, copy=False), lmax, input_alm=True)
        np.save(fn, mcm)
    mcm /= 4*np.pi
    mcm *= (2*np.arange(lmax + 1) + 1)
    mcm_inv = np.linalg.inv(mcm)

    fn = f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}2_coupling.npy'
    if os.path.exists(fn):
        coupling = np.load(fn)
    else:
        mask_alm = curvedsky.map2alm(mask**2, ainfo=ainfo_tri, method='cyl')
        coupling = smc.coupling_block('00', mask_alm.astype(np.complex128, copy=False), lmax, input_alm=True)
        np.save(fn, coupling)
    coupling /= 4*np.pi

    fn = f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_w2.npy'
    if os.path.exists(fn):
        w2 = np.load(fn)
    else:
        w2 = np.sum(mask**2 * mask.pixsizemap()) / 4 / np.pi
        np.save(fn, w2)
else:
    mcm = np.eye(lmax + 1)
    mcm_inv = mcm.copy()
    coupling = mcm.copy() / (2*np.arange(lmax + 1) + 1)
    w2 = 1

# now we do the fits
print('getting fits and results')

fn = f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_ps_{ps_name}_filt_{filt_name}_res_dict.npy'
if os.path.exists(fn):
    pass
else:  
    res_dict = {}
    res_dict['lmin2_fit'] = lmin2_fit
    res_dict['bmin2_fit'] = bmin2_fit
    res_dict['lmin4_fit'] = lmin4_fit
    res_dict['bmin4_fit'] = bmin4_fit

    def f(x, alpha, func, xmin, den):
        return np.divide(func(alpha)[xmin:], den[xmin:], where=den[xmin:]!=0, out=np.zeros_like(den[xmin:]))

    # get spectra
    pseudos = []
    for i in range(num_sims):
        pseudos.append(np.load(f'/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/misc/kspace_dependence/mask_{mask_name}_ps_{ps_name}_filt_{filt_name}_spec{i}.npy'))
    pseudos = np.array(pseudos)
    binned_pseudos = bin_spec(pseudos, bin_low, bin_high, lmin=0)    

    # pseudo spectra
    func = get_expected_spec_func(mcm, fl2, ps, bin=False, pseudo=True)
    den = func(0)

    res_dict['pseudo_mean'] = pseudos.mean(axis=0) 
    res_dict['pseudo_var'] = pseudos.var(axis=0, ddof=1) / num_sims
    res_dict['pseudo_den'] = den

    ydata = np.divide(res_dict['pseudo_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['pseudo_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    popt_pseudo, pcov_pseudo = curve_fit(partial(f, func=func, xmin=lmin2_fit, den=den), 1, ydata[lmin2_fit:], sigma=yerr[lmin2_fit:])
    pseudo_best_fit = f(1, popt_pseudo[0], func, 0, den)

    res_dict['pseudo_alpha'] = popt_pseudo[0]
    res_dict['pseudo_alpha_err'] = pcov_pseudo[0, 0]**0.5 
    res_dict['pseudo_best_fit'] = pseudo_best_fit
    res_dict['pseudo_err'] = ydata - pseudo_best_fit
    res_dict['pseudo_stderr'] = np.divide(ydata - pseudo_best_fit, yerr, where=yerr!=0, out=np.zeros_like(yerr))

    # binned pseudo spectra
    func = get_expected_spec_func(mcm, fl2, ps, bin=True, bin_low=bin_low, bin_high=bin_high, lmin=0, pseudo=True)
    den = func(0)

    res_dict['binned_pseudo_mean'] = binned_pseudos.mean(axis=0) 
    res_dict['binned_pseudo_var'] = binned_pseudos.var(axis=0, ddof=1) / num_sims
    res_dict['binned_pseudo_den'] = den

    ydata = np.divide(res_dict['binned_pseudo_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['binned_pseudo_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    popt_binned_pseudo, pcov_binned_pseudo = curve_fit(partial(f, func=func, xmin=bmin2_fit, den=den), 1, ydata[bmin2_fit:], sigma=yerr[bmin2_fit:])
    binned_pseudo_best_fit = f(1, popt_binned_pseudo[0], func, 0, den)

    res_dict['binned_pseudo_alpha'] = popt_binned_pseudo[0]
    res_dict['binned_pseudo_alpha_err'] = pcov_binned_pseudo[0, 0]**0.5 
    res_dict['binned_pseudo_best_fit'] = binned_pseudo_best_fit
    res_dict['binned_pseudo_err'] = ydata - binned_pseudo_best_fit
    res_dict['binned_pseudo_stderr'] = np.divide(ydata - binned_pseudo_best_fit, yerr, where=yerr!=0, out=np.zeros_like(yerr))

    # spectra and binned spectra
    eff_tf2 = fl2**popt_pseudo[0]
    eff_tf2_inv = np.divide(1, eff_tf2, where=eff_tf2!=0, out=np.full(eff_tf2.shape, 1_000_000, dtype=np.float64))
    pre_mcm_inv = np.einsum('r, rc -> rc', eff_tf2_inv, mcm_inv) # tf_inv @ mcm_inv

    specs = np.einsum('...Ll, ...il -> ...iL', pre_mcm_inv, pseudos)
    binned_specs = bin_spec(specs, bin_low, bin_high, lmin=0)    

    func = get_expected_spec_func(mcm, fl2, ps, bin=False, pseudo=False)
    den = func(0)

    res_dict['spec_mean'] = specs.mean(axis=0) 
    res_dict['spec_var'] = specs.var(axis=0, ddof=1) / num_sims
    res_dict['spec_den'] = den

    func = get_expected_spec_func(mcm, fl2, ps, bin=True, bin_low=bin_low, bin_high=bin_high, lmin=0, pseudo=False)
    den = func(0)

    res_dict['binned_spec_mean'] = binned_specs.mean(axis=0) 
    res_dict['binned_spec_var'] = binned_specs.var(axis=0, ddof=1) / num_sims
    res_dict['binned_spec_den'] = den

    # pseudo cov
    pseudo_cov_diags = num_sims/(num_sims-1) * (pseudos - pseudos.mean(axis=0))**2
    binned_pseudo_cov_diags = num_sims/(num_sims-1) * (binned_pseudos - binned_pseudos.mean(axis=0))**2

    func2 = get_expected_cov_func(mcm, w2, fl2, ps, coupling, pseudo=True, bin=False)
    den = func2(0)

    res_dict['pseudo_cov_diag_mean'] = pseudo_cov_diags.mean(axis=0)
    res_dict['pseudo_cov_diag_var'] = pseudo_cov_diags.var(axis=0, ddof=1) / num_sims
    res_dict['pseudo_cov_diag_den'] = den

    ydata = np.divide(res_dict['pseudo_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['pseudo_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    popt2, pcov2 = curve_fit(partial(f, func=func2, xmin=lmin4_fit, den=den), 1, ydata[lmin4_fit:], sigma=yerr[lmin4_fit:])
    pseudo_cov_diag_best_fit2 = f(1, popt2[0], func2, 0, den)

    res_dict['pseudo_cov_diag_alpha2'] = popt2[0]
    res_dict['pseudo_cov_diag_alpha2_err'] = pcov2[0, 0]**0.5 
    res_dict['pseudo_cov_diag_best_fit2'] = pseudo_cov_diag_best_fit2
    res_dict['pseudo_cov_diag_err2'] = ydata - pseudo_cov_diag_best_fit2
    res_dict['pseudo_cov_diag_stderr2'] = np.divide(ydata - pseudo_cov_diag_best_fit2, yerr, where=yerr!=0, out=np.zeros_like(yerr))

    func4 = get_expected_cov_func(mcm, w2, fl4, ps, coupling, pseudo=True, bin=False)
    assert np.all(den == func4(0))
    popt4, pcov4 = curve_fit(partial(f, func=func4, xmin=lmin4_fit, den=den), 1, ydata[lmin4_fit:], sigma=yerr[lmin4_fit:])
    pseudo_cov_diag_best_fit4 = f(1, popt4[0], func4, 0, den)

    res_dict['pseudo_cov_diag_alpha4'] = popt4[0]
    res_dict['pseudo_cov_diag_alpha4_err'] = pcov4[0, 0]**0.5 
    res_dict['pseudo_cov_diag_best_fit4'] = pseudo_cov_diag_best_fit4
    res_dict['pseudo_cov_diag_err4'] = ydata - pseudo_cov_diag_best_fit4
    res_dict['pseudo_cov_diag_stderr4'] = np.divide(ydata - pseudo_cov_diag_best_fit4, yerr, where=yerr!=0, out=np.zeros_like(yerr))

    # binned pseudo cov
    func2 = get_expected_cov_func(mcm, w2, fl2, ps, coupling, pseudo=True, bin=True, bin_low=bin_low, bin_high=bin_high, lmin=0)
    den = func2(0)

    res_dict['binned_pseudo_cov_diag_mean'] = binned_pseudo_cov_diags.mean(axis=0)
    res_dict['binned_pseudo_cov_diag_var'] = binned_pseudo_cov_diags.var(axis=0, ddof=1) / num_sims
    res_dict['binned_pseudo_cov_diag_den'] = den

    ydata = np.divide(res_dict['binned_pseudo_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['binned_pseudo_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    popt2, pcov2 = curve_fit(partial(f, func=func2, xmin=bmin4_fit, den=den), 1, ydata[bmin4_fit:], sigma=yerr[bmin4_fit:])
    binned_pseudo_cov_diag_best_fit2 = f(1, popt2[0], func2, 0, den)

    res_dict['binned_pseudo_cov_diag_alpha2'] = popt2[0]
    res_dict['binned_pseudo_cov_diag_alpha2_err'] = pcov2[0, 0]**0.5 
    res_dict['binned_pseudo_cov_diag_best_fit2'] = binned_pseudo_cov_diag_best_fit2
    res_dict['binned_pseudo_cov_diag_err2'] = ydata - binned_pseudo_cov_diag_best_fit2
    res_dict['binned_pseudo_cov_diag_stderr2'] = np.divide(ydata - binned_pseudo_cov_diag_best_fit2, yerr, where=yerr!=0, out=np.zeros_like(yerr))

    func4 = get_expected_cov_func(mcm, w2, fl4, ps, coupling, pseudo=True, bin=True, bin_low=bin_low, bin_high=bin_high, lmin=0)
    assert np.all(den == func4(0))
    popt4, pcov4 = curve_fit(partial(f, func=func4, xmin=bmin4_fit, den=den), 1, ydata[bmin4_fit:], sigma=yerr[bmin4_fit:])
    binned_pseudo_cov_diag_best_fit4 = f(1, popt4[0], func4, 0, den)

    res_dict['binned_pseudo_cov_diag_alpha4'] = popt4[0]
    res_dict['binned_pseudo_cov_diag_alpha4_err'] = pcov4[0, 0]**0.5 
    res_dict['binned_pseudo_cov_diag_best_fit4'] = binned_pseudo_cov_diag_best_fit4
    res_dict['binned_pseudo_cov_diag_err4'] = ydata - binned_pseudo_cov_diag_best_fit4
    res_dict['binned_pseudo_cov_diag_stderr4'] = np.divide(ydata - binned_pseudo_cov_diag_best_fit4, yerr, where=yerr!=0, out=np.zeros_like(yerr))

    # spec cov
    spec_cov_diags = num_sims/(num_sims-1) * (specs - specs.mean(axis=0))**2
    binned_spec_cov_diags = num_sims/(num_sims-1) * (binned_specs - binned_specs.mean(axis=0))**2
    
    func2 = get_expected_cov_func(mcm, w2, fl2, ps, coupling, pseudo=False, pre_mcm_inv=pre_mcm_inv, bin=False)
    den = func2(0)
    
    res_dict['spec_cov_diag_mean'] = spec_cov_diags.mean(axis=0)
    res_dict['spec_cov_diag_var'] = spec_cov_diags.var(axis=0, ddof=1) / num_sims
    res_dict['spec_cov_diag_den'] = den

    ydata = np.divide(res_dict['spec_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['spec_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    popt2, pcov2 = curve_fit(partial(f, func=func2, xmin=lmin4_fit, den=den), 1, ydata[lmin4_fit:], sigma=yerr[lmin4_fit:])
    spec_cov_diag_best_fit2 = f(1, popt2[0], func2, 0, den)

    res_dict['spec_cov_diag_alpha2'] = popt2[0]
    res_dict['spec_cov_diag_alpha2_err'] = pcov2[0, 0]**0.5 
    res_dict['spec_cov_diag_best_fit2'] = spec_cov_diag_best_fit2
    res_dict['spec_cov_diag_err2'] = ydata - spec_cov_diag_best_fit2
    res_dict['spec_cov_diag_stderr2'] = np.divide(ydata - spec_cov_diag_best_fit2, yerr, where=yerr!=0, out=np.zeros_like(yerr))

    func4 = get_expected_cov_func(mcm, w2, fl4, ps, coupling, pseudo=False, pre_mcm_inv=pre_mcm_inv, bin=False)
    assert np.all(den == func4(0))
    popt4, pcov4 = curve_fit(partial(f, func=func4, xmin=lmin4_fit, den=den), 1, ydata[lmin4_fit:], sigma=yerr[lmin4_fit:])
    spec_cov_diag_best_fit4 = f(1, popt4[0], func4, 0, den)

    res_dict['spec_cov_diag_alpha4'] = popt4[0]
    res_dict['spec_cov_diag_alpha4_err'] = pcov4[0, 0]**0.5 
    res_dict['spec_cov_diag_best_fit4'] = spec_cov_diag_best_fit4
    res_dict['spec_cov_diag_err4'] = ydata - spec_cov_diag_best_fit4
    res_dict['spec_cov_diag_stderr4'] = np.divide(ydata - spec_cov_diag_best_fit4, yerr, where=yerr!=0, out=np.zeros_like(yerr))

    # binned spec cov
    func2 = get_expected_cov_func(mcm, w2, fl2, ps, coupling, pseudo=False, pre_mcm_inv=pre_mcm_inv, bin=True, bin_low=bin_low, bin_high=bin_high, lmin=0)
    den = func2(0)

    res_dict['binned_spec_cov_diag_mean'] = binned_spec_cov_diags.mean(axis=0)
    res_dict['binned_spec_cov_diag_var'] = binned_spec_cov_diags.var(axis=0, ddof=1) / num_sims
    res_dict['binned_spec_cov_diag_den'] = den

    ydata = np.divide(res_dict['binned_spec_cov_diag_mean'], den, where=den!=0, out=np.zeros_like(den))
    yerr = np.divide(res_dict['binned_spec_cov_diag_var']**0.5, den, where=den!=0, out=np.zeros_like(den))
    popt2, pcov2 = curve_fit(partial(f, func=func2, xmin=bmin4_fit, den=den), 1, ydata[bmin4_fit:], sigma=yerr[bmin4_fit:])
    binned_spec_cov_diag_best_fit2 = f(1, popt2[0], func2, 0, den)

    res_dict['binned_spec_cov_diag_alpha2'] = popt2[0]
    res_dict['binned_spec_cov_diag_alpha2_err'] = pcov2[0, 0]**0.5 
    res_dict['binned_spec_cov_diag_best_fit2'] = binned_spec_cov_diag_best_fit2
    res_dict['binned_spec_cov_diag_err2'] = ydata - binned_spec_cov_diag_best_fit2
    res_dict['binned_spec_cov_diag_stderr2'] = np.divide(ydata - binned_spec_cov_diag_best_fit2, yerr, where=yerr!=0, out=np.zeros_like(yerr))

    func4 = get_expected_cov_func(mcm, w2, fl4, ps, coupling, pseudo=False, pre_mcm_inv=pre_mcm_inv, bin=True, bin_low=bin_low, bin_high=bin_high, lmin=0)
    assert np.all(den == func4(0))
    popt4, pcov4 = curve_fit(partial(f, func=func4, xmin=bmin4_fit, den=den), 1, ydata[bmin4_fit:], sigma=yerr[bmin4_fit:])
    binned_spec_cov_diag_best_fit4 = f(1, popt4[0], func4, 0, den)

    res_dict['binned_spec_cov_diag_alpha4'] = popt4[0]
    res_dict['binned_spec_cov_diag_alpha4_err'] = pcov4[0, 0]**0.5 
    res_dict['binned_spec_cov_diag_best_fit4'] = binned_spec_cov_diag_best_fit4
    res_dict['binned_spec_cov_diag_err4'] = ydata - binned_spec_cov_diag_best_fit4
    res_dict['binned_spec_cov_diag_stderr4'] = np.divide(ydata - binned_spec_cov_diag_best_fit4, yerr, where=yerr!=0, out=np.zeros_like(yerr))

    np.save(fn, res_dict)