"""
Plots all kinds of spectra of combination A x B
"""

from pspy import so_spectra, pspy_utils, so_cov, so_map, so_window, so_dict
from math import pi
import numpy as np
import healpy as hp
import scipy as sp
# import pylab as plt
from matplotlib import pyplot as plt
import os
from pspipe_utils import log
from pspy import pspy_utils
from matplotlib.colors import Normalize
import itertools
import yaml
import sys
from cobaya.run import run
from getdist.mcsamples import MCSamplesFromCobaya

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spectra_auto = ["TT", "EE", "BB"]
binning_file = "/global/cfs/cdirs/cmb/data/act_dr6/dr6.02/pspipe/binning/binning_50"

d = so_dict.so_dict()
log = log.get_logger(**d)

# Define spectra path and template to read it
try:
    d.read_from_file(sys.argv[1])
    spectra_path = d['spec_dir']
    yaml_path = d['plots_yaml']
except:
    spectra_path = "/pscratch/sd/m/merrydup/PSpipe_SO/spectra_1019_carlos_150"
    d.read_from_file(spectra_path + "/_paramfile.dict")
    yaml_path = "python/plots_1019.yaml"

with open(yaml_path, "r") as f:
    plot_info: dict = yaml.safe_load(f)['fit_noises.py']


calib_suffix = '_calib' if plot_info['use_cal'] else ''

spectra_cross_template = spectra_path + "/Dl_{}x{}_cross"+f"{calib_suffix}.dat"
spectra_auto_template = spectra_path + "/Dl_{}x{}_auto"+f"{calib_suffix}.dat"
spectra_noise_template = spectra_path + "/Dl_{}x{}_noise"+f"{calib_suffix}.dat"

surveys_arrays = [f"{survey}_{ar}" for survey in d['surveys'] for ar in d[f'arrays_{survey}']]

beams = {
    sv_ar: pspy_utils.naive_binning(
        np.loadtxt(d[f"beam_T_{sv_ar}"]).T[0],
        np.loadtxt(d[f"beam_T_{sv_ar}"]).T[1]
        / (max(np.loadtxt(d[f"beam_T_{sv_ar}"]).T[1])),
        d["binning_file"],
        lmax=d['lmax'],
    )[1]
    for sv_ar in surveys_arrays
}

# Define where and what to plot
save_path_noises = d['plots_dir'] + '/noises/'
os.makedirs(save_path_noises, exist_ok=True)

# Load spectra
Dls_cross = {}
Dls_noise = {}
for sv_ar in surveys_arrays:
    try:
        ls, Dls_noise[f"{sv_ar}x{sv_ar}"] = so_spectra.read_ps(
            spectra_noise_template.format(sv_ar, sv_ar), spectra=spectra
        )
    except:
        log.info(f'{sv_ar} noise not found')
fac = ls * (ls + 1) / (2 * np.pi)
clfile = "/global/cfs/cdirs/sobs/users/merrydup/deep56/proposal_plots/cmb.dat"
l, ps_theory = so_spectra.read_ps(clfile, spectra=spectra)


def compute_Nls(
    rms: float, l_knee: float = 700, alpha: float = -1.4, binning_file=None, spec="TT"
) -> dict[np.ndarray]:
    ls = np.arange(2, 10000)
    Nls = np.full_like(ls, ((rms / 60) ** 2) * ((np.pi / 180) ** 2), dtype=float)
    if l_knee is not None:
        # From SO forecasts paper 1808.07445 (eq. 1)
        Nls *= 1 + (ls / l_knee) ** alpha
    else:
        # Only polarization noise and assumes no cross noise
        Nls = np.zeros_like(ls, dtype=float)
    lb, Nlb = pspy_utils.naive_binning(ls, Nls, binning_file=binning_file, lmax=8000)
    return lb, Nlb


def fit_nls(lb, Clb, spec, LMIN, LMAX):
    """Fit a nls with rms, l_knee and alpha. Clb must not be beam deconvolved.

    Args:
        Clb (_type_): _description_
    """
    l_mask_data = (LMIN <= lb) & (lb < LMAX)
    Clb = Clb[l_mask_data]

    def logp(rms: float, l_knee: float, alpha: float):
        lb_n, Nlb = compute_Nls(
            rms, l_knee, alpha, binning_file=binning_file, spec=spec
        )
        l_mask_model = (LMIN <= lb_n) & (lb_n < LMAX)
        model = Nlb[l_mask_model]
        data = Clb
        var = data**2 / (2 * lb_n[l_mask_model] + 1)
        return -0.5 * np.sum((data - model) ** 2 / var)

    info = {}
    info["likelihood"] = {"my_like": logp}

    lknee_priors = {
        'TT': (-5, -2),
        'EE': (-3, -1),
        'BB': (-3, -1),
    }
    
    info["params"] = {
        "rms": {
            "prior": {
                "min": 2.0,
                "max": 400.0,
            },
            "ref": 20.0,
            "proposal": 2.0,
        },
        "l_knee": {
            "prior": {
                "min": 200,
                "max": 6000,
            },
            "ref": 800.0,
            "proposal": 50.0,
        },
        "alpha": {
            "prior": {
                "min": lknee_priors[spec][0],
                "max": lknee_priors[spec][1]
            },
            "ref": -2.5,
            "proposal": 0.3,
        },
    }
    info["sampler"] = {
        "mcmc": {
            "max_tries": 1e7,
            "burn_in": 20,
            "Rminus1_stop": 0.01,
        }
    }
    updated_info, sampler = run(info)
    gdsamples = MCSamplesFromCobaya(updated_info, sampler.products()["sample"])
    best_fit = gdsamples.getParamBestFitDict(best_sample=True)
    best_fit_dict = {
        "rms": float(best_fit["rms"]),
        "l_knee": float(best_fit["l_knee"]),
        "alpha": float(best_fit["alpha"]),
    }
    return best_fit_dict

def fit_nls_sp(lb, Clb, spec, LMIN, LMAX):
    l_mask_data = (LMIN <= lb) & (lb < LMAX)
    Clb = Clb[l_mask_data]
    
    def model_wrapper(lb, rms, lknee, alpha):
        l, nl = compute_Nls(rms, lknee, alpha, binning_file=binning_file, spec=spec)
        l_mask_model = (LMIN <= l) & (l < LMAX)
        return nl[l_mask_model]
    popt, _ = sp.optimize.curve_fit(
                model_wrapper,
                lb[l_mask_data], 
                Clb, 
                sigma=Clb*(2*lb[l_mask_data] + 1)**0.5, 
                p0=[60, 2000, -3.5], 
                bounds=(
                    [1, 200, -6], 
                    [600, 7000, 0]
                    )
                )
    best_fit_dict = {
        param: float(popt[i]) for i, param in enumerate(["rms", "l_knee", "alpha"])
    }
    return best_fit_dict


fit = True
LMIN_dict = plot_info['lmin_fit']
LMAX = plot_info['lmax_fit']
if fit:
    log.info("FITS FOR RMS L_KNEE AND ALPHA")
    noise_best_fits = {}
    for f in spectra_auto:
        noise_best_fits[f] = {}
        for sv_ar2 in surveys_arrays:
            log.info(f"FITTING {f}_{sv_ar2}")
            noise_best_fits[f][sv_ar2] = fit_nls_sp(
                ls,
                Dls_noise[f"{sv_ar2}x{sv_ar2}"][f] / fac * beams[sv_ar2] ** 2,
                spec=f,
                LMIN=LMIN_dict[f],
                LMAX=LMAX,
            )
    file = open(save_path_noises + "noise_best_fit.yaml", "w")
    yaml.dump(noise_best_fits, file)
    file.close()
else:
    with open(save_path_noises + "noise_best_fit.yaml", "r") as file:
        noise_best_fits: dict = yaml.safe_load(file)

for i, sv_ar2 in enumerate(surveys_arrays):
    fig, ax = plt.subplots(dpi=150, figsize=(10, 4))
    # ax.plot(l, ps_theory[f], color="black", label="theory", alpha=0.4)
    for j, f in enumerate(spectra_auto):
        rms = noise_best_fits[f][sv_ar2]["rms"]
        l_knee = noise_best_fits[f][sv_ar2]["l_knee"]
        alpha = noise_best_fits[f][sv_ar2]["alpha"]
        ax.plot(
            ls,
            Dls_noise[f"{sv_ar2}x{sv_ar2}"][f] / fac * beams[sv_ar2] ** 2,
            label=f"{f}",
            color=f'C{j}',
            mec="black",
            ms=10,
            marker='.',
            ls='--',
            alpha=0.3,
            zorder=-5,
        )
        ax.plot(
            *compute_Nls(
                rms,
                l_knee,
                alpha,
                binning_file=binning_file,
            ),
            color=f'C{j}',
            lw=1.5,
            zorder=0,
            label=fr'{f} best-fit $rms=${rms:.1f}arcmin.$\mu$K, $\ell_{{knee}}=${l_knee:.0f}, $\alpha=${alpha:.2f}',
        )

        ax.fill_betweenx([0, 1e3], 0, LMIN_dict[f], color='grey', alpha=0.15, zorder=-10)
    ax.fill_betweenx([0, 1e3],LMAX, 10000, color='grey', alpha=0.4, zorder=-10)
    ax.set_xlabel(r"$\ell$", fontsize=18)
    ax.set_ylabel(rf"$N^{{{f}}}_\ell$", fontsize=18)
    ax.set_title(sv_ar2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(*plot_info["noise_rms_BxB"]["ylims"][f])
    ax.set_xlim(*plot_info["noise_rms_BxB"]["xlims"][f])
    ax.legend(loc=(1.01, 0.))
    plt.tight_layout()
    plt.savefig(save_path_noises + f"noise_rms_fit_{sv_ar2}")
    plt.close()
