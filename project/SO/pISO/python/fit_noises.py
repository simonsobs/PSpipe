"""
Plots all kinds of spectra of combination A x B
"""

from pspy import so_spectra, pspy_utils, so_cov, so_map, so_window, so_dict
from math import pi
import numpy as np
import healpy as hp

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
spectra_auto = ["TT", "EE", 'BB'] 
binning_file = "/global/cfs/cdirs/cmb/data/act_dr6/dr6.02/pspipe/binning/binning_50"

d = so_dict.so_dict()
log = log.get_logger(**d)

with open(f"python/plots_1019.yaml", "r") as f:
    plot_info: dict = yaml.safe_load(f)

# Define spectra path and template to read it
try:
    d.read_from_file(sys.argv[1])
    spectra_path = sys.argv[2]
except:
    spectra_path = "/pscratch/sd/m/merrydup/PSpipe_SO/spectra_1019_carlos_150"
    spectra_path = "/pscratch/sd/m/merrydup/PSpipe_SO/spectra_1019_maskglitch"
    spectra_path = "/pscratch/sd/m/merrydup/PSpipe_SO/spectra_1022_all_coadd_xmask_type1"
    d.read_from_file(spectra_path + "/_paramfile.dict")

spectra_cross_template = spectra_path + "/Dl_{}x{}_cross.dat"
spectra_auto_template = spectra_path + "/Dl_{}x{}_auto.dat"
spectra_noise_template = spectra_path + "/Dl_{}x{}_auto.dat"

# Define surveys and arrays to plot
survey_A = "dr6"
arrays_A = d[f"arrays_{survey_A}"]
# arrays_A = ['pa5_f090']
surveys_arrays_A = [f"{survey_A}_{ar}" for ar in arrays_A]

survey_B = "SO"
arrays_B = d[f"arrays_{survey_B}"]
# arrays_B = ['i1_f090', 'i3_f090', 'i4_f090', 'i6_f090']
surveys_arrays_B = [f"{survey_B}_{ar}" for ar in arrays_B]

beams = {
    sv_ar: pspy_utils.naive_binning(
        np.loadtxt(d[f"beam_T_{sv_ar}"]).T[0],
        np.loadtxt(d[f"beam_T_{sv_ar}"]).T[1]
        / (max(np.loadtxt(d[f"beam_T_{sv_ar}"]).T[1])),
        d["binning_file"],
        lmax=8000,
    )[1]
    for sv_ar in surveys_arrays_B
}

surveys = [survey_A, survey_B]
surveys_arrays = surveys_arrays_A + surveys_arrays_B

# Define where and what to plot

save_path = spectra_path + "/plots/"
os.makedirs(save_path, exist_ok=True)
save_path_noises = save_path + "noises/"
os.makedirs(save_path_noises, exist_ok=True)

# Load spectra
Dls_cross = {}
Dls_noise = {}
Dls_auto = {}

fit_auto = True
for sv_ar2 in surveys_arrays_B:
    # ls, Dls_cross[f"{sv_ar1}x{sv_ar2}"] = so_spectra.read_ps(
    #     spectra_cross_template.format(sv_ar1, sv_ar2), spectra=spectra
    # )
    if fit_auto:
            ls, Dls_auto[f"{sv_ar2}x{sv_ar2}"] = so_spectra.read_ps(
                spectra_auto_template.format(sv_ar2, sv_ar2), spectra=spectra
            )
            ls, Dls_cross[f"{surveys_arrays_A[0]}x{sv_ar2}"] = so_spectra.read_ps(
                spectra_cross_template.format(surveys_arrays_A[0], sv_ar2), spectra=spectra
            )
            Dls_noise[f"{sv_ar2}x{sv_ar2}"] = {f: Dls_auto[f"{sv_ar2}x{sv_ar2}"][f] - Dls_cross[f"{surveys_arrays_A[0]}x{sv_ar2}"][f] for f in spectra}
    else:
        try:
            ls, Dls_noise[f"{sv_ar2}x{sv_ar2}"] = so_spectra.read_ps(
                spectra_noise_template.format(sv_ar2, sv_ar2), spectra=spectra
            )
        except:
            pass
fac = ls * (ls + 1) / (2 * np.pi)
clfile = "/pscratch/sd/m/merrydup/pipe0004_BN/spectra/LCDM_spectra.txt"
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
    var = Clb**2 / (2 * lb[l_mask_data] + 1)

    def logp(rms: float, l_knee: float, alpha: float):
        lb_n, Nlb = compute_Nls(
            rms, l_knee, alpha, binning_file=binning_file, spec=spec
        )
        l_mask_model = (LMIN <= lb_n) & (lb_n < LMAX)
        model = Nlb[l_mask_model]
        return -0.5 * np.sum((Clb - model) ** 2 / var)
    # print(logp(30, 1000, -2))
    info = {}
    info["likelihood"] = {"my_like": logp}

    info["params"] = {
        "rms": {
            "prior": {
                "min": 2.0,
                "max": 1000.0,
            },
            "ref": 100.0,
            "proposal": 5.0,
        },
        "l_knee": {
            "prior": {
                "min": LMIN,
                "max": 6000,
            },
            "ref": 1500.0,
            "proposal": 50.0,
        },
        "alpha": {
            "prior": {
                "min": -6.0,
                "max": -0.0,
            },
            "ref": -2.0,
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


fit = True
LMINs = {
    'TT':1000,
    'EE':400,
    'BB':400,
}
LMAX = 8000
if fit:
    log.info("FITS FOR RMS L_KNEE AND ALPHA")
    noise_best_fits = {}
    for f in spectra_auto:
        noise_best_fits[f] = {}
        for sv_ar2 in surveys_arrays_B:
            log.info(f"FITTING {f}_{sv_ar2}")
            noise_best_fits[f][sv_ar2] = fit_nls(
                ls,
                Dls_noise[f"{sv_ar2}x{sv_ar2}"][f] / fac * beams[sv_ar2] ** 2,
                spec=f,
                LMIN=LMINs[f],
                LMAX=LMAX,
            )
    file = open(save_path_noises + "noise_best_fit.yaml", "w")
    yaml.dump(noise_best_fits, file)
    file.close()
else:
    with open(save_path_noises + "noise_best_fit.yaml", "r") as file:
        noise_best_fits: dict = yaml.safe_load(file)

for i, sv_ar2 in enumerate(surveys_arrays_B):
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
            lw=2.5,
            zorder=0,
            label=fr'{f} best-fit $rms=${rms:.1f}arcmin.$\mu$K, $\ell_{{knee}}=${l_knee:.0f}, $\alpha=${alpha:.2f}',
        )

    for lmin in LMINs.values():
        ax.fill_betweenx([0, 1e3], 0, lmin, color='grey', alpha=0.1, zorder=-10)
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
