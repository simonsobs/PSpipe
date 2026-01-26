"""
This script is used to calibrate the polarization efficiencies
for each array with respect to a LCDM bestfit (default: cmb.dat)
"""
from pspy import so_dict, pspy_utils, so_spectra, so_cov
from getdist.mcsamples import loadMCSamples
from pspipe_utils import best_fits
import matplotlib.pyplot as plt
import getdist.plots as gdplt
from cobaya.run import run
import numpy as np
import sys
import yaml
import scipy.stats as ss

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# log calib infos from calib yaml file
with open(d['poleff_yaml'], "r") as f:
    calib_dict: dict = yaml.safe_load(f)
calib_infos: dict = calib_dict['get_polar_eff_LCDM.py']

# Set up directories
spec_dir = d['spec_dir']
cov_dir = d['cov_dir']
bestfit_dir = d['best_fits_dir']
mcm_dir = f'old_cov/mcm/'
planck_corr = False
use_leakage_cov = False

if planck_corr:
    spec_dir = "spectra_leak_corr_planck_bias_corr"

output_dir = d['poleff_dir']
pspy_utils.create_directory(output_dir)
plots_dir = d['plots_dir'] + '/poleff/'
pspy_utils.create_directory(plots_dir)

ps_filename = f"{bestfit_dir}/cmb.dat"

# Load paramfiles info
surveys = d["surveys"]
arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
lmax_paramfile = d["lmax"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

# Load ps theory file
l_th, ps_th = so_spectra.read_ps(ps_filename, spectra=spectra)
l_th = l_th.astype(int)

# Load foreground dict and dust prior, then normalize (with polarized dust normalized to 1)
do_bandpass_integration = d["do_bandpass_integration"]
fg_components = d["fg_components"]
fg_params = d["fg_params"]

dust_priors = {
    "EE": {"loc": fg_params["a_gee"], "scale": 0.008},
    "TE": {"loc": fg_params["a_gte"], "scale": 0.015}
}

fg_components["ee"] = ["dust"]
fg_components["te"] = ["dust"]
fg_params["a_gee"] = 1.
fg_params["a_gte"] = 1.


sets_to_measure = calib_infos['sets_to_measure']
use_cal = calib_infos['use_cal']
cal_suffix = '_calib' if use_cal else ''

passbands = {}
for sv_ar in sets_to_measure:
        freq_info = d[f"freq_info_{sv_ar}"]
        if do_bandpass_integration:
            nu_ghz, pb = np.loadtxt(freq_info["passband"]).T        
            # delete any 0-freq entries
            good_idxs = nu_ghz > 0
            nu_ghz = nu_ghz[good_idxs]
            pb = pb[good_idxs]
        else:
            nu_ghz, pb = np.array([freq_info["freq_tag"]]), np.array([1.])

        passbands[f"{sv_ar}"] = [nu_ghz, pb]

fg_dict = best_fits.get_foreground_dict(
    l_th, 
    passbands, 
    fg_components, 
    fg_params, 
    d["fg_norm"],
)

# Define useful functions
def get_model(cmb_th, fg_th, Bbl, dust_amp, pol_eff, mode):
    ps_theory = (cmb_th + dust_amp * fg_th) * pol_eff ** mode.count("E")
    return Bbl @ ps_theory

modes = ['EE', 'TE']

ylims = {
    'TE': (-200, 200),
    'EE': (-20, 70),
}

pol_eff_mean, pol_eff_std, dust_mean, dust_std = {}, {}, {}, {}
for sv_ar in sets_to_measure:
        lmin, lmax = calib_infos["ell_ranges"][sv_ar]
        for spectrum in modes:
            print(spectrum, sv_ar, calib_infos["ell_ranges"][f"{sv_ar}"])

            # Load ps and cov
            spec_name = f"{sv_ar}x{sv_ar}"
            lb, ps = so_spectra.read_ps(f"{spec_dir}/Dl_{spec_name}_cross{cal_suffix}.dat", spectra=spectra)
            cov = np.load(f"{cov_dir}/analytic_cov_{spec_name}_{spec_name}.npy")
            if use_leakage_cov:
                leakage_cov = np.load(f"{cov_dir}/leakage_cov_{spec_name}_{spec_name}.npy")
                cov += leakage_cov
            # Select the spectrum
            ps = ps[spectrum]
            n_bins = len(lb)
            
            cov = so_cov.selectblock(cov, spectra, n_bins=n_bins, block=spectrum+spectrum)

            # Multipole cuts
            id = np.where((lb >= lmin) & (lb <= lmax))[0]
            ps_plot = ps.copy()
            ps = ps[id]
            cov_plot = cov.copy()
            cov = cov[np.ix_(id, id)]
            invcov_plot = np.linalg.inv(cov_plot)
            invcov = np.linalg.inv(cov)
            # Load Bbl
            spin_pair = "spin2xspin2" if spectrum == "EE" else ("spin0xspin2" if spectrum == "TE" else None)
            if spin_pair is None:
                raise ValueError("spectrum must be set to either 'EE' or 'TE'")
            Bbl = np.load(f"{mcm_dir}/{spec_name}_Bbl_{spin_pair}.npy")
            Bbl = Bbl[:n_bins, :lmax_paramfile]

            # Get theory
            cmb_th = ps_th[spectrum][:lmax_paramfile]
            fg_th = fg_dict[spectrum.lower(), "dust", f"{sv_ar}", f"{sv_ar}"][:lmax_paramfile]
            # Define loglike
            def loglike(pol_eff, dust_amp):
                theory = get_model(cmb_th, fg_th, Bbl, dust_amp, pol_eff, mode=spectrum)
                theory = theory[id]
                residual = ps - theory
                chi2 = residual @ invcov @ residual
                return -0.5 * chi2

            loc, scale = dust_priors[spectrum]["loc"], dust_priors[spectrum]["scale"]

            # Prepare MCMC sampling
            info = {
                "likelihood": {
                    "pol_eff": loglike
                },
                "params": {
                    "pol_eff": {
                        "prior": {
                            "min": 0.5,
                            "max": 1.5
                        },
                        "latex": r"\epsilon_\mathrm{pol}^{%s}" % f"{sv_ar}".replace("_", "\_")
                    },
                    "dust_amp": {
                        "prior": {
                            "dist": "norm",
                            "loc": loc,
                            "scale": scale
                        },
                        "latex": r"A_\mathrm{dust}^{%s}" % spectrum
                    },
                },
                "sampler": {
                    "mcmc": {
                        "max_tries": 10**6,
                        "Rminus1_stop": 0.015,
                        "Rminus1_cl_stop": 0.015
                    }
                },
                "output": f"{output_dir}/chain_{spectrum}_{sv_ar}",
                "force": True,
            }

            updated_info, sampler = run(info)

            samples = loadMCSamples(f"{output_dir}/chain_{spectrum}_{sv_ar}", settings={"ignore_rows": 0.5})
            pol_eff_mean[sv_ar, spectrum] = samples.mean("pol_eff")
            pol_eff_std[sv_ar, spectrum] = np.sqrt(samples.cov(["pol_eff"])[0, 0])
            dust_mean[sv_ar, spectrum] = samples.mean("dust_amp")
            dust_std[sv_ar, spectrum] = np.sqrt(samples.cov(["dust_amp"])[0, 0])

            # gdplot = gdplt.get_subplot_plotter()
            # gdplot.triangle_plot(samples, ["pol_eff", "dust_amp"], filled=True, title_limit=1)
            # plt.savefig(f"{plots_dir}/posterior_dist_{spectrum}_{sv_ar}.png", dpi=300, bbox_inches="tight")
            # plt.clf()
            # plt.close()
            
            fig, ax = plt.subplots(
                2,
                gridspec_kw={"hspace": 0, "height_ratios": (2, 1)},
                figsize=(8, 6),
                sharex=True,
            )
            
            errorbar = np.sqrt(cov_plot.diagonal())
            model = get_model(cmb_th, fg_th, Bbl, dust_mean[sv_ar, spectrum], 1., spectrum)
            data = ps_plot
            bestfit = ps_plot / (pol_eff_mean[sv_ar, spectrum] ** spectrum.count("E"))
            bestfit_upper = ps_plot / ((pol_eff_mean[sv_ar, spectrum] + pol_eff_std[sv_ar, spectrum]) ** spectrum.count("E"))
            bestfit_lower = ps_plot / ((pol_eff_mean[sv_ar, spectrum] - pol_eff_std[sv_ar, spectrum]) ** spectrum.count("E"))
            pte_uncal = 1 - ss.chi2(len(lb[id])).cdf((data[id] - model[id]) @ invcov @ (data[id] - model[id]))
            pte_cal = 1 - ss.chi2(len(lb[id]) - 2).cdf((bestfit[id] - model[id]) @ invcov @ (bestfit[id] - model[id]))  # FIXME : should there be - 2 in the ndof ?

            ax[0].errorbar(
                lb,
                model,
                color="black",
                label='theory',
                ls="-",
                alpha=1.,
                marker='',
                linewidth=1,
                elinewidth=1,
            )
            ax[0].errorbar(
                lb-4,
                data,
                errorbar,
                color="tab:grey",
                label='data',
                ls="-",
                alpha=.6,
                marker='.',
                mfc='white',
                mec='tab:blue',
                linewidth=1.,
                elinewidth=1,
            )
            ax[0].errorbar(
                lb+4,
                bestfit,
                errorbar,
                color="tab:blue",
                label='bestfit',
                ls="-",
                alpha=1.,
                marker='.',
                mfc='white',
                mec='tab:blue',
                linewidth=.5,
                elinewidth=1,
            )
            ax[0].fill_between(
                lb+4,
                bestfit_lower,
                bestfit_upper,
                color="tab:blue",
                # label='bestfit',
                ls="",
                alpha=.3,
            )
            
            ax[1].errorbar(lb, (data - model) / errorbar,
                        yerr=lb * 0. + 1.,
                        ls="None", marker = ".",
                        linewidth=.5,
                        alpha=.5,
                        color="tab:grey",
                        label=f"data (PTE = ${{{pte_uncal:.4f}}}$)")
            ax[1].errorbar(lb, (bestfit - model) / errorbar,
                        yerr=lb * 0. + 1.,
                        ls="None", marker = ".",
                        linewidth=.5,
                        alpha=.5,
                        color="tab:blue",
                        label=f"bestfit (PTE = ${{{pte_cal:.4f}}}$)")
            ax[1].axhline(0, color='black', zorder=-10)

            ax[1].axvspan(xmin=0, xmax=lmin,
                        color="gray", alpha=0.7, zorder=-20)
            ax[1].axvspan(xmin=lmax, xmax=10000,
                        color="gray", alpha=0.7, zorder=-20)

            ax[0].legend(title=f'Poleff={pol_eff_mean[sv_ar, spectrum]:.3f}+-{pol_eff_std[sv_ar, spectrum]:.3f}   Ampdust={dust_mean[sv_ar, spectrum]:.3f}')
            ax[0].set_xlim(0, lmax + 500)
            ax[0].set_ylim(ylims[spectrum])
            ax[0].set_title(f'{sv_ar}')

            ax[1].legend(loc='lower right')
            ax[1].set_ylim(-8, 7)
            ax[1].set_ylabel(fr"$\Delta D_\ell^\mathrm{{{spectrum}}} / \sigma(\Delta D_\ell^\mathrm{{{spectrum}}})$", fontsize=16)
            ax[1].set_xlabel(r"$\ell$", fontsize=20)

            plt.savefig(f"{plots_dir}/poleff_full_{sv_ar}_{spectrum}.png")
            plt.close()

# Save results to a yaml file
poleffs_to_save = {
    'bestfits' : {
        sv_ar: {
            mode: float(pol_eff_mean[sv_ar, mode])
            for mode in ["EE", "TE"]
        }
        for sv_ar in sets_to_measure
    },
    'std' : {
        sv_ar: {
            mode: float(pol_eff_std[sv_ar, mode])
            for mode in ["EE", "TE"]
        }
        for sv_ar in sets_to_measure
    },
}

file = open(f"{d['calib_dir']}/poleffs_dict.yaml", "w")
yaml.dump(poleffs_to_save, file)
file.close()


# Plot and print results
color_list = ["blue", "red", "green"]
fig, ax = plt.subplots(figsize=(15, 8))
for i, sv_ar in enumerate(sets_to_measure):
    ax.axhline(1, color='grey', ls='--')

    for j, mode in enumerate(modes):
        print(f"**************")
        p_eff, std = pol_eff_mean[sv_ar, mode], pol_eff_std[sv_ar, mode]
        print(f"{sv_ar} {mode} {p_eff} {std}")
        cal, std = poleffs_to_save["bestfits"][sv_ar][mode], poleffs_to_save["std"][sv_ar][mode]
        print(f"{mode}, cal: {cal:.5f}, sigma cal: {std:.5f}")

        ax.errorbar(
            i + 0.95 + j * 0.1,
            cal,
            std,
            label=mode,
            color=color_list[j],
            marker=".",
            ls="None",
            markersize=6.5,
            markeredgewidth=2,
        )
    print(f"**************")

    if i == 0:
        ax.legend(fontsize=15)

x = np.arange(1, len(sets_to_measure) + 1)
ax.set_xticks(x, sets_to_measure)
ax.set_ylabel('Polarization efficiency', fontsize=18)
# plt.ylim(0.967, 1.06)
plt.tight_layout()
plt.savefig(f"{plots_dir}/poleff_summary.pdf", bbox_inches="tight")
plt.clf()
plt.close()
