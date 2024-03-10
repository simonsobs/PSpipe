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

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# Set up directories
ps_dir = "spectra_leak_corr"
planck_corr = False

if planck_corr:
    ps_dir = "spectra_leak_corr_planck_bias_corr"



cov_dir = "covariances"
bestfit_dir = "best_fits"
mcm_dir = "mcms"
output_dir = "pol_eff_results"
pspy_utils.create_directory(output_dir)

ps_filename = f"{bestfit_dir}/cmb.dat"

# Load paramfiles info
surveys = d["surveys"]
arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
lmax = d["lmax"]

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

print(dust_priors)

passbands = {}
for sv in surveys:
    for ar in arrays[sv]:
        freq_info = d[f"freq_info_{sv}_{ar}"]
        if do_bandpass_integration:
            nu_ghz, pb = np.loadtxt(freq_info["passband"]).T
        else:
            nu_ghz, pb = np.array([freq_info["freq_tag"]]), np.array([1.])

        passbands[f"{sv}_{ar}"] = [nu_ghz, pb]

fg_dict = best_fits.get_foreground_dict(l_th, passbands, fg_components, fg_params, d["fg_norm"])

# Load priors on dust amplitudes
# from Planck 353 GHz spectra
# computed in ACT DR6 windows


# Calibration range
lmin_cal = 500
lmax_cal = {}
lmax_cal["dr6_pa4_f220"] = 2000
lmax_cal["dr6_pa5_f090"] = 1200
lmax_cal["dr6_pa5_f150"] = 2000
lmax_cal["dr6_pa6_f090"] = 1200
lmax_cal["dr6_pa6_f150"] = 2000

lmax_cal["Planck_f100"] = 1200
lmax_cal["Planck_f143"] = 2000
lmax_cal["Planck_f217"] = 2000




arrays_dr6 = [ "pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]
arrays_Planck = ["f100", "f143", "f217"]



# Define useful functions
def get_model(cmb_th, fg_th, Bbl, dust_amp, pol_eff, mode):
    ps_theory = (cmb_th + dust_amp * fg_th) * pol_eff ** mode.count("E")
    return Bbl @ ps_theory

pol_eff_mean, pol_eff_std, dust_mean, dust_std = {}, {}, {}, {}
for sv in surveys:
    for ar in arrays[sv]:
        for spectrum in ["EE", "TE"]:
        
            print(spectrum, sv, ar, lmin_cal, lmax_cal[f"{sv}_{ar}"])

            # Load ps and cov
            spec_name = f"{sv}_{ar}x{sv}_{ar}"
            lb, ps = so_spectra.read_ps(f"{ps_dir}/Dl_{spec_name}_cross.dat", spectra=spectra)
            cov = np.load(f"{cov_dir}/analytic_cov_{spec_name}_{spec_name}.npy")
            leakage_cov = np.load(f"{cov_dir}/leakage_cov_{spec_name}_{spec_name}.npy")

            cov += leakage_cov
            # Select the spectrum
            ps = ps[spectrum]
            n_bins = len(lb)
            
            cov = so_cov.selectblock(cov, spectra, n_bins=n_bins, block=spectrum+spectrum)

            # Multipole cuts
            id = np.where((lb >= lmin_cal) & (lb <= lmax_cal[f"{sv}_{ar}"]))[0]
            ps = ps[id]
            cov = cov[np.ix_(id, id)]
            invcov = np.linalg.inv(cov)

            # Load Bbl
            spin_pair = "spin2xspin2" if spectrum == "EE" else ("spin0xspin2" if spectrum == "TE" else None)
            if spin_pair is None:
                raise ValueError("spectrum must be set to either 'EE' or 'TE'")
            Bbl = np.load(f"{mcm_dir}/{spec_name}_Bbl_{spin_pair}.npy")
            Bbl = Bbl[:n_bins, :lmax]

            # Get theory
            cmb_th = ps_th[spectrum][:lmax]
            fg_th = fg_dict[spectrum.lower(), "dust", f"{sv}_{ar}", f"{sv}_{ar}"][:lmax]

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
                        "latex": r"\epsilon_\mathrm{pol}^{%s}" % f"{sv}_{ar}".replace("_", "\_")
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
                        "Rminus1_stop": 0.01,
                        "Rminus1_cl_stop": 0.01
                    }
                },
                "output": f"{output_dir}/chain_{spectrum}_{sv}_{ar}",
                "force": True,
            }

            updated_info, sampler = run(info)

            samples = loadMCSamples(f"{output_dir}/chain_{spectrum}_{sv}_{ar}", settings={"ignore_rows": 0.5})
            pol_eff_mean[sv, ar, spectrum] = samples.mean("pol_eff")
            pol_eff_std[sv, ar, spectrum] = np.sqrt(samples.cov(["pol_eff"])[0, 0])
            dust_mean[sv, ar, spectrum] = samples.mean("dust_amp")
            dust_std[sv, ar, spectrum] = np.sqrt(samples.cov(["dust_amp"])[0, 0])

            
            gdplot = gdplt.get_subplot_plotter()
            gdplot.triangle_plot(samples, ["pol_eff", "dust_amp"], filled=True, title_limit=1)
            plt.savefig(f"{output_dir}/posterior_dist_{spectrum}_{sv}_{ar}.png", dpi=300, bbox_inches="tight")


for sv in surveys:
    for ar in arrays[sv]:
        print(f"**************")
        for spectrum in ["EE", "TE"]:
            p_eff, std = pol_eff_mean[sv, ar, spectrum], pol_eff_std[sv, ar, spectrum]
            print(f"{sv} {ar} {spectrum} {p_eff} {std}")

        print(f"**************")
