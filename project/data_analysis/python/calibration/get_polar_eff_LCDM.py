from pspy import so_dict, pspy_utils, so_spectra, so_cov
from getdist.mcsamples import loadMCSamples
from pspipe_utils import best_fits
import matplotlib.pyplot as plt
import getdist.plots as gdplt
from cobaya.run import run
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("dictfile", type=str)
parser.add_argument("-p", "--ps_filename", type=str, required=True)
parser.add_argument("-s", "--spectrum", type=str, required=True)
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.dictfile)

ps_filename = args.ps_filename
spectrum = args.spectrum

# Set up directories
ps_dir = "spectra"
cov_dir = "covariances"
mcm_dir = "mcms"
output_dir = "pol_eff_results"
pspy_utils.create_directory(output_dir)

# Load paramfiles info
surveys = d["surveys"]
arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
lmax = 4000#d["lmax"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

# Load ps theory file
l_th, ps_th = pspy_utils.ps_lensed_theory_to_dict(ps_filename, "Dl", lmax=lmax)
l_th = l_th.astype(int)
so_spectra.write_ps(f"{output_dir}/ps_theory_to_calibrate_{spectrum}.dat", l_th, ps_th, type="Dl", spectra=spectra)

# Load foreground dict (with polarized dust normalized to 1)
do_bandpass_integration = d["do_bandpass_integration"]
fg_components = d["fg_components"]
fg_params = d["fg_params"]

fg_components["ee"] = ["dust"]
fg_components["te"] = ["dust"]

fg_params["a_gee"] = 1.
fg_params["a_gte"] = 1.

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
dust_priors = {
    "EE": {"loc": 0.205, "scale": 0.008},
    #"EE": {"loc": 0.271, "scale": 0.012},
    "TE": {"loc": 0.541, "scale": 0.015}
}

# Calibration range
lmin_cal = 1000
lmax_cal = 1500

# Define useful functions
def get_model(cmb_th, fg_th, Bbl, dust_amp, pol_eff, mode):
    ps_theory = (cmb_th + dust_amp * fg_th) * pol_eff ** mode.count("E")
    return Bbl @ ps_theory

for sv in surveys:
    for ar in arrays[sv]:
        
        # Load ps and cov
        spec_name = f"{sv}_{ar}x{sv}_{ar}"
        lb, ps = so_spectra.read_ps(f"{ps_dir}/Dl_{spec_name}_cross.dat", spectra=spectra)
        cov = np.load(f"{cov_dir}/analytic_cov_{spec_name}_{spec_name}.npy")

        # Select the spectrum
        ps = ps[spectrum]
        n_bins = len(lb)
        cov = so_cov.selectblock(cov, spectra, n_bins=n_bins, block=spectrum+spectrum)
        
        # Multipole cuts
        id = np.where((lb >= lmin_cal) & (lb <= lmax_cal))[0]
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
                    "Rminus1_stop": 0.001,
                    "Rminus1_cl_stop": 0.03
                }
            },
            "output": f"{output_dir}/chain_{spectrum}_{sv}_{ar}",
            "force": True,
        }

        updated_info, sampler = run(info)
        
        samples = loadMCSamples(f"{output_dir}/chain_{spectrum}_{sv}_{ar}", settings={"ignore_rows": 0.5})
        gdplot = gdplt.get_subplot_plotter()
        gdplot.triangle_plot(samples, ["pol_eff", "dust_amp"], filled=True, title_limit=1)
        plt.savefig(f"{output_dir}/posterior_dist_{spectrum}_{sv}_{ar}.png", dpi=300, bbox_inches="tight")



        