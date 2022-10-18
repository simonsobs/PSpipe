"""
This script extract pol efficiency by comparing the EE power spectra to
a theory file, this is only used by lensers
not that for it to works as expected, the spectra should be computed with pol_eff = 1 in the global file
"""

import numpy as np
import pylab as plt
from pspy import so_spectra, so_cov, so_dict, pspy_utils, so_mcm
from pspipe_utils import pspipe_list, best_fits, covariance
import sys
from cobaya.run import run
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt
from scipy.stats import norm


def get_binned_model(Bbl_EE, ps_th_EE, dust_th_ee, a_gee, pol_eff):
    model = (ps_th_EE + a_gee * dust_th_ee) * pol_eff ** 2
    bin_model = np.dot(Bbl_EE, model)
    return bin_model
    
def compute_loglike(a_gee, pol_eff):
    bin_model = get_binned_model(Bbl_EE, ps_th_EE, dust_th_ee, a_gee, pol_eff)
    chi2 = np.sum((ps_EE[id] - bin_model[id]) ** 2 / err_EE[id] ** 2)
    return -0.5 * chi2
    
def compute_loglike_cov(a_gee, pol_eff):
    bin_model = get_binned_model(Bbl_EE, ps_th_EE, dust_th_ee, a_gee, pol_eff)
    diff = ps_EE[id] - bin_model[id]
    chi2 = diff @ inv_cov @ diff
    return -0.5 * chi2

def fit_pol_eff(ps_EE, err_EE, Bbl_EE, l_th, ps_th_EE, dust_th_ee, nu, chain_name, pol_eff_name, inv_cov=None):


    info = {
        "likelihood": {"my_like": compute_loglike},
        "params": {
            "pol_eff": {"prior": {"min": 0.5, "max": 1.5}, "latex": pol_eff_name},
            "a_gee": {"prior": {"dist": "norm", "loc": 0.271, "scale": 0.012}, "latex": "ag_{ee}"},
        },
        "sampler": {
            "mcmc": {
                "max_tries": 10 ** 8,
                "Rminus1_stop": 0.002,
                "Rminus1_cl_stop": 0.02,
            }
        },
        "output": f"{chain_name}",
        "force": True,
        "debug": False,
    }
    
    if inv_cov is not None:
        info["likelihood"]["my_like"] = compute_loglike_cov

    updated_info, sampler = run(info)
    
def plot_chain(chain_name, params):
    samples = loadMCSamples( f"{chain_name}", settings = {"ignore_rows": 0.5})
    print(samples.getInlineLatex("pol_eff", limit=1))

    gdplot = gdplt.get_subplot_plotter()
    gdplot.triangle_plot(samples, params, filled = True, title_limit=1)
    plt.savefig(f"{chain_name}.png", dpi = 300)
    plt.clf()
    plt.close()

def read_Bbl(prefix, spin_pairs=None):
    Bbl = {}
    for spin in spin_pairs:
        Bbl[spin] = np.load(prefix + "_Bbl_%s.npy" % spin)
    return Bbl

    
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
cosmo_params = d["cosmo_params"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
cov_spec_order = ["TT", "TE", "ET", "EE"]

which="_poleffone"

spec_dir = "spectra" + which
cov_dir = "covariances" + which
mcm_dir = "mcms"
chain_dir = "chains" + which

pspy_utils.create_directory(chain_dir)

fg_norm = d["fg_norm"]
fg_params = d["fg_params"]
fg_components = d["fg_components"]
lmax = d["lmax"]
type = d["type"]

ps_file = "cosmo2017_10K_acc3_lensedCls.dat"
l_th, ps_th = pspy_utils.ps_lensed_theory_to_dict(ps_file, "Dl", lmax=lmax)
l_th = l_th.astype("int")
freq_list = pspipe_list.get_freq_list(d)

lmin_cal = 1000
lmax_cal = 1500
spectrum = "EE"

for sv in surveys:
    arrays = d[f"arrays_{sv}"]

    for ar in arrays:
        if ar == "pa4_f220": continue
        
        spec_name = f"{sv}_{ar}x{sv}_{ar}"
        
        nu = d[f"nu_eff_{sv}_{ar}"]

        l, ps = so_spectra.read_ps(f"{spec_dir}/Dl_{spec_name}_cross.dat", spectra=spectra)
        cov = np.load(f"{cov_dir}/analytic_cov_{spec_name}_{spec_name}.npy")
        
        spec = f"{sv}_{ar}x{sv}_{ar}"
        prefix= f"{mcm_dir}/{spec}"
        Bbl = read_Bbl(prefix=prefix,spin_pairs=spin_pairs)
        
        n_bins = len(l)
        
        # load the EE Bbl, the EE ps and its error
        Bbl_EE = Bbl["spin2xspin2"][:n_bins, :lmax]
        ps_EE = ps["EE"]
        err_EE = so_cov.get_sigma(cov, cov_spec_order, n_bins, "EE")
        
        # load the lensers th EE power spectrum and a template for the dust
        ps_th_EE = ps_th["EE"]
        fg_params["a_gee"] = 1
        fg_dict = best_fits.get_foreground_dict(l_th, freq_list, fg_components, fg_params, fg_norm=fg_norm)
        dust_th_ee = fg_dict["ee", "dust", nu, nu]
                
        id = np.where((l > lmin_cal) & (l < lmax_cal))[0]
        
        sub_cov = so_cov.selectblock(cov, ["TT", "TE", "ET", "EE"], n_bins = n_bins, block = "EEEE")
        sub_cov = sub_cov[np.ix_(id, id)]
        inv_cov = np.linalg.inv(sub_cov)
        
        chain_name = f"{chain_dir}/{spec_name}"
        wafer, freq = ar.split("_")
        pol_eff_name = "p^{%s %s}_{eff}" % (wafer, freq)
        
        fit_pol_eff(ps_EE, err_EE, Bbl_EE, l_th, ps_th_EE, dust_th_ee, nu, chain_name, pol_eff_name, inv_cov=inv_cov)

for sv in surveys:
    arrays = d[f"arrays_{sv}"]
    for ar in arrays:
        if ar == "pa4_f220": continue
        spec_name = f"{sv}_{ar}x{sv}_{ar}"
               
        chain_name = f"{chain_dir}/{spec_name}"

        params = ["pol_eff", "a_gee"]
        plot_chain(chain_name, params)
