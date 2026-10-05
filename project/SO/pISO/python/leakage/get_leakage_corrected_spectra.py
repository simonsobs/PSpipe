"""
This script correct the power spectra from  T->P leakage
the idea is to subtract from each data spectra the expected contribution from the leakage
computed from the planet beam leakage measurement and a best fit model.
Note that this assume we have a best fit model, but realistically, not knowing the best fit
presicely is going to be a second order correction
# TODO: unify this script and get_leakage_corrected_spectra_per_split that Serena wrote
"""

import matplotlib
import argparse
matplotlib.use("Agg")

import sys

import numpy as np
from pspipe_utils import leakage, pspipe_list, log
from pspy import  so_dict, so_spectra, pspy_utils
import pylab as plt

parser = argparse.ArgumentParser()
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--use-sim', default=False,
                    help='Use simulated spectra or not')
parser.add_argument('--sim-name', default="",
                    help='')
parser.add_argument('--sim-start', default=0, type=int,
                    help='')
parser.add_argument('--sim-stop', default=0, type=int,
                    help='')
parser.add_argument('--plot', default=False,
                    help='')

args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
binning_file = d["binning_file"]
type = d["type"]

bestfit_dir = d["best_fits_dir"]
spec_dir = d["spec_dir"] if not args.use_sim else d["sim_spec_dir"]
spec_corr_dir = spec_dir + "/leakage_corr/"
plot_dir = d["plots_dir"] + "/leakage/"

pspy_utils.create_directory(spec_corr_dir)
pspy_utils.create_directory(plot_dir)

# read the leakage model
gamma, var = {}, {}

for sv in surveys:

    arrays = d[f"arrays_{sv}"]
for sv in surveys:
    for ar in arrays:
        name = f"{sv}_{ar}"
        pol_eff = d[f"pol_eff_{name}"]

        gamma[name], var[name] = {}, {}
        try:
            l, gamma[name]["TE"], err_m_TE, gamma[name]["TB"], err_m_TB = leakage.read_leakage_model(d[f"leakage_beam_{name}_TE"],
                                                                                                    d[f"leakage_beam_{name}_TB"],
                                                                                                    lmax,
                                                                                                    lmin=2,
                                                                                                    pol_eff=pol_eff)
        except:
            log.info(f"{name} leakage not found, using a dummy one.")
            l = np.arange(2, lmax)
            gamma[name]["TE"] = np.zeros(len(l))
            err_m_TE = np.zeros((len(l), len(l)))
            gamma[name]["TB"] = np.zeros(len(l))
            err_m_TB = np.zeros((len(l), len(l)))

        var[name]["TETE"] = leakage.error_modes_to_cov(err_m_TE).diagonal()
        var[name]["TBTB"] = leakage.error_modes_to_cov(err_m_TB).diagonal()
        var[name]["TETB"] = var[name]["TETE"] * 0

if args.plot:
    plt.figure(figsize=(12, 8))
    for sv in surveys:
        for ar in arrays:
            plt.subplot(2,1,1)
            plt.errorbar(l, gamma[name]["TE"], np.sqrt(var[name]["TETE"]), fmt=".", label=name)
            plt.ylabel(r"$\gamma^{TE}_{\ell}$", fontsize=17)
            plt.legend()
            plt.subplot(2,1,2)
            plt.ylabel(r"$\gamma^{TB}_{\ell}$", fontsize=17)
            plt.xlabel(r"$\ell$", fontsize=17)
            plt.errorbar(l, gamma[name]["TB"], np.sqrt(var[name]["TBTB"]), fmt=".", label=name)
            plt.legend()
    plt.savefig(f"{plot_dir}/beam_leakage.png", bbox_inches="tight")
    plt.clf()
    plt.close()

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

for spec_name in spec_name_list:

    name1, name2 = spec_name.split("x")

    l_th, ps_th = so_spectra.read_ps(f"{bestfit_dir}/cmb_and_fg_{spec_name}.dat", spectra=spectra)
    
    id = np.where(l_th < lmax)
    l_th = l_th[id]
    for spec in spectra:
        ps_th[spec] = ps_th[spec][id]

    lb, residual = leakage.leakage_correction(l_th,
                                            ps_th,
                                            gamma[name1],
                                            var[name1],
                                            lmax,
                                            return_residual=True,
                                            gamma_beta=gamma[name2],
                                            binning_file=binning_file)
    
    if not args.use_sim:
        tags = [""]
    else:
        tags = [f"{args.sim_name}_{iii:05d}" for iii in np.arange(args.sim_start, args.sim_stop, dtype=int)]
    
    for tag in tags:
        log.info(f"correcting spectra {tag} {spec_name}")

        lb, ps = so_spectra.read_ps(spec_dir + f"/{type}{tag}_{spec_name}_cross.dat", spectra=spectra)
        
        ps_corr = {}
        for spec in spectra:
        
            ps_corr[spec] = ps[spec] - residual[spec]

            if args.plot:
                plt.figure(figsize=(12, 8))
                plt.plot(lb, ps[spec], label="pre correction")
                plt.plot(lb, ps_corr[spec], label="post correction")
                plt.legend(fontsize=12)
                plt.savefig(f"{plot_dir}/{spec_name}_{spec}{tag}.png", bbox_inches="tight")
                plt.legend()
                plt.clf()
                plt.close()

        so_spectra.write_ps(spec_corr_dir + f"/{type}{tag}_{spec_name}_cross.dat", lb, ps_corr, type, spectra=spectra)
