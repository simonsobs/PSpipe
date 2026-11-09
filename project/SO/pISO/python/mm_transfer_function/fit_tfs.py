
import matplotlib
from pspy import so_spectra, pspy_utils, so_cov, so_map, so_window, so_dict
from math import pi
import numpy as np
import healpy as hp
# import pylab as plt
from matplotlib import pyplot as plt
import os
import pickle
import getdist.plots as gdplt
import itertools
import scipy.stats as ss
import argparse
from pspipe_utils import transfer_function as tf_tools
from pspipe_utils import consistency, log
from cobaya.run import run
from getdist.mcsamples import loadMCSamples
from getdist import MCSamples
import time
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('paramfile', help='Paramfile to compute TF from')
parser.add_argument('--surveys', nargs="+", help='Surveys to compute TF')
parser.add_argument('--models', nargs="+", help='TF Fit models', default=["logistic"])
parser.add_argument('--ref-array', help='Reference array for TF. If reference is R, will compute all TFs AxR/RxR for A all arrays in --surveys', default="planck_f143")
parser.add_argument('--lmin', default=100, type=float)
parser.add_argument('--lmax', default=1200, type=float)
parser.add_argument('--dontfit', type=bool, default=False)
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)
log.info(args.dontfit)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

spec_dir = d["spec_dir"]
best_fits_dir = d["best_fits_dir"]
cov_dir = d["cov_dir"]
tf_dir = d["tf_dir"]
pspy_utils.create_directory(tf_dir)
plots_dir = d["plots_dir"] + "/TF/"
pspy_utils.create_directory(plots_dir)
chains_dir = d["tf_dir"] + "/chains/"
pspy_utils.create_directory(chains_dir)

# clfile = '../spectra/cmb.dat'
# l, ps_theory = so_spectra.read_ps(clfile, spectra=spectra)

f = 'TT'

spec_template = spec_dir + "/Dl_{}x{}_cross.dat"
bestfit_template = best_fits_dir + "/fg_{}x{}.dat"
cov_template = cov_dir + "/analytic_cov_{}x{}_{}x{}.npy"

ref_array = args.ref_array
tf_arrays = [f"{sv}_{ar}" for sv in args.surveys for ar in d[f"arrays_{sv}"]]

# Load spectra and covs for consitency.compute_ps_and_cov_ratio
log.info("Load spectra and covs")
t0 = time.time()
lb, Dls_ref = so_spectra.read_ps(spec_template.format(ref_array, ref_array), spectra=spectra)
_, Dls_fg_ref = pspy_utils.naive_binning(
    so_spectra.read_ps(bestfit_template.format(ref_array, ref_array), spectra=spectra)[0],
    so_spectra.read_ps(bestfit_template.format(ref_array, ref_array), spectra=spectra)[1][f],
    d["binning_file"],
    d["lmax"]
)
lmin, lmax = args.lmin, args.lmax
ell_mask = (lmin < lb) & (lb < lmax)


Dls_tfs = {
    (ref_array, ar): 
        so_spectra.read_ps(spec_template.format(ref_array, ar), spectra=spectra)[1][f][ell_mask]
        - pspy_utils.naive_binning(
            so_spectra.read_ps(bestfit_template.format(ref_array, ar), spectra=spectra)[0],
            so_spectra.read_ps(bestfit_template.format(ref_array, ar), spectra=spectra)[1][f],
            d["binning_file"],
            d["lmax"]
        )[1][ell_mask]
    for ar in tf_arrays
}
ps_dict = {(ref_array, ref_array): Dls_ref[f][ell_mask] - Dls_fg_ref[ell_mask]} | Dls_tfs

cov_ref = np.load(cov_template.format(ref_array, ref_array, ref_array, ref_array))[np.ix_(ell_mask, ell_mask)]
cov_tfs_RRRA = {
    ((ref_array, ref_array), (ref_array, ar)): np.load(cov_template.format(ref_array, ref_array, ref_array, ar))[np.ix_(ell_mask, ell_mask)]
    for ar in tf_arrays
}
cov_tfs_RARA = {
    ((ref_array, ar), (ref_array, ar)): np.load(cov_template.format(ref_array, ar, ref_array, ar))[np.ix_(ell_mask, ell_mask)]
    for ar in tf_arrays
}
cov_dict = {((ref_array, ref_array), (ref_array, ref_array)): cov_ref} | cov_tfs_RRRA | cov_tfs_RARA
log.info(f"Loaded spectra in {(time.time() - t0):.3f}s")

# Compute ratio and ratio cov
log.info("Compute ratios")
t0 = time.time()
ratio_dict = {}
ratio_cov_dict = {}
snr_cut_dict = {}
for ar in tf_arrays:
    ratio_dict[ar], ratio_cov_dict[ar], snr_cut_dict[ar] = consistency.compute_ps_and_cov_ratio(ps_dict, cov_dict, ((ref_array, ar), (ref_array, ref_array)), snr_threshold=2.5)
log.info(f"Computed ratios in {(time.time() - t0):.3f}s")

fixed_amp_dict = {
    "logistic":True,
    "logistic2":True,
    "soft":True,
    "lorentz":False,
}

prior_dict = tf_tools.prior_dict
if not args.dontfit:
    t0 = time.time()
    for ar in tf_arrays:

        for i, method in enumerate(args.models):
            tf_tools.fit_tf(
                lb[ell_mask],
                ratio_dict[ar],
                ratio_cov_dict[ar],
                prior_dict,
                chain_name=f"{chains_dir}/{method}_{ar}",
                fixed_amp=fixed_amp_dict[method],
                method=method,
                no_mpi=True
            )
    log.info(f"TF fitting took {(time.time() - t0):.3f}s")
else: 
    log.info("Don't fit, use existing chains instead")

bf_dict = {}
samples: dict[tuple[str], MCSamples] = {}
for ar in tf_arrays:
    bf_dict[ar] = {}
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.axhline(1, color="grey", ls='--', lw=1)
    ax.errorbar(lb[ell_mask], ratio_dict[ar], np.sqrt(ratio_cov_dict[ar].diagonal()), color="black", ls='', marker='.')
    ax.set_xlabel(r"$\ell$", fontsize=18)
    ax.set_ylabel(fr"$D_\ell^{{Pl\ x\ {ar}GHz}} / D_\ell^{{Pl\ x\ Pl}}$", fontsize=18)
    for i, method in enumerate(args.models):
        tf_bf = tf_tools.get_tf_bestfit(lb[ell_mask], chain_name=f"{chains_dir}/{method}_{ar}", method=method, fixed_amp=fixed_amp_dict[method])
        chi2 = (ratio_dict[ar] - tf_bf) @ np.linalg.inv(ratio_cov_dict[ar]) @ (ratio_dict[ar] - tf_bf)
        pte = 1 - ss.chi2.cdf(chi2, len(lb[ell_mask]) - 3)
        
        ls_plot = np.arange(0, max(lb[ell_mask])+100)
        tf_bf = tf_tools.get_tf_bestfit(ls_plot, chain_name=f"{chains_dir}/{method}_{ar}", method=method, fixed_amp=fixed_amp_dict[method])
        
        ax.plot(ls_plot, tf_bf, label=f"{method} {pte=:.4f}", c=f"C{i+1}")
        
        mu, _ = tf_tools.get_parameter_mean_and_std(chain_name=f"{chains_dir}/{method}_{ar}", pars=["bb", "cc"])
        samples[method, ar] = loadMCSamples(f"{chains_dir}/{method}_{ar}", settings = {"ignore_rows": 0.5})
        bf_dict[ar][method] = [float(param) for param in mu]
        
        ls_save = np.arange(2, 10000)
        tf_save = tf_tools.get_tf_bestfit(ls_save, chain_name=f"{chains_dir}/{method}_{ar}", method=method, fixed_amp=fixed_amp_dict[method])
        np.savetxt(tf_dir + f"/tf_fit_{ar}_logistic.dat", np.array([ls_save, tf_save]).T)
        
        # Add the area between 1 and the TF as a derived parameter
        samples_params = samples[method, ar].getParams()
        ls_compute_area = np.linspace(0, 2000, 4000)
        samples[method, ar].addDerived([np.sum(1 - tf_tools.tf_model(ls_plot, 1, bb, cc, method=method)) for bb, cc in zip(samples_params.bb, samples_params.cc)], name="tf_area", label=r'TF\ area')
    ax.legend()
    plt.savefig(f"{plots_dir}/{ar}_tf.png")
    plt.close()
    
    # for i, method in enumerate(args.models):
    #     params = ["bb", "cc"] if fixed_amp_dict[method] else ["aa", "bb", "cc"]
    #     params = ["bb", "cc", "tf_area"]
    #     gdplot = gdplt.get_subplot_plotter(width_inch=5)
    #     gdplot.triangle_plot(
    #         [samples[method, ar]],
    #         params=params,
    #         title_limit=1,
    #         dpi=200,
    #     )
    #     plt.savefig(f'{plots_dir}/{method}_{ar}.png')
    #     plt.clf()
    #     plt.close()

for i, method in enumerate(args.models):
    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    params = ["bb", "cc"] if fixed_amp_dict[method] else ["aa", "bb", "cc"]
    params = ["bb", "cc", "tf_area"]

    gdplot.triangle_plot(
        [samples[method, ar] for ar in tf_arrays],
        contour_colors=[f"C{c}" for c, _ in enumerate(tf_arrays)],
        contour_ls=['-' for c, _ in enumerate(tf_arrays)],
        contour_lws=[1.5 for c, _ in enumerate(tf_arrays)],
        legend_labels=[label.replace('lat_iso_', '') for c, label in enumerate(tf_arrays)],
        contour_args=[{"alpha": .7} for c, label in enumerate(tf_arrays)],
        params=params,
    )
    plt.savefig(f'{plots_dir}/{method}_all.png')
    plt.clf()
    plt.close()
    
for i, method in enumerate(args.models):
    fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
    
    ax.axhline(1, color="grey", ls='--', lw=1)
    ax.set_xlabel(r"$\ell$", fontsize=18)
    ax.set_ylabel(fr"$D_\ell^{{Pl\ x\ LAT }} / D_\ell^{{Pl\ x\ Pl}}$", fontsize=18)
    for a, ar in enumerate(tf_arrays):
        lss = ['-', '--', '-.']

        ls_plot = np.arange(0, max(lb[ell_mask])+100)
        tf_bf = tf_tools.get_tf_bestfit(ls_plot, chain_name=f"{chains_dir}/{method}_{ar}", method=method, fixed_amp=fixed_amp_dict[method])
        
        ax.plot(ls_plot, tf_bf, label=f"{ar.replace('lat_iso_', '')}", ls=lss[a//10])
    
    ax.legend()
    plt.savefig(f"{plots_dir}/{method}_tfs.png")

        
file = open(f"{tf_dir}/TF_best_fits.yaml", "w")
yaml.dump(bf_dict, file)
file.close()
