"""
Unlike get_calibs.py, this script measure multiple calibration factors at the same time, using a given set of spectra, including cross spectra.
only does AxA' - BxB for now, #TODO: include others 
#TODO: include foregrounds too
"""

import matplotlib
from cobaya.run import run
matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_spectra, so_cov
from pspipe_utils import consistency, log
import numpy as np
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt
import pylab as plt
from os.path import join as opj
import pickle
import sys
import yaml
from scipy.optimize import curve_fit
import scipy.stats as ss



d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

# log calib infos from calib yaml file
with open(d["calib_cross_yaml"], "r") as f:
    calib_dict: dict = yaml.safe_load(f)
calib_infos: dict = calib_dict["get_calibs_cross.py"]

planck_corr = calib_infos["planck_corr"]
subtract_bf_fg = calib_infos["subtract_bf_fg"]

data_dir = d["data_dir"]
plot_dir = d["plots_dir"]
spec_dir = d["spec_dir"]
cov_dir = d["cov_dir"]
bestfit_dir = d["best_fits_dir"]

# Create output dirs
calib_dir = d["calib_dir"]
residual_output_dir = f"{calib_dir}/residuals"
chains_dir = f"{calib_dir}/chains"
plot_output_dir = plot_dir + "/calib/"
plot_output_spec_dir = plot_dir + "/calib/spec_plots/"

if planck_corr:
    spec_dir = "spectra_leak_corr_planck_bias_corr"
    calib_dir += "_planck_bias_corrected"

_, _, lb, _ = pspy_utils.read_binning_file(d["binning_file"], d["lmax"])
n_bins = len(lb)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if d["cov_T_E_only"] == True:
    modes = ["TT", "TE", "ET", "EE"]
else:
    modes = spectra

pspy_utils.create_directory(calib_dir)
pspy_utils.create_directory(residual_output_dir)
pspy_utils.create_directory(chains_dir)
pspy_utils.create_directory(plot_output_dir)
pspy_utils.create_directory(plot_output_spec_dir)

# Define the projection pattern - i.e. which
# spectra combination will be used to compute
# the residuals with
#   A: the array you want to calibrate
#   B: the reference array


spec_template = spec_dir + "/Dl_{}x{}_cross.dat"
cov_name = "analytic_cov"
cov_template = opj(cov_dir, cov_name + "_{}x{}_{}x{}.npy")

test_names = list(calib_infos["calib_spec_sets"].keys())
results_dict = {}
samples = {}
pte_before_cal = {}
pte_after_cal = {}
cal_vec = {}
for test, calib_spec_set_infos in calib_infos["calib_spec_sets"].items():
    calib_spec_set = [((infos[0][0], infos[0][1]), (infos[1][0], infos[1][1])) for infos in calib_spec_set_infos]
    calib_spec_set_keys = [(infos[0][0], infos[0][1]) for infos in calib_spec_set_infos]
    ell_ranges = [infos[2]for infos in calib_spec_set_infos]
    
    for elem in calib_spec_set:
        if calib_spec_set.count(elem) > 1:
            raise ValueError(f'{elem} appears {calib_spec_set.count(elem)} times !')
    # Starts by reading ell_ranges to determine vector sizes
    ell_masks = []
    len_ls_list = []
    for i, ((spec1, spec2), (ref_spec1, ref_spec2)) in enumerate(calib_spec_set):
        ell_masks.append((lb > ell_ranges[i][0]) & (lb < ell_ranges[i][1]))
        len_ls_list.append(sum(ell_masks[i]))

    len_ls_cumsum = np.concatenate(([0], np.cumsum(len_ls_list)))
    size = np.sum(len_ls_list)
    log.info(f"{test} vec size : {size}, creating empty arrays")
    data_vec = np.zeros(size, dtype=np.float64)
    model_vec = np.zeros(size, dtype=np.float64)
    data_fg_vec = np.zeros(size, dtype=np.float64)
    model_fg_vec = np.zeros(size, dtype=np.float64)
    cov_mat = np.zeros((size, size), dtype=np.float64)
    cov_mat_d = np.zeros((size, size), dtype=np.float64)
    cov_mat_m = np.zeros((size, size), dtype=np.float64)
    cov_mat_dm = np.zeros((size, size), dtype=np.float64)
    cov_mat_md = np.zeros((size, size), dtype=np.float64)   

    # Define data (ex:LAT spectra) and model (ex:planck spectra) from yaml file
    # Loop over all entries of calib_spec_set, then calibrates spec1xspec2 with ref_spec1xref_spec2
    log.info('Loading spectra and covs:')
    for i, ((spec1, spec2), (ref_spec1, ref_spec2)) in enumerate(calib_spec_set):
        # Load data
        ls, Dls = so_spectra.read_ps(spec_template.format(spec1, spec2), spectra=spectra)
        data_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] = Dls['TT'][ell_masks[i]]
        
        # Load model (planck :p)
        ls, Dls = so_spectra.read_ps(spec_template.format(ref_spec1, ref_spec2), spectra=spectra)
        model_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] = Dls['TT'][ell_masks[i]]
        
        # Load cov, including cov with other calib sets
        for j, ((spec3, spec4), (ref_spec3, ref_spec4)) in enumerate(calib_spec_set):
            try:
                cov_mat_d[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(spec1, spec2, spec3, spec4)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
            except:
                cov_mat_d[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(spec3, spec4, spec1, spec2)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
            
            try:
                cov_mat_m[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(ref_spec1, ref_spec2, ref_spec3, ref_spec4)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
            except:
                cov_mat_m[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(ref_spec3, ref_spec4, ref_spec1, ref_spec2)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
            cov_mat_dm[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(ref_spec3, ref_spec4, spec1, spec2)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
            cov_mat_md[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(ref_spec1, ref_spec2, spec3, spec4)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
  
        if subtract_bf_fg:
            
            l_fg, bf_fg = so_spectra.read_ps(f"{bestfit_dir}/fg_{spec1}x{spec2}.dat", spectra=spectra)
            _, bf_fg_TT_binned = pspy_utils.naive_binning(l_fg, bf_fg["TT"], d["binning_file"], d["lmax"])
            data_fg_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] = bf_fg_TT_binned[ell_masks[i]]
            
            l_fg, bf_fg = so_spectra.read_ps(f"{bestfit_dir}/fg_{ref_spec1}x{ref_spec2}.dat", spectra=spectra)
            _, bf_fg_TT_binned = pspy_utils.naive_binning(l_fg, bf_fg["TT"], d["binning_file"], d["lmax"])
            model_fg_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] = bf_fg_TT_binned[ell_masks[i]]

    log.info('Inverting cov mat')
    cov_mat_res = cov_mat_d + cov_mat_m - cov_mat_dm - cov_mat_md.T
    inv_cov_mat_res = np.linalg.inv(cov_mat_res)

    # Make calib vector
    arrays_set = {ar for tpl in calib_spec_set_keys for ar in tpl}
    
    arrays_set = [f'lat_iso_{ar}' for ar in d['arrays_lat_iso'] if f'lat_iso_{ar}' in arrays_set]   # Put arrays in the right order
    arrays_calib_map = {ar: id for id, ar in enumerate(arrays_set)}
    arrays_calib_names = {ar: f"cal{id}" for id, ar in enumerate(arrays_set)}

    fig, ax = plt.subplots(figsize=(size/3, size/3))
    p = ax.imshow(cov_mat_res)
    plt.colorbar(p)
    ax.set_xticks(ticks=(len_ls_cumsum[1:] + len_ls_cumsum[:-1])/2, labels=list(calib_spec_set_keys))
    ax.set_yticks(ticks=(len_ls_cumsum[1:] + len_ls_cumsum[:-1])/2, labels=list(calib_spec_set_keys), rotation='vertical')
    plt.savefig(opj(plot_output_dir, f'invcov_{test}.png'))
    plt.close()

    i = 0
    # TODO: make this more elegant ?
    def get_calib_vec(*cals):
        calib_vector = np.ones(size, dtype=np.float64)
        for i, (spec1, spec2) in enumerate(calib_spec_set_keys):
            calib_vector[len_ls_cumsum[i]:len_ls_cumsum[i+1]] *= cals[arrays_calib_map[spec1]] * cals[arrays_calib_map[spec2]]
        return calib_vector


    def logL(cal0=None, cal1=None, cal2=None, cal3=None, cal4=None, cal5=None, cal6=None, cal7=None, cal8=None, cal9=None, cal10=None, cal11=None, cal12=None):
        all_cals = [cal0, cal1, cal2, cal3, cal4, cal5, cal6, cal7, cal8, cal9, cal10, cal11, cal12]
        notNone_cals = [cal for cal in all_cals if cal is not None]
        calib_vec = get_calib_vec(*notNone_cals)
        res_vec = data_vec - (model_vec / calib_vec) - data_fg_vec + model_fg_vec   # Don't calib fg ?
        chi2 = res_vec @ inv_cov_mat_res @ res_vec
        return -0.5 * chi2
    
    chain_name = f"{chains_dir}/chain_{test}"
    info = {
        "likelihood": {"my_like": logL},
        "params": {
            f"cal{i}": {
                "prior": {
                    "min": 0.5,
                    "max": 1.5
                            },
                "proposal": 1e-2,
                "latex": fr"{name[-7:].replace('_', r'\_')}"
                    }
                for i, name in enumerate(arrays_set)
                },
        "sampler": {
            "mcmc": {
                "max_tries": 1e7,
                "Rminus1_stop": 0.01,
                "Rminus1_cl_stop": 0.2,
                # "learn_proposal_update": 5000,
                "output_every": "20s",
                    }
                    },
        "output": chain_name,
        "force": True,
            }

    updated_info, sampler = run(info)
    samples[test] = loadMCSamples(chain_name, settings = {"ignore_rows": 0.5})

    results_dict[test] =  {ar: [samples[test].mean(name), np.sqrt(samples[test].cov([name])[0, 0])] for ar, name in arrays_calib_names.items()}
    cal_vec[test] = get_calib_vec(*[results_dict[test][ar][0] for ar in arrays_calib_names.keys()])

    pte_before_cal[test] = {}
    pte_after_cal[test] = {}
    for i, ((spec1, spec2), (ref_spec1, ref_spec2)) in enumerate(calib_spec_set):
        
        pte_before_cal[test][(spec1, spec2), (ref_spec1, ref_spec2)] = 1 - ss.chi2(len(ls[ell_masks[i]])).cdf((data_vec - model_vec)[len_ls_cumsum[i]:len_ls_cumsum[i+1]] @ np.linalg.inv(cov_mat_res[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[i]:len_ls_cumsum[i+1]]) @ (data_vec - model_vec)[len_ls_cumsum[i]:len_ls_cumsum[i+1]])
        pte_after_cal[test][(spec1, spec2), (ref_spec1, ref_spec2)] = 1 - ss.chi2(len(ls[ell_masks[i]])).cdf((data_vec - model_vec / cal_vec[test])[len_ls_cumsum[i]:len_ls_cumsum[i+1]] @ np.linalg.inv(cov_mat_res[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[i]:len_ls_cumsum[i+1]]) @ (data_vec - model_vec / cal_vec[test])[len_ls_cumsum[i]:len_ls_cumsum[i+1]])
        

#plit plot everyting
log.info("Plot calibrated spectra")
for test, calib_spec_set_infos in calib_infos["calib_spec_sets"].items():

    ell_masks = []
    len_ls_list = []
    ell_range_plot = [0, 2500]
    for i, ((spec1, spec2), (ref_spec1, ref_spec2)) in enumerate(calib_spec_set):
        ell_masks.append((lb > ell_range_plot[0]) & (lb < ell_range_plot[1]))
        len_ls_list.append(sum(ell_masks[i]))

    len_ls_cumsum = np.concatenate(([0], np.cumsum(len_ls_list)))
    size = np.sum(len_ls_list)
    log.info(f"{test} plot vec size : {size}")
    data_vec = np.zeros(size, dtype=np.float64)
    model_vec = np.zeros(size, dtype=np.float64)
    data_fg_vec = np.zeros(size, dtype=np.float64)
    model_fg_vec = np.zeros(size, dtype=np.float64)
    cov_mat = np.zeros((size, size), dtype=np.float64)
    cov_mat_d = np.zeros((size, size), dtype=np.float64)
    cov_mat_m = np.zeros((size, size), dtype=np.float64)
    cov_mat_dm = np.zeros((size, size), dtype=np.float64)
    cov_mat_md = np.zeros((size, size), dtype=np.float64)   

    # Define data (ex:LAT spectra) and model (ex:planck spectra) from yaml file
    # Loop over all entries of calib_spec_set, then calibrates spec1xspec2 with ref_spec1xref_spec2
    for i, ((spec1, spec2), (ref_spec1, ref_spec2)) in enumerate(calib_spec_set):
        # Load data
        ls, Dls = so_spectra.read_ps(spec_template.format(spec1, spec2), spectra=spectra)
        data_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] = Dls['TT'][ell_masks[i]]
        
        # Load model (planck :p)
        ls, Dls = so_spectra.read_ps(spec_template.format(ref_spec1, ref_spec2), spectra=spectra)
        model_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] = Dls['TT'][ell_masks[i]]
        
        # Load cov, including cov with other calib sets
        for j, ((spec3, spec4), (ref_spec3, ref_spec4)) in enumerate(calib_spec_set):
            try:
                cov_mat_d[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(spec1, spec2, spec3, spec4)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
            except:
                cov_mat_d[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(spec3, spec4, spec1, spec2)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
            
            try:
                cov_mat_m[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(ref_spec1, ref_spec2, ref_spec3, ref_spec4)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
            except:
                cov_mat_m[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(ref_spec3, ref_spec4, ref_spec1, ref_spec2)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
            cov_mat_dm[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(ref_spec3, ref_spec4, spec1, spec2)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
            cov_mat_md[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(ref_spec1, ref_spec2, spec3, spec4)), spectra_order=spectra, n_bins=len(lb), block='TTTT')[np.ix_(ell_masks[i], ell_masks[j])]
  
        if subtract_bf_fg:
            
            l_fg, bf_fg = so_spectra.read_ps(f"{bestfit_dir}/fg_{spec1}x{spec2}.dat", spectra=spectra)
            _, bf_fg_TT_binned = pspy_utils.naive_binning(l_fg, bf_fg["TT"], d["binning_file"], d["lmax"])
            data_fg_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] = bf_fg_TT_binned[ell_masks[i]]
            
            l_fg, bf_fg = so_spectra.read_ps(f"{bestfit_dir}/fg_{ref_spec1}x{ref_spec2}.dat", spectra=spectra)
            _, bf_fg_TT_binned = pspy_utils.naive_binning(l_fg, bf_fg["TT"], d["binning_file"], d["lmax"])
            model_fg_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] = bf_fg_TT_binned[ell_masks[i]]

    cov_mat_res = cov_mat_d + cov_mat_m - cov_mat_dm - cov_mat_md.T
    
    def get_calib_vec(*cals):
        calib_vector = np.ones(size, dtype=np.float64)
        for i, (spec1, spec2) in enumerate(calib_spec_set_keys):
            calib_vector[len_ls_cumsum[i]:len_ls_cumsum[i+1]] *= cals[arrays_calib_map[spec1]] * cals[arrays_calib_map[spec2]]
        return calib_vector
    cal_vec[test] = get_calib_vec(*[results_dict[test][ar][0] for ar in arrays_calib_names.keys()])
    
    for i, ((spec1, spec2), (ref_spec1, ref_spec2)) in enumerate(calib_spec_set):
        fig, ax = plt.subplots(2, figsize=(8, 6), gridspec_kw={"hspace": 0, "height_ratios": (2.5, 1.3)}, sharex=True)
       
        ax[1].axvspan(xmin=0, xmax=ell_ranges[i][0], color="gray", alpha=0.5, zorder=-20)
        ax[1].axvspan(xmin=ell_ranges[i][1], xmax=10000, color="gray", alpha=0.5, zorder=-20)

        ax[1].axhline(0., color='black')
        ax[0].errorbar(ls[ell_masks[i]], data_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]], np.sqrt(cov_mat_d[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[i]:len_ls_cumsum[i+1]].diagonal()), label=(spec1, spec2), color='red', lw=.8, capsize=1.5, marker='.')
        ax[0].errorbar(ls[ell_masks[i]], cal_vec[test][len_ls_cumsum[i]:len_ls_cumsum[i+1]] * data_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]], np.sqrt(cov_mat_d[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[i]:len_ls_cumsum[i+1]].diagonal()), label=f'{(spec1, spec2)} cal', color='blue', lw=.8, capsize=1.5, marker='.')
        ax[0].errorbar(ls[ell_masks[i]], model_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]], np.sqrt(cov_mat_m[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[i]:len_ls_cumsum[i+1]].diagonal()), label=(ref_spec1, ref_spec2), color='black', lw=.8, capsize=1.5, marker='.')
        ax[0].plot(ls[ell_masks[i]], data_fg_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]], label=f'{(spec1, spec2)} fg', color='blue')
        ax[0].plot(ls[ell_masks[i]], model_fg_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]], label=f'{(ref_spec1, ref_spec2)} fg', color='black')
        ax[1].errorbar(
            ls[ell_masks[i]]+1, 
            (data_vec - model_vec)[len_ls_cumsum[i]:len_ls_cumsum[i+1]] / np.sqrt(cov_mat_res[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[i]:len_ls_cumsum[i+1]].diagonal()),
            np.ones_like(ls[ell_masks[i]]),
            label=f'Before cal PTE={pte_before_cal[test][(spec1, spec2), (ref_spec1, ref_spec2)]:.4f}',
            color='red',
            ls='',
            marker='.'
        )
        ax[1].errorbar(
            ls[ell_masks[i]]-1, 
            (data_vec - model_vec / cal_vec[test])[len_ls_cumsum[i]:len_ls_cumsum[i+1]] / np.sqrt(cov_mat_res[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[i]:len_ls_cumsum[i+1]].diagonal()),
            np.ones_like(ls[ell_masks[i]]),
            label=f'After cal PTE={pte_after_cal[test][(spec1, spec2), (ref_spec1, ref_spec2)]:.4f}',
            color='blue',
            ls='',
            marker='.'
        )
        ax[0].set_xlim(*ell_range_plot)
        ax[0].set_ylim(4e1, 9e3)
        ax[0].set_yscale('log')
        ax[0].legend()
        ax[1].legend()
        plt.savefig(opj(plot_output_spec_dir, f'{spec1}x{spec2}-{ref_spec1}x{ref_spec2}.png'))
        plt.close()

color_list = ["blue", "red", "green", "orange", "purple", "darkcyan"]

gdplot = gdplt.get_subplot_plotter(width_inch=5)
gdplot.triangle_plot(
    [samples[test] for test in calib_infos["calib_spec_sets"].keys()],
    info['params'],
    title_limit=1,
    legend_labels=[test for test in calib_infos["calib_spec_sets"].keys()],
    colors=color_list[:len(test_names)],
)
plt.savefig(f'{plot_output_dir}/triangle_plot.png')
plt.clf()

# plot the cal factors
fig, ax = plt.subplots(figsize=(12, 6))
ax.axhline(1, color='grey', ls='--')


x_shift = np.linspace(-.025*len(test_names), .025*len(test_names), len(test_names))
for j, (test, results_subdict) in enumerate(results_dict.items()):
    for i, (name, (cal, std)) in enumerate(results_subdict.items()):
        print(f"**************")
        print(f"calibration factor of {name} with {test}: {cal:.4f}±{std:.4f}")
        ax.errorbar(
            i + x_shift[j],
            cal,
            std,
            color=color_list[j],
            marker=".",
            ls="None",
            label=test if i == 0 else None,
            markersize=6.5,
            markeredgewidth=2,
        )

ax.legend(fontsize=15)

x = np.arange(0, len(arrays_set))
ax.set_xticks(x, [ar[-7:] for ar in arrays_set])
ax.set_ylabel('Calibration factor', fontsize=18)
plt.tight_layout()
plt.savefig(f"{plot_output_dir}/calibs_summary{'_'.join(test_names)}.pdf", bbox_inches="tight")
plt.clf()
plt.close()


for j, (test, results_subdict) in enumerate(results_dict.items()):
    calibs_to_save = {
        name: float(cal)
        for name, (cal, std) in results_subdict.items()
    }

    file = open(f"{calib_dir}/calibs_dict_{test}.yaml", "w")
    yaml.dump(calibs_to_save, file)
    file.close()
    
    with open(f"{calib_dir}/calibs_dict_{test}.pickle", 'wb') as handle:
        pickle.dump(calibs_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    