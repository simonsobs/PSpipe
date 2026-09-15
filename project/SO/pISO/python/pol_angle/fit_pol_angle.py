"""
This script performs EB and TB null tests and additionaly fit for a polarisation angle
"""

from pspy import so_dict, pspy_utils, so_cov, so_spectra
from pspipe_utils import  covariance, pspipe_list, pol_angle
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import sys
from cobaya.run import run
import getdist.plots as gdplt
from getdist.mcsamples import loadMCSamples, MCSamples
import argparse
import yaml
from os.path import join as opj


parser = argparse.ArgumentParser()
parser.add_argument('paramfile', help='Paramfile ou quoi')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

_, _, lb, _ = pspy_utils.read_binning_file(d["binning_file"], d["lmax"])
n_bins = len(lb)

spec_dir = d["spec_dir"]
cov_dir = d["cov_dir"]
bestfit_dir = d["best_fits_dir"]
pol_angle_dir = d["pol_angle_dir"]
plot_dir = d["plots_dir"]
plot_output_dir = plot_dir + "/pol_angle/"
plot_output_spec_dir = plot_output_dir + "/spec/"
chains_dir = pol_angle_dir + "/chains/"

pspy_utils.create_directory(pol_angle_dir)
pspy_utils.create_directory(plot_output_dir)
pspy_utils.create_directory(plot_output_spec_dir)
pspy_utils.create_directory(chains_dir)


# log angle infos from calib yaml file
with open(d["pol_angle_yaml"], "r") as f:
    pol_angle_dict: dict = yaml.safe_load(f)
pol_angle_infos: dict = pol_angle_dict["fit_pol_angle.py"]
test_names = list(pol_angle_infos["angle_spec_sets"].keys())

spec_template = spec_dir + "/Dl_{}x{}_cross.dat"
best_fit_template = bestfit_dir + "cmb_and_fg_{}x{}.dat"
cov_name = "analytic_cov"
cov_template = opj(cov_dir, cov_name + "_{}x{}_{}x{}.npy")
bbl_template = d["mcm_dir"] + "{}x{}_Bbl.npy"

use_mc_cov = False      # TODO: put these in parser
use_beam_cov = False
use_leakage_cov = False

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
ylim = {}
ylim["EB"] = ylim["BE"] =  [-5, 5]
ylim["TB"] = ylim["BT"] = [-10, 10]

mode = "EB"
do_crossmode = mode != mode[::-1]

samples: dict[str, MCSamples] = {}
results_dict = {}
all_params_set = {}
all_params_labels_set = {}
means_vec = {}
for test, angle_spec_set_infos in pol_angle_infos["angle_spec_sets"].items():
    angle_spec_set = [(infos[0][0], infos[0][1]) for infos in angle_spec_set_infos]
    ell_ranges = [infos[1]for infos in angle_spec_set_infos]
    
    # Check that there are no duplicates in angle_spec_set
    for elem in angle_spec_set:
        if angle_spec_set.count(elem) > 1:
            raise ValueError(f'{elem} appears {angle_spec_set.count(elem)} times !')

    # Starts by reading ell_ranges to determine vector sizes and modes
    ell_masks = []
    len_ls_list = []
    modes = []
    angle_spec_set_full = []
    for i, (spec1, spec2) in enumerate(angle_spec_set):
        ell_masks.append((lb > ell_ranges[i][0]) & (lb < ell_ranges[i][1]))
        len_ls_list.append(sum(ell_masks[i]))
        modes.append(mode)
        angle_spec_set_full.append([spec1, spec2])
        if (spec1 != spec2) and do_crossmode:
            # If spec1 != spec2, add BE
            ell_masks.append((lb > ell_ranges[i][0]) & (lb < ell_ranges[i][1]))
            len_ls_list.append(sum(ell_masks[i]))
            modes.append(mode[::-1])
            angle_spec_set_full.append([spec1, spec2])

    len_ls_cumsum = np.concatenate(([0], np.cumsum(len_ls_list)))
    size = np.sum(len_ls_list)
    print(f"{test} vec size : {size}, creating empty arrays")
    data_vec = np.zeros(size, dtype=np.float64)
    model_vec = np.zeros(size, dtype=np.float64)
    data_fg_vec = np.zeros(size, dtype=np.float64)
    model_fg_vec = np.zeros(size, dtype=np.float64)
    cov_mat = np.zeros((size, size), dtype=np.float64)
    cov_mat_d = np.zeros((size, size), dtype=np.float64)
    cov_mat_m = np.zeros((size, size), dtype=np.float64)
    cov_mat_dm = np.zeros((size, size), dtype=np.float64)
    cov_mat_md = np.zeros((size, size), dtype=np.float64)       

    # Define data (ex:LAT spectra) from yaml file
    # Loop over all entries of angle_spec_set, then calibrates spec1xspec2 EB and BE with its rotated EE - BB best_fit counterpart
    print('Loading spectra and covs:')
    for i, (spec1, spec2) in enumerate(angle_spec_set_full):
        # Load data
        lb, Dlb = so_spectra.read_ps(spec_template.format(spec1, spec2), spectra=spectra)
        data_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] = Dlb[modes[i]][ell_masks[i]]
        
        ls, Dls = so_spectra.read_ps(best_fit_template.format(spec1, spec2), spectra=spectra)
        model_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] = pspy_utils.naive_binning(ls, Dls["EE"] - Dls["BB"], d["binning_file"], d["lmax"])[1][ell_masks[i]]     # TODO: USE BBL

        # Load cov, including cov with other calib sets
        for j, (spec3, spec4) in enumerate(angle_spec_set_full):
            try:
                cov_mat_d[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(spec1, spec2, spec3, spec4)), spectra_order=spectra, n_bins=len(lb), block=modes[i]+modes[j])[np.ix_(ell_masks[i], ell_masks[j])]
            except:
                cov_mat_d[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[j]:len_ls_cumsum[j+1]] = so_cov.selectblock(np.load(cov_template.format(spec3, spec4, spec1, spec2)), spectra_order=spectra, n_bins=len(lb), block=modes[j]+modes[i])[np.ix_(ell_masks[i], ell_masks[j])]

    n_bins = len(lb)
    inv_cov_mat_d = np.linalg.inv(cov_mat_d)

    # Make angle vector
    arrays_set_set = {ar for tpl in angle_spec_set_full for ar in tpl}  # set of a variable that ends with set :)
    arrays_set = [f'{sv}_{ar}' for sv in d["surveys"] for ar in d[f'arrays_{sv}'] if f'{sv}_{ar}' in arrays_set_set]
    arrays_angle_indices = {ar: id for id, ar in enumerate(arrays_set)}
    arrays_angle_names = {ar: f"phi{id}" for id, ar in enumerate(arrays_set)}

    i = 0
    def get_angles_vec(*phis):
        angle_vec = np.zeros(size, dtype=np.float64)
        for i, (spec1, spec2) in enumerate(angle_spec_set_full):
            if modes[i] == "EB":
                angle_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] += np.sin(4 * np.deg2rad(phis[arrays_angle_indices[spec2]])) / 2
            elif modes[i] == "BE":
                angle_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] += np.sin(4 * np.deg2rad(phis[arrays_angle_indices[spec1]])) / 2
            else:
                raise ValueError("modes must contain EB or BE")
        return angle_vec

    # TODO: make this more elegant ?
    def logL(phi0=None, phi1=None, phi2=None, phi3=None, phi4=None, phi5=None, phi6=None, phi7=None, phi8=None, phi9=None, phi10=None, phi11=None, phi12=None):
        all_phis = [phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9, phi10, phi11, phi12]
        notNone_phis = [phi for phi in all_phis if phi is not None]
        phi_vec = get_angles_vec(*notNone_phis)
        res_vec = data_vec - phi_vec * model_vec
        chi2 = res_vec @ inv_cov_mat_d @ res_vec
        return -0.5 * chi2

    chain_name = f"{chains_dir}/chain_{test}"
    info = {
        "likelihood": {"my_like": logL},
        "params": {
            f"phi{i}": {
                "prior": {
                    "min": -5.,
                    "max": 5
                            },
                "proposal": 1e-1,
                "latex": fr"{name.replace('lat_iso_', '').replace('_', r'\_')}"
                    }
                for i, name in enumerate(arrays_set)
                },
        "sampler": {
            "mcmc": {
                "max_tries": 1e7,
                "Rminus1_stop": 0.005,
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
    
    results_dict[test] =  {ar: [samples[test].mean(name), np.sqrt(samples[test].cov([name])[0, 0])] for ar, name in arrays_angle_names.items()}

    all_params_set[test] = [_ for _ in samples[test].getParamNames().list() if "chi" not in _] # get param names phi1, phi2 ...
    all_params_labels_set[test] = [_ for _ in samples[test].getParamNames().labels() if "chi" not in _] # get param labels i1_f090, ...

    phi_vec = get_angles_vec(*[results_dict[test][ar][0] for ar in arrays_angle_names.keys()])

    for i, (spec1, spec2) in enumerate(angle_spec_set_full):
        fig, ax  = plt.subplots()
        ax.axhline(0., color='black', ls='--', lw=1)
        ax.errorbar(lb[ell_masks[i]], data_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]], np.sqrt(cov_mat_d[len_ls_cumsum[i]:len_ls_cumsum[i+1], len_ls_cumsum[i]:len_ls_cumsum[i+1]].diagonal()), label=(spec1, spec2), color='black', lw=.8, marker='.')
        ax.plot(lb[ell_masks[i]], phi_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]] * model_vec[len_ls_cumsum[i]:len_ls_cumsum[i+1]], color="blue")
        ax.set_xlabel(r"$\ell$", fontsize=18)
        ax.set_ylabel(fr"$D_\ell^{{{modes[i]}}}$", fontsize=18)
        fig.suptitle(f"{spec1}x{spec2}")
        plt.savefig(plot_output_spec_dir + f"/{test}_{spec1}x{spec2}_{modes[i]}.png")
        plt.close()
    
    # Compute the average polarization angle
    cov_params = samples[test].cov(pars=all_params_set[test])
    means_vec = np.array([samples[test].mean(p) for p in all_params_set[test]])

    cov_inv = np.linalg.inv(cov_params)
    ones = np.ones(len(all_params_set[test]))

    results_dict[test]["mean"] = [(ones @ cov_inv @ means_vec) / (ones @ cov_inv @ ones), np.sqrt(1.0 / (ones @ cov_inv @ ones))]
    
    # Combine all points using inverse variance
    H = np.tile(np.eye(len_ls_cumsum[1]), (len(angle_spec_set_full), 1))
    cov_comb = np.linalg.inv(H.T @ inv_cov_mat_d @ H)
    mu_comb = cov_comb @ H.T @ inv_cov_mat_d @ data_vec
    
    pte_zero = 1 - ss.chi2(len(lb[ell_masks[0]])).cdf((mu_comb) @ np.linalg.inv(cov_comb) @ (mu_comb))
    pte_bestfit = 1 - ss.chi2(len(lb[ell_masks[0]])).cdf((mu_comb - np.sin(4 * np.deg2rad(results_dict[test]["mean"][0])) / 2 * model_vec[len_ls_cumsum[0]:len_ls_cumsum[1]]) @ np.linalg.inv(cov_comb) @ (mu_comb - np.sin(4 * np.deg2rad(results_dict[test]["mean"][0])) / 2 * model_vec[len_ls_cumsum[0]:len_ls_cumsum[1]]))
    
    fig, ax  = plt.subplots(dpi=200)
    ax.axhline(0., color='black', ls='--', lw=1, label=f"Null hypothesis : PTE = {pte_zero:.3f}")
    ax.plot(lb[ell_masks[0]], np.sin(4 * np.deg2rad(results_dict[test]["mean"][0])) / 2 * model_vec[len_ls_cumsum[0]:len_ls_cumsum[1]], color="blue", label=f"Bestfit ({results_dict[test]["mean"][0]:.2f}deg) : PTE = {pte_bestfit:.3f}")
    ax.fill_between(
        lb[ell_masks[0]],
        np.sin(4 * np.deg2rad(results_dict[test]["mean"][0] - results_dict[test]["mean"][1])) / 2 * model_vec[len_ls_cumsum[0]:len_ls_cumsum[1]],
        np.sin(4 * np.deg2rad(results_dict[test]["mean"][0] + results_dict[test]["mean"][1])) / 2 * model_vec[len_ls_cumsum[0]:len_ls_cumsum[1]],
        color="blue",
        alpha=.4,
    )
    ax.errorbar(lb[ell_masks[0]], mu_comb, np.sqrt(cov_comb.diagonal()), color='black', lw=1, marker='.', ls='', )
    ax.set_xlabel(r"$\ell$", fontsize=18)
    ax.set_ylabel(fr"$D_\ell^{{{modes[i]}}}$", fontsize=16)
    ax.legend()
    fig.suptitle(f"Mean")
    plt.tight_layout()
    plt.savefig(plot_output_dir + f"/{test}_MEAN.png")
    plt.close()
    
    
color_list = ["blue", "red", "green", "darkorange", "purple", "darkcyan", "chocolate"]

for j, (test, results_subdict) in enumerate(results_dict.items()):
    results_to_save = {
        name: float(cal)
        for name, (cal, std) in results_subdict.items()
    }
    stds_to_save = {
        name: float(std)
        for name, (cal, std) in results_subdict.items()
    }

    file = open(f"{pol_angle_dir}/angles_dict_{test}.yaml", "w")
    yaml.dump(results_to_save, file)
    file.close()
    
    file = open(f"{pol_angle_dir}/angles_errs_dict_{test}.yaml", "w")
    yaml.dump(stds_to_save, file)
    file.close()

gdplot = gdplt.get_subplot_plotter(width_inch=5)
gdplot.triangle_plot(
    [samples[test] for test in pol_angle_infos["angle_spec_sets"].keys()],
    params=list(all_params_set[test]),  # TODO: make this dynamical for tests
    title_limit=1,
    legend_labels=[test for test in pol_angle_infos["angle_spec_sets"].keys()],
    contour_colors=color_list[:len(test_names)],
    dpi=200,
)
plt.savefig(f'{plot_output_dir}/triangle_plot_{'_'.join(test_names)}.png')
plt.close()

for test, smpl in samples.items():
    fig, ax = plt.subplots(dpi=200)
    params_1DD = {}
    phi_linspace = np.linspace(-2, 2, 1000)
    ax.axvline(0, color='black', ls='--')
    for param, label in zip(all_params_set[test], all_params_labels_set[test]):
        pdf = smpl.get1DDensity(name=param).Prob(x=phi_linspace)
        ax.plot(phi_linspace, pdf, label=fr"${label}$", lw=2)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Polarization angle (degree)", fontsize=16)
    ax.legend()
    plt.savefig(f'{plot_output_dir}/{test}_angles_distrib.png')
    plt.close()
