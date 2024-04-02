"""
This script compute the montecarlo chi2 distribution using the estimted simulation power spectra, the covariance matrix and the input theory, we use the ell cut of ACT DR6
You can choose sim_spec_dir = "sim_spectra" for sim without systematic model or sim_spec_dir = "sim_spectra_syst" for sim with systematic, the code will choose
the corresponding covariance matrix
"""
import numpy as np
import pylab as plt
from pspipe_utils import covariance, pspipe_list, log
from pspy import so_cov, so_dict, pspy_utils, so_spectra
from pixell import utils
import os
import scipy.stats as stats


d = so_dict.so_dict()
d.read_from_file('/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/tlouis_test_cov_20240125/global_dr6_v4.dict')
log = log.get_logger(**d)

my_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])
log.info(my_idx)
    
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
binning_file = d["binning_file"]
lmax = d["lmax"]
type = d["type"]
iStart = d["iStart"]
iStop = d["iStop"]

cov_dir = d['data_dir'] + "covariances"
bestfit_dir = d['data_dir'] + "best_fits"
sim_spec_dir =  d['data_dir'] + "sim_spectra"
mcm_dir = d['data_dir'] + "mcms"
plot_dir = "/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/tlouis_test_cov_20240125/zatkins_results/plots/chi2/20240324"

pspy_utils.create_directory(plot_dir)

bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

# if sim_spec_dir == "sim_spectra":
#     x_ar_cov = np.load(f"{cov_dir}/x_ar_final_cov_sim.npy")
# if sim_spec_dir == "sim_spectra_syst":
#     x_ar_cov = np.load(f"{cov_dir}/x_ar_final_cov_data.npy")
x_ar_cov_old = np.load(f"{cov_dir}/x_ar_analytic_cov.npy")
x_ar_cov_zach_20240202 = np.load('/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/cov_dr6_v4_20231128/covariances_efficient/x_ar_analytic_cov_zach_bugfix.npy')
x_ar_cov_zach_20240324 = np.load('/scratch/gpfs/zatkins/data/simonsobs/PSpipe/project/ana_cov_comp/cov_dr6_v4_20240324/covariances/x_ar_analytic_cov.npy')

x_ar_mc_cov = np.load(f"{cov_dir}/x_ar_mc_cov.npy")
x_ar_cov_old = covariance.correct_analytical_cov_keep_res_diag(x_ar_cov_old, x_ar_mc_cov)
x_ar_cov_zach_20240202 = covariance.correct_analytical_cov_keep_res_diag(x_ar_cov_zach_20240202, x_ar_mc_cov)
x_ar_cov_zach_20240324 = covariance.correct_analytical_cov_keep_res_diag(x_ar_cov_zach_20240324, x_ar_mc_cov)

cov_list = [x_ar_cov_old, x_ar_cov_zach_20240202, x_ar_cov_zach_20240324]
cov_name_list = ["old analytic", "new analytic 20240202", "new analytic 20240324"]

selected_spectra = [spectra, ["TT", "TE", "ET", "EE"], ["TT"], ["TE"], ["ET"], ["TB"], ["BT"], ["EE"], ["EB"], ["BE"], ["BB"]]
name_list = ["all", "TT-TE-ET-EE", "TT", "TE", "ET", "TB", "BT", "EE", "EB", "BE", "BB"]

# Note that to match the null test selection and likelihood selection, we use a slightly different lmin
# This is because the likelihood cut based on the bin center, while most of the power spectrum pipeline cut
# based on the bin edges.
spectra_cuts = {
    "dr6_pa4_f220": dict(T=[975, lmax], P=[lmax, lmax]),
    "dr6_pa5_f150": dict(T=[775, lmax], P=[475, lmax]),
    "dr6_pa6_f150": dict(T=[575, lmax], P=[475, lmax]),
    "dr6_pa5_f090": dict(T=[975, lmax], P=[475, lmax]),
    "dr6_pa6_f090": dict(T=[975, lmax], P=[475, lmax]),
}
only_TT_map_set = ["dr6_pa4_f220"]

theory_vec = covariance.read_x_ar_theory_vec(bestfit_dir, mcm_dir, spec_name_list, lmax, spectra_order=spectra)

idx = 0

for name, select in zip(name_list, selected_spectra):

    if idx == my_idx:

        bin_out_dict, indices = covariance.get_indices(bin_low,
                                                    bin_high,
                                                    bin_mean,
                                                    spec_name_list,
                                                    spectra_cuts=spectra_cuts,
                                                    spectra_order=spectra,
                                                    selected_spectra=select,
                                                    only_TT_map_set=only_TT_map_set)
                                                    
                                                    
                                                    

        # some plot to check that the selection worked, we use sim 00000
        spec_plot_dir = f"{plot_dir}/{name}"
        pspy_utils.create_directory(spec_plot_dir)
        
        data_vec = covariance.read_x_ar_spectra_vec(sim_spec_dir, spec_name_list, f"cross_00000", spectra_order=spectra, type=type)
        sub_data_vec = data_vec[indices]
        sub_theory_vec = theory_vec[indices]
        sub_cov = x_ar_cov_zach_20240324[np.ix_(indices,indices)]

        for my_spec in bin_out_dict.keys():
            s_name, spectrum = my_spec
            id, lb = bin_out_dict[my_spec]

            lb_, Db = so_spectra.read_ps(f"{sim_spec_dir}/Dl_{s_name}_cross_00000.dat", spectra=spectra)
            
            plt.figure(figsize=(12,8))
            plt.title(f"{my_spec}, min={np.min(lb)}, max={np.max(lb)}")
            if spectrum == "TT":
                plt.semilogy()
            plt.plot(lb_, Db[spectrum], label="original spectrum")
            plt.errorbar(lb, sub_data_vec[id], np.sqrt(sub_cov[np.ix_(id,id)].diagonal()), fmt=".", label="selected spectrum")
            plt.plot(lb, sub_theory_vec[id], "--", color="gray",alpha=0.3, label="theory")
            plt.legend()
            plt.savefig(f"{spec_plot_dir}/{spectrum}_{s_name}.png", bbox_inches="tight")
            plt.clf()
            plt.close()

        plt.figure(figsize=(12,8))

        inv_sub_cov_list = []
        cov_chi2_list = []
        for cov_name, cov in zip(cov_name_list, cov_list):
            print(cov_name)
            inv_sub_cov_list.append(np.linalg.inv(cov[np.ix_(indices,indices)]))
            cov_chi2_list.append([])
        
        for iii in range(iStart, iStop + 1):
            data_vec = covariance.read_x_ar_spectra_vec(sim_spec_dir, spec_name_list, f"cross_{iii:05d}", spectra_order=spectra, type=type)
            res = data_vec[indices] - theory_vec[indices]

            for cov_name, inv_sub_cov, chi2_list in zip(cov_name_list, inv_sub_cov_list, cov_chi2_list):
                chi2 = res @ inv_sub_cov @ res
                chi2_list += [chi2]
                log.info(f"{name} Sim number, {iii}, {cov_name} chi2 = {chi2:.3f}, dof = {len(res)}")
        
        for cov_name, chi2_list in zip(cov_name_list, cov_chi2_list):
            y = np.array(chi2_list)
            plt.hist(y, bins=40, density=True, histtype="step", label=f"sims chi2 distribution: {cov_name} dof={y.mean():.1f}")
            

        x = np.arange(len(res) - 400, len(res) + 400)
        y = stats.chi2.pdf(x, df=len(res))
        plt.title(name + ' MC corrected')
        plt.plot(
            x,
            y,
            "-",
            linewidth=2,
            color="black",
            label=f"expected distribution dof={len(res)}",
        )
        plt.legend()
        plt.savefig(f"{plot_dir}/histo_{name}_corrected.png", bbox_inches="tight")
        plt.clf()
        plt.close()

    idx += 1