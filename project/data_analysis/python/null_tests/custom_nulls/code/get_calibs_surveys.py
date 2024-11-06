import matplotlib
matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_spectra, so_cov
from pspipe_utils import consistency, log
import numpy as np
import pylab as plt
import pickle
import sys


def get_proj_pattern(test, map_set, ref_map_set):

    if test == "AxA-AxP":
        proj_pattern = np.array([1, -1, 0])
        #    [1, -1, 0]]  x [ AxA, AxP, PxP].T    =  AxA - AxP
        name = f"{map_set}x{map_set}-{map_set}x{ref_map_set}"
    elif test == "AxA-PxP":
        proj_pattern = np.array([1, 0, -1])
         #    [1, 0, -1]]  x [ AxA, AxP, PxP].T    =   AxA - PxP
        name = f"{map_set}x{map_set}-{ref_map_set}x{ref_map_set}"
    elif test == "PxP-AxP":
        proj_pattern = np.array([0, -1, 1])
        #    [[0, -1, 1]]   x [ AxA, AxP, PxP].T    =    PxP - AxP
        name = f"{ref_map_set}x{ref_map_set}-{map_set}x{ref_map_set}"

    return  name, proj_pattern
    
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

planck_corr = True
subtract_bf_fg = True


output_dir = "calibration_results"
spec_dir = "spectra"
bestfir_dir = "best_fits"
cov_dir = "covariances"

if planck_corr:
    spec_dir = "spectra_leak_corr_planck_bias_corr"
    output_dir += "_planck_bias_corrected"

if subtract_bf_fg:
    output_dir += "_fg_sub"



_, _, lb, _ = pspy_utils.read_binning_file(d["binning_file"], d["lmax"])
n_bins = len(lb)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if d["cov_T_E_only"] == True:
    modes = ["TT", "TE", "ET", "EE"]
else:
    modes = spectra


# Create output dirs

residual_output_dir = f"{output_dir}/residuals"
plot_output_dir = f"{output_dir}/plots"
chains_dir = f"{output_dir}/chains"

pspy_utils.create_directory(output_dir)
pspy_utils.create_directory(residual_output_dir)
pspy_utils.create_directory(chains_dir)
pspy_utils.create_directory(plot_output_dir)


surveys = d["surveys"]
try:
    surveys.remove("Planck")
except:
    pass
    
results_dict = {}
ref_map_sets = {}
for sv in surveys:

    # Define the multipole range used to obtain
    # the calibration amplitudes
    multipole_range = {f"{sv}_pa4_f150": [1250, 1800],
                       f"{sv}_pa4_f220": [1250, 2000],
                       f"{sv}_pa5_f090": [1000, 2000],
                       f"{sv}_pa5_f150": [800, 2000],
                       f"{sv}_pa6_f090": [1000, 2000],
                       f"{sv}_pa6_f150": [600, 2000]}

    # Define the reference arrays
    ref_map_sets[f"{sv}_pa4_f150"] = "Planck_f143"
    ref_map_sets[f"{sv}_pa4_f220"] = "Planck_f217"
    ref_map_sets[f"{sv}_pa5_f090"] = "Planck_f143"
    ref_map_sets[f"{sv}_pa5_f150"] = "Planck_f143"
    ref_map_sets[f"{sv}_pa6_f090"] = "Planck_f143"
    ref_map_sets[f"{sv}_pa6_f150"] = "Planck_f143"

    y_lims = {"TT": (-100000, 75000),}

    tests = ["AxA-AxP", "AxA-PxP", "PxP-AxP"]

    for test in tests:
        for ar in d[f"arrays_{sv}"]:
    
            map_set = f"{sv}_{ar}"
            ref_map_set = ref_map_sets[map_set]

            name, proj_pattern = get_proj_pattern(test, map_set, ref_map_set)
        
            spectra_for_cal = [(map_set, map_set, "TT"),
                               (map_set, ref_map_set, "TT"),
                               (ref_map_set, ref_map_set, "TT")]

            # Load spectra and cov
            ps_dict = {}
            cov_dict = {}
            for i, (ms1, ms2, m1) in enumerate(spectra_for_cal):
                _, ps = so_spectra.read_ps(f"{spec_dir}/Dl_{ms1}x{ms2}_cross.dat", spectra=spectra)
                
                ps_dict[ms1, ms2, m1] = ps[m1]
            
                if (m1 == "TT") & (subtract_bf_fg):
                    log.info(f"remove fg {m1}  {ms1} x {ms2}")
                    l_fg, bf_fg = so_spectra.read_ps(f"{bestfir_dir}/fg_{ms1}x{ms2}.dat", spectra=spectra)
                    _, bf_fg_TT_binned = pspy_utils.naive_binning(l_fg, bf_fg["TT"], d["binning_file"], d["lmax"])
                    ps_dict[ms1, ms2, m1] -= bf_fg_TT_binned


                ps_dict[ms1, ms2, m1] = ps[m1]
                for j, (ms3, ms4, m2) in enumerate(spectra_for_cal):
                    if j < i: continue
                    cov = np.load(f"{cov_dir}/analytic_cov_{ms1}x{ms2}_{ms3}x{ms4}.npy")
                    cov = so_cov.selectblock(cov, modes, n_bins = n_bins, block = m1 + m2)
                    cov_dict[(ms1, ms2, m1), (ms3, ms4, m2)] = cov

            # Concatenation
            spec_vec, full_cov = consistency.append_spectra_and_cov(ps_dict, cov_dict, spectra_for_cal)

            # Save and plot residuals before calibration
            res_spectrum, res_cov = consistency.project_spectra_vec_and_cov(spec_vec, full_cov, proj_pattern)
            np.savetxt(f"{residual_output_dir}/residual_{name}_before.dat", np.array([lb, res_spectrum]).T)
            np.savetxt(f"{residual_output_dir}/residual_cov_{name}.dat", res_cov)

            lmin, lmax = multipole_range[map_set]
            id = np.where((lb >= lmin) & (lb <= lmax))[0]
            consistency.plot_residual(lb, res_spectrum, {"analytical": res_cov}, "TT", f"{map_set} {test}",
                                    f"{plot_output_dir}/residual_{name}_before",
                                    lrange=id, l_pow=1, ylims=y_lims["TT"])

            # Calibrate the spectra
            cal_mean, cal_std = consistency.get_calibration_amplitudes(spec_vec, full_cov,
                                                                    proj_pattern, "TT", id,
                                                                    f"{chains_dir}/{name}")


            results_dict[name] = {"multipole_range": multipole_range[map_set],
                                "ref_map_set": ref_map_set,
                                "calibs": [cal_mean, cal_std]}

            calib_vec = np.array([cal_mean**2, cal_mean, 1])
            res_spectrum, res_cov = consistency.project_spectra_vec_and_cov(spec_vec, full_cov,
                                                                            proj_pattern,
                                                                            calib_vec = calib_vec)
  
            np.savetxt(f"{residual_output_dir}/residual_{name}_after.dat", np.array([lb, res_spectrum]).T)
            consistency.plot_residual(lb, res_spectrum, {"analytical": res_cov}, "TT", f"{map_set} {test}",
                                    f"{plot_output_dir}/residual_{name}_after",
                                    lrange=id, l_pow=1, ylims=y_lims["TT"])





for sv in surveys:

    # plot the cal factors
    color_list =  ["blue", "red", "green"]

    for i, ar in enumerate(d[f"arrays_{sv}"]):
        map_set = f"{sv}_{ar}"
        ref_map_set = ref_map_sets[map_set]
        print(f"**************")
        print(f"calibration {map_set} with {ref_map_set}")

        for j, test in enumerate(tests):
            name, _ = get_proj_pattern(test, map_set, ref_map_set)
            cal, std = results_dict[name]["calibs"]
            print(f"{test}, cal: {cal}, sigma cal: {std}")

            plt.errorbar(i + 0.9 + j * 0.1, cal, std, label = test,
                        color = color_list[j], marker = ".",
                        ls = "None",
                        markersize=6.5,
                        markeredgewidth=2)

        if i == 0:
            plt.legend(fontsize = 15)

    x = np.arange(1, len(d[f"arrays_{sv}"]) + 1)
    plt.xticks(x, d[f"arrays_{sv}"])
    plt.ylim(0.967, 1.06)
    plt.tight_layout()
    plt.savefig(f"{plot_output_dir}/calibs_summary_{sv}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()

    pickle.dump(results_dict, open(f"{output_dir}/calibs_dict_{sv}.pkl", "wb"))
