"""
This script compute the analytical covariance matrix elements.
"""
import sys

import numpy as np
from pspipe_utils import best_fits, log, pspipe_list, misc
from pspy import pspy_utils, so_cov, so_dict, so_map, so_mcm, so_mpi

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

mcms_dir = "mcms"
spectra_dir = "spectra"
noise_dir = "noise_model"
cov_dir = "covariances"
bestfit_dir = "best_fits"
sq_win_alms_dir = "sq_win_alms"

pspy_utils.create_directory(cov_dir)
surveys = d["surveys"]
binning_file = d["binning_file"]
lmax = d["lmax"]
niter = d["niter"]
binned_mcm = d["binned_mcm"]
apply_kspace_filter = d["apply_kspace_filter"]
cov_T_E_only = d["cov_T_E_only"]


spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

# fast_coupling is designed to be fast but is not for general usage
# In particular we assume that the same window function is used in T and Pol
fast_coupling = True

arrays, n_splits, bl_dict = {}, {}, {}
for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    for ar in arrays[sv]:
        l_beam, bl = misc.read_beams(d[f"beam_T_{sv}_{ar}"], d[f"beam_pol_{sv}_{ar}"])
        id_beam = np.where((l_beam >= 2) & (l_beam < lmax))
        bl_dict[sv, ar] = {}
        for field in ["T", "E", "B"]:
            bl_dict[sv, ar][field] = bl[field][id_beam]
            
        n_splits[sv] = d[f"n_splits_{sv}"]
        assert n_splits[sv] == len(d.get(f"maps_{sv}_{ar}", [0]*n_splits[sv])), "the number of splits does not correspond to the number of maps"

        if fast_coupling:
            # This loop check that this is what was specified in the dictfile
            assert d[f"window_T_{sv}_{ar}"] == d[f"window_pol_{sv}_{ar}"], "T and pol windows have to be the same"

l_cmb, cmb_dict = best_fits.cmb_dict_from_file(bestfit_dir + "/cmb.dat", lmax, spectra)

array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]
l_fg, fg_dict = best_fits.fg_dict_from_files(bestfit_dir + "/fg_{}x{}.dat", array_list, lmax, spectra)

f_name_noise = noise_dir + "/mean_{}x{}_{}_noise.dat"
l_noise, nl_dict = best_fits.noise_dict_from_files(f_name_noise, surveys, arrays, lmax, spectra, n_splits=n_splits)

spec_name_list = pspipe_list.get_spec_name_list(d)
l, ps_all, nl_all = best_fits.get_all_best_fit(spec_name_list,
                                               l_cmb,
                                               cmb_dict,
                                               fg_dict,
                                               spectra,
                                               nl_dict=nl_dict,
                                               bl_dict=bl_dict)

ncovs, na_list, nb_list, nc_list, nd_list = pspipe_list.get_covariances_list(d)

if d["use_toeplitz_cov"] == True:
    log.info("we will use the toeplitz approximation")
    l_exact, l_band, l_toep = 800, 2000, 2750
else:
    l_exact, l_band, l_toep = None, None, None

log.info(f"number of covariance matrices to compute : {ncovs}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=ncovs - 1)
log.info(subtasks)
for task in subtasks:
    task = int(task)
    na, nb, nc, nd = na_list[task], nb_list[task], nc_list[task], nd_list[task]
    na_r, nb_r, nc_r, nd_r = na.replace("&", "_"), nb.replace("&", "_"), nc.replace("&", "_"), nd.replace("&", "_")

    log.info(f"[{task}] cov element ({na_r} x {nb_r}, {nc_r} x {nd_r})")

    if fast_coupling:

        coupling = so_cov.fast_cov_coupling_spin0and2(sq_win_alms_dir,
                                                     [na_r, nb_r, nc_r, nd_r],
                                                     lmax,
                                                     l_exact=l_exact,
                                                     l_band=l_band,
                                                     l_toep=l_toep)

    else:
        win = {}
        win["Ta"] = so_map.read_map(d[f"window_T_{na_r}"])
        win["Tb"] = so_map.read_map(d[f"window_T_{nb_r}"])
        win["Tc"] = so_map.read_map(d[f"window_T_{nc_r}"])
        win["Td"] = so_map.read_map(d[f"window_T_{nd_r}"])
        win["Pa"] = so_map.read_map(d[f"window_pol_{na_r}"])
        win["Pb"] = so_map.read_map(d[f"window_pol_{nb_r}"])
        win["Pc"] = so_map.read_map(d[f"window_pol_{nc_r}"])
        win["Pd"] = so_map.read_map(d[f"window_pol_{nd_r}"])

        coupling = so_cov.cov_coupling_spin0and2_simple(win,
                                                        lmax,
                                                        niter=niter,
                                                        l_exact=l_exact,
                                                        l_band=l_band,
                                                        l_toep=l_toep)



    try: mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix=f"{mcms_dir}/{na_r}x{nb_r}", spin_pairs=spin_pairs)
    except: mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix=f"{mcms_dir}/{nb_r}x{na_r}", spin_pairs=spin_pairs)

    try: mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix=f"{mcms_dir}/{nc_r}x{nd_r}", spin_pairs=spin_pairs)
    except:  mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix=f"{mcms_dir}/{nd_r}x{nc_r}", spin_pairs=spin_pairs)


    analytic_cov = so_cov.generalized_cov_spin0and2(coupling,
                                                    [na, nb, nc, nd],
                                                    n_splits,
                                                    ps_all,
                                                    nl_all,
                                                    lmax,
                                                    binning_file,
                                                    mbb_inv_ab,
                                                    mbb_inv_cd,
                                                    binned_mcm=binned_mcm,
                                                    cov_T_E_only=cov_T_E_only,
                                                    dtype=np.float32)

    if apply_kspace_filter == True:
        # Some heuristic correction for the number of modes lost due to the transfer function
        # This should be tested against simulation and revisited

        one_d_tf_ab = np.loadtxt(f"{spectra_dir}/one_dimension_kspace_tf_{na_r}x{nb_r}.dat")
        one_d_tf_cd = np.loadtxt(f"{spectra_dir}/one_dimension_kspace_tf_{nc_r}x{nd_r}.dat")
        one_d_tf = np.minimum(one_d_tf_ab, one_d_tf_cd)
        # sqrt(tf) is an approx for the number of modes masked in a given map so (2l+1)*fsky*sqrt(tf)
        # is our proxy for the number of modes
        analytic_cov /= np.outer(one_d_tf ** (1.0 / 4.0), one_d_tf ** (1.0 / 4.0))

    np.save(f"{cov_dir}/analytic_cov_{na_r}x{nb_r}_{nc_r}x{nd_r}.npy", analytic_cov)
