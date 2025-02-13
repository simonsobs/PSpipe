"""
This script compute the analytical covariance matrix elements
between split power spectra
"""
import sys
import numpy as np
from pspipe_utils import best_fits, log
from pspy import pspy_utils, so_cov, so_dict, so_map, so_mcm, so_mpi, so_spectra
from itertools import combinations_with_replacement as cwr
from itertools import combinations

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

mcms_dir = "mcms"
spectra_dir = "spectra"
noise_dir = "split_noise"
cov_dir = "split_covariances"
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

l_cmb, cmb_dict = best_fits.cmb_dict_from_file(bestfit_dir + "/cmb.dat", lmax, spectra)

arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
n_splits = {sv: d[f"n_splits_{sv}"] for sv in surveys}

array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]

l_fg, fg_dict = best_fits.fg_dict_from_files(bestfit_dir + "/fg_{}x{}.dat", array_list, lmax, spectra)

spec_name_list = []
noise_dict = {}
f_name_noise = f"{noise_dir}/Dl_" + "{}x{}_{}_noise_model.dat"
for sv in surveys:
    for ar in arrays[sv]:

        split_list = {sv: [f"{ar}_{i}" for i in range(n_splits[sv])]}
        _, nlth = best_fits.noise_dict_from_files(f_name_noise, [sv], split_list, lmax=d["lmax"],spectra=spectra)
        noise_dict.update(nlth)

        for id_split_1, id_split_2 in cwr(range(n_splits[sv]), 2):
            spec_name_list.append(f"{sv}&{ar}&{id_split_1}x{sv}&{ar}&{id_split_2}")

bl_dict = {}
for sv in surveys:
    for ar in arrays[sv]:
        l_beam, bl_dict[sv, ar] = pspy_utils.read_beam_file(d[f"beam_{sv}_{ar}"])
        id_beam = np.where((l_beam >= 2) & (l_beam < lmax))
        bl_dict[sv, ar] = bl_dict[sv, ar][id_beam]

lth, ps_all_th, nl_all_th = best_fits.get_all_best_fit(spec_name_list, l_cmb,
                                                       cmb_dict, fg_dict, spectra,
                                                       nl_dict=noise_dict, bl_dict=bl_dict)

ps_and_noise_dict = {k: ps_all_th[k] + nl_all_th[k] for k in ps_all_th.keys()}

cov_name = []
for sv in surveys:
    for ar in arrays[sv]:
        for id1, xsplit1 in enumerate(combinations(range(n_splits[sv]), 2)):
            for id2, xsplit2 in enumerate(combinations(range(n_splits[sv]), 2)):
                if id1 > id2: continue

                na = f"{sv}&{ar}&{xsplit1[0]}"
                nb = f"{sv}&{ar}&{xsplit1[1]}"
                nc = f"{sv}&{ar}&{xsplit2[0]}"
                nd = f"{sv}&{ar}&{xsplit2[1]}"

                cov_name.append((na, nb, nc, nd))

ncovs = len(cov_name)

if d["use_toeplitz_cov"] == True:
    log.info("we will use the toeplitz approximation")
    l_exact, l_band, l_toep = 800, 2000, 2750
else:
    l_exact, l_band, l_toep = None, None, None

log.info(f"number of covariance matrices to compute : {ncovs}")

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=ncovs - 1)
for task in subtasks:

    task = int(task)
    na, nb, nc, nd = cov_name[task]
    sv_a, ar_a, split_a = na.split("&")
    sv_b, ar_b, split_b = nb.split("&")
    sv_c, ar_c, split_c = nc.split("&")
    sv_d, ar_d, split_d = nd.split("&")

    na = f"{sv_a}&{ar_a}"
    nb = f"{sv_b}&{ar_b}"
    nc = f"{sv_c}&{ar_c}"
    nd = f"{sv_d}&{ar_d}"

    na_r, nb_r, nc_r, nd_r = na.replace("&", "_"), nb.replace("&", "_"), nc.replace("&", "_"), nd.replace("&", "_")
    log.info(f"[task] cov element ({na_r} x {nb_r}, {nc_r} x {nd_r}) {split_a}{split_b}{split_c}{split_d}")

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

    cross_dict = {
        "a": f"{sv_a}&{ar_a}&{split_a}",
        "b": f"{sv_b}&{ar_b}&{split_b}",
        "c": f"{sv_c}&{ar_c}&{split_c}",
        "d": f"{sv_d}&{ar_d}&{split_d}"}

    Dlth_dict = {}
    for field1 in ["T", "E", "B"]:
        for id_1, cross_name_1 in cross_dict.items():
            for field2 in ["T", "E", "B"]:
                for id_2, cross_name_2 in cross_dict.items():
                    Dlth_dict[f"{field1}{id_1}{field2}{id_2}"] = ps_and_noise_dict[cross_name_1, cross_name_2, field1+field2]

    analytic_cov = so_cov.cov_spin0and2(Dlth_dict,
                                        coupling,
                                        binning_file,
                                        lmax,
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
        analytic_cov /= np.outer(one_d_tf ** (1./4.), one_d_tf ** (1./4.))

    np.save(f"{cov_dir}/analytic_cov_{na_r}_{split_a}x{nb_r}_{split_b}_{nc_r}_{split_c}x{nd_r}_{split_d}.npy", analytic_cov)
