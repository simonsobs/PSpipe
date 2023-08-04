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
from pspipe_utils import best_fits,  pspipe_list

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

windows_dir = "windows"
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


# l_cmb, cmb_dict = best_fits.cmb_dict_from_file(bestfit_dir + "/cmb.dat", lmax, spectra)

arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
n_splits = {sv: d[f"n_splits_{sv}"] for sv in surveys}
array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]

spec_name_list = pspipe_list.get_spec_name_list(d)
sv_array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]
l_fg, fg_dict = best_fits.fg_dict_from_files(bestfit_dir + "/fg_{}x{}.dat", sv_array_list, lmax, spectra)

l_cmb, cmb_dict = best_fits.cmb_dict_from_file(bestfit_dir + "/cmb.dat", lmax, spectra)
l, ps_all = best_fits.get_all_best_fit(spec_name_list,
                                               l_cmb,
                                               cmb_dict,
                                               fg_dict,
                                               spectra)

noise_dict = {}
f_name_noise = f"{noise_dir}/Dl_" + "{}x{}_{}_noise_model.dat"
for sv in surveys:
    for ar in arrays[sv]:
        split_list = {sv: [f"{ar}_{i}" for i in range(n_splits[sv])]}
        _, nlth = best_fits.noise_dict_from_files(f_name_noise, [sv], split_list, lmax=d["lmax"],spectra=spectra)
        noise_dict.update(nlth)


bl_dict = {}
for sv in surveys:
    for ar in arrays[sv]:
        l_beam, bl_dict[sv, ar] = pspy_utils.read_beam_file(d[f"beam_{sv}_{ar}"])
        id_beam = np.where((l_beam >= 2) & (l_beam < lmax))
        bl_dict[sv, ar] = bl_dict[sv, ar][id_beam]

cov_name = []
for sv in surveys:
    for ar in arrays[sv]:
        for id1, xsplit1 in enumerate(combinations(range(n_splits[sv]), 2)):
            for id2, xsplit2 in enumerate(combinations(range(n_splits[sv]), 2)):
                if id1 > id2: continue


                for pol in ("T",):

                    na = f"{sv}&{ar}&{xsplit1[0]}&{pol}"
                    nb = f"{sv}&{ar}&{xsplit1[1]}&{pol}"
                    nc = f"{sv}&{ar}&{xsplit2[0]}&{pol}"
                    nd = f"{sv}&{ar}&{xsplit2[1]}&{pol}"

                    cov_name.append((na, nb, nc, nd))

ncovs = len(cov_name)


log.info(f"number of covariance matrices to compute : {ncovs}")

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=ncovs - 1)

if len(sys.argv) == 4:
    log.info(f"computing only the covariance matrices : {int(sys.argv[2])}:{int(sys.argv[3])}")
    subtasks = subtasks[int(sys.argv[2]):int(sys.argv[3])]
log.info(subtasks)

for task in subtasks:
    na, nb, nc, nd = cov_name[task]


    sv_a, ar_a, split_a, pol_a = na.split("&")
    sv_b, ar_b, split_b, pol_b = nb.split("&")
    sv_c, ar_c, split_c, pol_c = nc.split("&")
    sv_d, ar_d, split_d, pol_d = nd.split("&")

    na = f"{sv_a}&{ar_a}"
    nb = f"{sv_b}&{ar_b}"
    nc = f"{sv_c}&{ar_c}"
    nd = f"{sv_d}&{ar_d}"

    na_r, nb_r, nc_r, nd_r = na.replace("&", "_"), nb.replace("&", "_"), nc.replace("&", "_"), nd.replace("&", "_")
    log.info(f"[task] cov element ({na_r} x {nb_r}, {nc_r} x {nd_r}) {split_a}{split_b}{split_c}{split_d} {pol_a}{pol_b}{pol_c}{pol_d}")

    splits = [split_a, split_b, split_c, split_d]
    survey_id = [pol_a + "a", pol_b + "b", pol_c + "c", pol_d + "d"]
    splitname2splitindex = dict(zip(splits, range(len(splits))))

    survey_combo = sv_a, sv_b, sv_c, sv_d
    array_combo = ar_a, ar_b, ar_c, ar_d

    win, var = {}, {}
    white_noise = {}
    for splitname1, id1, sv1, ar1 in zip(splits, survey_id, survey_combo, array_combo):
    #    log.info(f"{name1}, {id1}, {sv1}, {ar1}")
        split_index = splitname2splitindex[splitname1]
        ivar_fname = d[f"ivar_{sv1}_{ar1}"][split_index]
        ivar1 = so_map.read_map(ivar_fname).data
        var1 = np.reciprocal(ivar1,where=(ivar1!=0))
        maskpol = "T" if id1[0] == "T" else "pol"
        win[id1] = so_map.read_map(d[f"window_{maskpol}_{sv1}_{ar1}"])
        white_noise[id1] = so_cov.measure_white_noise_level(var1, win[id1].data)
        var[id1] = so_cov.make_weighted_variance_map(var1, win[id1])


    print(white_noise)


    Clth_dict = {}
    Rl_dict = {}

    for splitname1, id1, sv1, ar1 in zip(splits, survey_id, survey_combo, array_combo):
        s1 = splitname1.replace("set", "")
        X1 = id1[0]
        ndl = noise_dict[sv1, (f"{ar1}_{s1}"), (f"{ar1}_{s1}")][X1 + X1]
        ell = np.arange(2, lmax)
        dlfac = (ell * (ell + 1) / (2 * np.pi))  
        nl = ndl[:(lmax-2)] / dlfac
        
        Rl_dict[id1] = np.sqrt(nl / white_noise[id1])
        
        for name2, id2, sv2, ar2 in zip(splits, survey_id, survey_combo, array_combo):
            s2 = name2.replace("set", "")
            X2 = id2[0]
            # k = (f"{sv1}&{ar1}&{s1}", f"{sv2}&{ar2}&{s2}", X1+X2)
            dlth = ps_all[f"{sv1}&{ar1}", f"{sv2}&{ar2}", X1+X2]
            Clth_dict[id1 + id2] = dlth / dlfac[:(lmax-2)]


    couplings = so_cov.generate_aniso_couplings_TTTT(survey_id, splits, win, var, lmax)

    cov_tttt = so_cov.coupled_cov_aniso_same_pol(survey_id, Clth_dict, Rl_dict, couplings)

    np.save(f"{cov_dir}/analytic_aniso_coupled_cov_{na_r}_{split_a}x{nb_r}_{split_b}_{nc_r}_{split_c}x{nd_r}_{split_d}.npy", 
        cov_tttt)
