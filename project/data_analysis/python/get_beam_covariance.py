"""
This script compute the analytical beam covariance matrix elements.
(TO BE TESTED AGAINS MONTECARLO)
"""
import sys

import numpy as np
from pspy import pspy_utils, so_dict, so_mpi, so_cov
from pspipe_utils import pspipe_list, best_fits, log

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

cov_dir = "covariances"
bestfit_dir = "best_fits"

pspy_utils.create_directory(cov_dir)
surveys = d["surveys"]
arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
binning_file = d["binning_file"]
lmax = d["lmax"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
cov_T_E_only = d["cov_T_E_only"]

array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]
lth, cmb_and_fg_dict = best_fits.fg_dict_from_files(bestfit_dir + "/fg_{}x{}.dat",
                                                    array_list,
                                                    lmax,
                                                    spectra,
                                                    f_name_cmb=bestfit_dir + "/cmb.dat")


ps_all, norm_beam_cov = {}, {}


log.info(f"construct best fit for all cross array spectra")

for id_sv1, sv1 in enumerate(surveys):
    for id_ar1, ar1 in enumerate(d[f"arrays_{sv1}"]):

        # technically we should treat both T and P beam, but they are the same for ACT
        # and for Planck beam cov is assumed to be negligible
        data = np.loadtxt(d[f"beam_T_{sv1}_{ar1}"])

        _, bl, error_modes  = data[2: lmax, 0], data[2: lmax, 1], data[2: lmax, 2:]
        beam_cov =  error_modes.dot(error_modes.T)

        norm_beam_cov[sv1, ar1] = beam_cov / np.outer(bl, bl)


        for id_sv2, sv2 in enumerate(surveys):
            for id_ar2, ar2 in enumerate(d[f"arrays_{sv2}"]):

                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue

                for spec in spectra:
                    ps_all[f"{sv1}&{ar1}", f"{sv2}&{ar2}", spec] = cmb_and_fg_dict[f"{sv1}_{ar1}", f"{sv2}_{ar2}"][spec]
                    ps_all[f"{sv2}&{ar2}", f"{sv1}&{ar1}", spec] = ps_all[f"{sv1}&{ar1}", f"{sv2}&{ar2}", spec]


# prepare the mpi computation


log.info(f"construct block beam covariance")

ncovs, na_list, nb_list, nc_list, nd_list = pspipe_list.get_covariances_list(d)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=ncovs - 1)
print(subtasks)
for task in subtasks:
    task = int(task)
    na, nb, nc, nd = na_list[task], nb_list[task], nc_list[task], nd_list[task]
    id_element = [na, nb, nc, nd]

    beam_cov = so_cov.covariance_element_beam(id_element, ps_all, norm_beam_cov, binning_file, lmax, cov_T_E_only=cov_T_E_only)

    na_r, nb_r, nc_r, nd_r = na.replace("&", "_"), nb.replace("&", "_"), nc.replace("&", "_"), nd.replace("&", "_")

    log.info(f"beam_cov_{na_r}x{nb_r}_{nc_r}x{nd_r}")

    np.save(f"{cov_dir}/beam_cov_{na_r}x{nb_r}_{nc_r}x{nd_r}.npy", beam_cov)
