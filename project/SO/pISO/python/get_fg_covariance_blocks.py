import sys

import numpy as np
from pspy import pspy_utils, so_dict, so_mpi, so_cov
from pspipe_utils import pspipe_list, best_fits, log

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

cov_dir = d["cov_dir"]
bestfit_dir = d["best_fits_dir"] 

pspy_utils.create_directory(cov_dir)
surveys = d["surveys"]
arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
binning_file = d["binning_file"]
lmax = d["lmax"]
spectra = ["TT", "TE", "ET", "EE"]
chain_filename = d["p_act_chain_filename"]

array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]

# list of components to be read
component_list_tt = ["tt_tSZ", "tt_kSZ", "tt_cibp", "tt_cibc", "tt_dust",  "tt_radio", "tt_tSZxCIB"]
component_list_te = ["te_dust", "te_radio"]
component_list_ee = ["ee_dust",  "ee_radio"]



fg_dict = {}
for ic, comp in enumerate(component_list_tt):
    lth, fg_dict[comp] = best_fits.fg_dict_from_files(bestfit_dir + f"/components/{comp}"+"_{}x{}.dat",
                                                    array_list,
                                                    lmax,
                                                    ["TT"],
                                                    f_name_cmb= None)

for ic, comp in enumerate(component_list_te):
    # using spec = "TT" to avoid an error when calling fg_dict_from_files, when using TE it 
    # also tries to read ET, which we don't have here (it's the same as te). 
    # this anyway only sets the key of the dictionary fg_dict["te_..."], fixing it later
    lth, fg_dict[comp] = best_fits.fg_dict_from_files(bestfit_dir + f"/components/{comp}"+"_{}x{}.dat",
                                                    array_list,
                                                    lmax,
                                                    ["TT"],
                                                    f_name_cmb= None)
for ic, comp in enumerate(component_list_ee):
    lth, fg_dict[comp] = best_fits.fg_dict_from_files(bestfit_dir + f"/components/{comp}"+"_{}x{}.dat",
                                                    array_list,
                                                    lmax,
                                                    ["EE"],
                                                    f_name_cmb= None)


ps_all = {}

log.info(f"construct best fit for all cross array spectra")

for id_sv1, sv1 in enumerate(surveys):
    for id_ar1, ar1 in enumerate(d[f"arrays_{sv1}"]):

        for id_sv2, sv2 in enumerate(surveys):
            for id_ar2, ar2 in enumerate(d[f"arrays_{sv2}"]):

                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue


                for comp in (component_list_tt + component_list_ee):
                    spec = f"{comp[:2].upper()}"

                    ps_all[f"{sv1}&{ar1}", f"{sv2}&{ar2}", comp, spec] = fg_dict[comp][f"{sv1}_{ar1}", f"{sv2}_{ar2}"][spec]
                    ps_all[f"{sv2}&{ar2}", f"{sv1}&{ar1}", comp, spec] = fg_dict[comp][f"{sv2}_{ar2}", f"{sv1}_{ar1}"][spec]

                for comp in component_list_te:
                    # fixing the dictionary label to TE here
                    spec_new = "TE"
                    spec = "TT"
                    
                    ps_all[f"{sv1}&{ar1}", f"{sv2}&{ar2}", comp, spec_new] = fg_dict[comp][f"{sv1}_{ar1}", f"{sv2}_{ar2}"][spec]
                    ps_all[f"{sv2}&{ar2}", f"{sv1}&{ar1}", comp, spec_new] = fg_dict[comp][f"{sv2}_{ar2}", f"{sv1}_{ar1}"][spec]


                        
log.info(f"construct block fg covariance")

ncovs, na_list, nb_list, nc_list, nd_list = pspipe_list.get_covariances_list(d)

fg_par_list = ['a_tSZ','a_kSZ','a_p','a_c','a_gtt','a_s','xi', 'a_gte','a_pste', 'a_gee','a_psee']
fg_params_covmat = pspy_utils.fg_params_covmat(chain_filename, fg_par_list)


so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=ncovs - 1)
print(subtasks)

bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
nbins = len(bin_hi)
for task in subtasks:
    task = int(task)
    na, nb, nc, nd = na_list[task], nb_list[task], nc_list[task], nd_list[task]
    id_element = [na, nb, nc, nd]

    nspec = len(spectra)
    fg_cov = np.zeros((nspec * nbins, nspec * nbins))   
    
    fg_list_ab_tt = np.zeros((len(component_list_tt), lmax - 2))
    fg_list_cd_tt = np.zeros((len(component_list_tt), lmax - 2))

    fg_list_ab_te = np.zeros((len(component_list_te), lmax - 2))
    fg_list_cd_te = np.zeros((len(component_list_te), lmax - 2))

    fg_list_ba_te = np.zeros((len(component_list_te), lmax - 2))
    fg_list_dc_te = np.zeros((len(component_list_te), lmax - 2))
    
    fg_list_ab_ee = np.zeros((len(component_list_ee), lmax - 2))
    fg_list_cd_ee = np.zeros((len(component_list_ee), lmax - 2))

    for ic,comp in enumerate(component_list_tt):
        fg_list_ab_tt[ic] = ps_all[na, nb, comp, "TT"]
        fg_list_cd_tt[ic] = ps_all[nc, nd, comp, "TT"]
        
    for ic,comp in enumerate(component_list_te):
        fg_list_ab_te[ic] = ps_all[na, nb, comp, "TE"]
        fg_list_cd_te[ic] = ps_all[nc, nd, comp, "TE"]
        # to take into account the ET spectra we flip the arrays
        fg_list_ba_te[ic] = ps_all[nb, na, comp, "TE"]
        fg_list_dc_te[ic] = ps_all[nd, nc, comp, "TE"]

    for ic,comp in enumerate(component_list_ee):
        fg_list_ab_ee[ic] = ps_all[na, nb, comp, "EE"]
        fg_list_cd_ee[ic] = ps_all[nc, nd, comp, "EE"]
    
    for i, spec1 in enumerate(spectra):
        for j, spec2 in enumerate(spectra):

            if spec1 == "TT":
                ab = fg_list_ab_tt
                idxab = np.arange(len(component_list_tt))
                
            if spec2 == "TT":
                cd = fg_list_cd_tt
                idxcd = np.arange(len(component_list_tt))
                
            if spec1 == "TE" or spec1 == "ET":
                if spec1 == "TE":
                    ab = fg_list_ab_te
                if spec1 == "ET":
                    ab = fg_list_ba_te
                # te or et block in fg covmat is the same
                idxab = np.arange(len(component_list_tt), len(component_list_tt) + len(component_list_te))
                
            if spec2 == "TE" or spec2 == "ET":
                if spec2 == "TE":
                    cd = fg_list_cd_te
                if spec2 == "ET":
                    cd = fg_list_dc_te
                
                idxcd = np.arange(len(component_list_tt), len(component_list_tt) + len(component_list_te))

            if spec1 == "EE":
                ab = fg_list_ab_ee
                idxab = np.arange(len(component_list_tt) + len(component_list_te),
                                 len(component_list_tt) + len(component_list_te) + len(component_list_ee))
                
            if spec2 == "EE":
                cd = fg_list_cd_ee
                idxcd = np.arange(len(component_list_tt) + len(component_list_te),
                                 len(component_list_tt) + len(component_list_te) + len(component_list_ee))

                
            
            fg_params_covmat_block = fg_params_covmat[np.ix_(idxab, idxcd)]
            fg_cov_unbinned = ab.T @ fg_params_covmat_block @ cd 
                
            fg_cov[i * nbins: (i + 1) * nbins, j * nbins: (j + 1) * nbins] = so_cov.bin_mat(fg_cov_unbinned, binning_file, lmax)

    na_r, nb_r, nc_r, nd_r = na.replace("&", "_"), nb.replace("&", "_"), nc.replace("&", "_"), nd.replace("&", "_")

    log.info(f"fg_marginalization_cov_{na_r}x{nb_r}_{nc_r}x{nd_r}")

    np.save(f"{cov_dir}/fg_marginalization_cov_{na_r}x{nb_r}_{nc_r}x{nd_r}.npy", fg_cov)            
