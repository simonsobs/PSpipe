"""
This script compute the analytical covariance matrix elements.
"""
from pspy import pspy_utils, so_dict, so_map, so_mpi, so_mcm, so_spectra, so_cov
import numpy as np
import data_analysis_utils
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

windows_dir = "windows"
mcms_dir = "mcms"
spectra_dir = "spectra"
ps_model_dir = "noise_model"
cov_dir = "covariances"
bestfit_dir = "best_fits"

pspy_utils.create_directory(cov_dir)

surveys = d["surveys"]
binning_file = d["binning_file"]
lmax = d["lmax"]
niter = d["niter"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

ps_all = {}
nl_all = {}
spec_name = []
ns = {}

for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    
    for id_ar1, ar1 in enumerate(arrays_1):
        _, bl1 = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv1, ar1)])
        bl1 = bl1[2:lmax + 2]
        
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            
            for id_ar2, ar2 in enumerate(arrays_2):
                _, bl2 = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv2, ar2)])
                bl2 = bl2[2:lmax + 2]

                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue

                if (sv1 == sv2) & (ar1 == ar2):
                    spec_name_noise = "mean_%sx%s_%s_noise" % (ar1, ar2, sv1)
                    _, Nl = so_spectra.read_ps(ps_model_dir + "/%s.dat" % spec_name_noise, spectra=spectra)
                
                for spec in ["TT", "TE", "ET", "EE"]:
                    name = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
            
                    if spec == "ET":
                        _, ps_th = np.loadtxt("%s/best_fit_%s_%s.dat"%(bestfit_dir, name, "TE"), unpack=True)
                    else:
                        _, ps_th = np.loadtxt("%s/best_fit_%s_%s.dat"%(bestfit_dir, name, spec), unpack=True)

            
                    ps_all["%s&%s" % (sv1, ar1), "%s&%s" % (sv2, ar2), spec] = bl1 * bl2 * ps_th[:lmax]
                    
                    if (sv1 == sv2) & (ar1 == ar2):
                        ns[sv1] = len(d["maps_%s_%s" % (sv1, ar1)])

                        nl_all["%s&%s" % (sv1, ar1), "%s&%s" % (sv2, ar2), spec] = Nl[spec][:lmax] * ns[sv1]
                    else:
                        nl_all["%s&%s" % (sv1, ar1), "%s&%s" % (sv2, ar2), spec] = np.zeros(lmax)
                    
                    ps_all["%s&%s" % (sv2, ar2), "%s&%s"%(sv1, ar1), spec] = ps_all["%s&%s"%(sv1, ar1), "%s&%s"%(sv2, ar2), spec]
                    nl_all["%s&%s" % (sv2, ar2), "%s&%s"%(sv1, ar1), spec] = nl_all["%s&%s"%(sv1, ar1), "%s&%s"%(sv2, ar2), spec]
                
                spec_name += ["%s&%sx%s&%s" % (sv1, ar1, sv2, ar2)]

na_list, nb_list, nc_list, nd_list = [], [], [], []
ncovs = 0

for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1 > sid2: continue
        na, nb = spec1.split("x")
        nc, nd = spec2.split("x")
        na_list += [na]
        nb_list += [nb]
        nc_list += [nc]
        nd_list += [nd]
        ncovs += 1

nspecs=len(spec_name)

print("number of covariance matrices to compute : %s" % ncovs)
#so_mpi.init(True)
#subtasks = so_mpi.taskrange(imin=0, imax=ncovs - 1)
#print(subtasks)
for task in range(55):#subtasks:
    task = int(task)
    
    na, nb, nc, nd = na_list[task], nb_list[task], nc_list[task], nd_list[task]
    na_r, nb_r, nc_r, nd_r = na.replace("&", "_"), nb.replace("&", "_"), nc.replace("&", "_"), nd.replace("&", "_")
    print("cov element (%s x %s, %s x %s)" % (na_r, nb_r, nc_r, nd_r))
    win = {}
    win["Ta"] = so_map.read_map("%s/window_T_%s.fits"%(windows_dir, na_r))
    win["Tb"] = so_map.read_map("%s/window_T_%s.fits"%(windows_dir, nb_r))
    win["Tc"] = so_map.read_map("%s/window_T_%s.fits"%(windows_dir, nc_r))
    win["Td"] = so_map.read_map("%s/window_T_%s.fits"%(windows_dir, nd_r))
    win["Pa"] = so_map.read_map("%s/window_pol_%s.fits"%(windows_dir, na_r))
    win["Pb"] = so_map.read_map("%s/window_pol_%s.fits"%(windows_dir, nb_r))
    win["Pc"] = so_map.read_map("%s/window_pol_%s.fits"%(windows_dir, nc_r))
    win["Pd"] = so_map.read_map("%s/window_pol_%s.fits"%(windows_dir, nd_r))

    coupling = so_cov.cov_coupling_spin0and2_simple(win, lmax, niter=niter)
    
    try:
        mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix="%s/%sx%s" % (mcms_dir, na_r, nb_r), spin_pairs=spin_pairs)
    except:
        mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix="%s/%sx%s" % (mcms_dir, nb_r, na_r), spin_pairs=spin_pairs)

    try:
        mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix="%s/%sx%s" % (mcms_dir, nc_r, nd_r), spin_pairs=spin_pairs)
    except:
        mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix="%s/%sx%s" % (mcms_dir, nd_r, nc_r), spin_pairs=spin_pairs)


    analytic_cov = data_analysis_utils.covariance_element(coupling,
                                                          [na, nb, nc, nd],
                                                          ns,
                                                          ps_all,
                                                          nl_all,
                                                          binning_file,
                                                          mbb_inv_ab,
                                                          mbb_inv_cd)
                                                          
    np.save("%s/analytic_cov_%sx%s_%sx%s.npy" % (cov_dir, na_r, nb_r, nc_r, nd_r), analytic_cov)


