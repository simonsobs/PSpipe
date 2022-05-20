"""
This script generate a montecarlo simulation of beam uncertainties, it then compares the result with
the analytic beam covariance computed by get_multifrequency_covmat.py
"""


from pspy import pspy_utils, so_dict, so_cov
import numpy as np
import sys, os
import data_analysis_utils


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]

bestfit_dir = "best_fits"
cov_dir = "covariances"
cov_plot_dir = "plots/full_beam_covariance"
multistep_path = d["multistep_path"]

lmax = d["lmax"]
binning_file = d["binning_file"]

pspy_utils.create_directory(cov_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

spec_list = []
id_list = []
bl = {}
sqrt_beam_cov = {}
ps_th = {}

# read the data needed for the simulations

for id_sv1, sv1 in enumerate(surveys):
    for id_ar1, ar1 in enumerate(d["arrays_%s" % sv1]):
        print(sv1, ar1)
    
        data = np.loadtxt(d["beam_%s_%s" % (sv1, ar1)])
        _, bl[sv1, ar1], error_modes  = data[2: lmax + 2, 0], data[2: lmax + 2, 1], data[2: lmax + 2, 2:]
        
        #sqrt_beam_cov[sv1, ar1] = np.ones((lmax, lmax))
        
        beam_cov =  error_modes.dot(error_modes.T)
        eig_values, eig_vectors = np.linalg.eig(beam_cov)
        
        #the abs is to regularize small eigeinvalues
        sqrt_beam_cov[sv1, ar1] = (eig_vectors @ np.diag(np.sqrt(np.abs(eig_values))) @ eig_vectors.T).real

        freq1 = d["nu_eff_%s_%s" % (sv1, ar1)]

        for id_sv2, sv2 in enumerate(surveys):
            for id_ar2, ar2 in enumerate(d["arrays_%s" % sv2]):
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                
                freq2 = d["nu_eff_%s_%s" % (sv2, ar2)]
                
                spec = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                id = [sv1, ar1, sv2, ar2]
                
                for field in ["TT", "TE", "ET", "EE"]:

                    name = "%sx%s_%s" % (freq1, freq2, field)

                    l_th, ps_th[spec, field] = np.loadtxt("%s/best_fit_%s.dat" % (bestfit_dir, name), unpack=True)
                    ps_th[spec, field] = ps_th[spec, field][:lmax]

                spec_list += [spec]
                id_list += [id]

spec_name_list = []
n_sims = 1000
mc_cov = 0
mean = 0
for iii in range(n_sims):
    print(iii)
    bl_sim = {}
    for sv in surveys:
        arrays = d["arrays_%s" % sv]
        for ar in arrays:
            bl_sim[sv, ar] = bl[sv, ar] + np.dot(sqrt_beam_cov[sv, ar], np.random.randn(lmax))

    vec_restricted = []
    for field in ["TT", "TE", "ET", "EE"]:
        for id, spec in zip(id_list, spec_list):
        
            sv1, ar1, sv2, ar2 = id
            Dl_th = ps_th[spec, field] *  bl[sv1, ar1] *  bl[sv2, ar2] / (bl_sim[sv1, ar1] *  bl_sim[sv2, ar2])
            lb, Db = pspy_utils.naive_binning(l_th, Dl_th, binning_file, lmax)
            
            if (sv1 == sv2) & (ar1 == ar2):
                if field == "TT" or field == "EE" or field == "TE":
                    vec_restricted = np.append(vec_restricted, Db)
                    spec_name_list += ["%s_%s_%sx%s_%s" % (field, sv1, ar1, sv2, ar2)]

            else:
                if field == "TT" or field == "EE" or field == "TE" or field == "ET":
                    vec_restricted = np.append(vec_restricted, Db)
                    spec_name_list += ["%s_%s_%sx%s_%s" % (field, sv1, ar1, sv2, ar2)]

            
    mean += vec_restricted
    mc_cov += np.outer(vec_restricted, vec_restricted)

mean /= n_sims
mc_cov = mc_cov / n_sims - np.outer(mean, mean)

np.save("%s/truncated_mc_beam_cov.npy" % cov_dir, mc_cov)
analytic_cov = np.load("%s/truncated_analytic_beam_cov.npy" % cov_dir)


data_analysis_utils.interactive_covariance_comparison(analytic_cov, mc_cov, spec_name_list, binning_file, lmax, cov_plot_dir, multistep_path, corr_range=1)



