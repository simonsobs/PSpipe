"""
This script compute the analytical beam covariance matrix elements.
"""
import sys

import data_analysis_utils
import numpy as np
from pspy import pspy_utils, so_dict, so_mpi

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
bestfit_dir = "best_fits"

pspy_utils.create_directory(cov_dir)
surveys = d["surveys"]
binning_file = d["binning_file"]
lmax = d["lmax"]

ps_all = {}
norm_beam_cov = {}

spec_name = []

_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

for id_sv1, sv1 in enumerate(surveys):
    for id_ar1, ar1 in enumerate(d["arrays_%s" % sv1]):
    
        data = np.loadtxt(d["beam_%s_%s" % (sv1, ar1)])
        
        _, bl, error_modes  = data[2: lmax + 2, 0], data[2: lmax + 2, 1], data[2: lmax + 2, 2:]
        beam_cov =  error_modes.dot(error_modes.T)
        
        norm_beam_cov[sv1, ar1] = beam_cov / np.outer(bl, bl)
        
        freq1 = d["nu_eff_%s_%s" % (sv1, ar1)]

        for id_sv2, sv2 in enumerate(surveys):
            for id_ar2, ar2 in enumerate(d["arrays_%s" % sv2]):
            
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                
                freq2 = d["nu_eff_%s_%s" % (sv2, ar2)]

                
                for spec in ["TT", "TE", "ET", "EE"]:
                        
                    name = "%sx%s_%s" % (freq1, freq2, spec)
                    _, ps_th = np.loadtxt("%s/best_fit_%s.dat"%(bestfit_dir, name), unpack=True)

                    ps_all["%s&%s" % (sv1, ar1), "%s&%s" % (sv2, ar2), spec] = ps_th[:lmax]
                    ps_all["%s&%s" % (sv2, ar2), "%s&%s" % (sv1, ar1), spec] = ps_th[:lmax]

                spec_name += ["%s&%sx%s&%s" % (sv1, ar1, sv2, ar2)]

# prepare the mpi computation
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

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=ncovs - 1)
print(subtasks)
for task in subtasks:
    task = int(task)
    na, nb, nc, nd = na_list[task], nb_list[task], nc_list[task], nd_list[task]
    id_element = [na, nb, nc, nd]
        
    analytic_beam_cov = data_analysis_utils.covariance_element_beam(id_element, ps_all, norm_beam_cov, binning_file, lmax)
        
    na_r, nb_r, nc_r, nd_r = na.replace("&", "_"), nb.replace("&", "_"), nc.replace("&", "_"), nd.replace("&", "_")
       
    np.save("%s/analytic_beam_cov_%sx%s_%sx%s.npy" % (cov_dir, na_r, nb_r, nc_r, nd_r), analytic_beam_cov)
