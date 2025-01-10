from pspy import pspy_utils, so_dict, so_spectra
import numpy as np
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
experiments = d["experiments"]
iStart = d["iStart"]
iStop = d["iStop"]
lmax = d["lmax"]

specDir = "spectra"
mcm_dir = "mcms"
mc_dir = "montecarlo"

pspy_utils.create_directory(mc_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

for kind in ["cross", "noise", "auto"]:
    
    vec_list = []
    vec_list_restricted = []

    for iii in range(iStart, iStop):
        vec = []
        vec_restricted = []
        for spec in spectra:
            for id_exp1, exp1 in enumerate(experiments):
                freqs1 = d["freqs_%s" % exp1]
                for id_f1, f1 in enumerate(freqs1):
                    for id_exp2, exp2 in enumerate(experiments):
                        freqs2 = d["freqs_%s"%exp2]
                        for id_f2, f2 in enumerate(freqs2):
                            
                            if  (id_exp1 == id_exp2) & (id_f1 > id_f2) : continue
                            if  (id_exp1 > id_exp2) : continue
                            if (exp1 != exp2) & (kind == "noise"): continue
                            if (exp1 != exp2) & (kind == "auto"): continue

                            spec_name = "%s_%s_%sx%s_%s_%s_%05d" % (type, exp1, f1, exp2, f2, kind, iii)
                            
                            lb, Db = so_spectra.read_ps(specDir + "/%s.dat" % spec_name, spectra=spectra)

                            n_bins = len(lb)
                            vec = np.append(vec, Db[spec])
                            
                            if (exp1 == exp2) & (f1 == f2):
                                if spec == "TT" or spec == "EE" or spec == "TE" :
                                    vec_restricted = np.append(vec_restricted, Db[spec])
                            else:
                                if spec == "TT" or spec == "EE" or spec == "TE" or spec == "ET":
                                    vec_restricted = np.append(vec_restricted, Db[spec])

                                
        vec_list += [vec]
        vec_list_restricted += [vec_restricted]

    mean_vec = np.mean(vec_list, axis=0)
    mean_vec_restricted = np.mean(vec_list_restricted, axis=0)

    cov = 0
    cov_restricted = 0

    for iii in range(iStart, iStop):
        cov += np.outer(vec_list[iii], vec_list[iii])
        cov_restricted += np.outer(vec_list_restricted[iii], vec_list_restricted[iii])

    cov = cov / (iStop-iStart) - np.outer(mean_vec, mean_vec)
    cov_restricted = cov_restricted / (iStop-iStart) - np.outer(mean_vec_restricted, mean_vec_restricted)

    np.save("%s/cov_all_%s.npy" % (mc_dir, kind), cov)
    np.save("%s/cov_restricted_all_%s.npy" % (mc_dir, kind), cov_restricted)

    id_spec = 0
    for spec in spectra:
        for id_exp1, exp1 in enumerate(experiments):
            freqs1 = d["freqs_%s" % exp1]
            for id_f1, f1 in enumerate(freqs1):
                for id_exp2, exp2 in enumerate(experiments):
                    freqs2 = d["freqs_%s" % exp2]
                    for id_f2, f2 in enumerate(freqs2):
                        if  (id_exp1 == id_exp2) & (id_f1 > id_f2) : continue
                        if  (id_exp1 > id_exp2) : continue
                        if (exp1 != exp2) & (kind == "noise"): continue
                        if (exp1 != exp2) & (kind == "auto"): continue
                
                        mean = mean_vec[id_spec * n_bins:(id_spec + 1) * n_bins]
                        std = np.sqrt(cov[id_spec * n_bins:(id_spec + 1) * n_bins, id_spec * n_bins:(id_spec + 1) * n_bins].diagonal())
                        
                        np.savetxt("%s/spectra_%s_%s_%sx%s_%s_%s.dat" % (mc_dir, spec, exp1, f1, exp2, f2, kind), np.array([lb,mean,std]).T)
                                   
                        id_spec += 1

