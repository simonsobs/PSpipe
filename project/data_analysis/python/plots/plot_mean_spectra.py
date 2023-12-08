import matplotlib
matplotlib.use("Agg")
from matplotlib.pyplot import cm
from pspy import pspy_utils, so_dict, so_spectra, so_cov
from pspipe_utils import pspipe_list, external_data
import numpy as np
import pylab as plt
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]
lmax = d["lmax"]

bestfit_dir = "best_fits"
cov_dir = "covariances"
spec_dir = "spectra"
mcm_dir = "mcms"
plot_dir = "plots/spectra/"

pspy_utils.create_directory(plot_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


for scale in ["log", "linear"]:
    for kind in ["cross", "noise", "auto"]:
        for spec in ["TT", "TE", "ET", "EE"]:
        
            spec_list = pspipe_list.get_spec_name_list(d, delimiter="_", kind=kind)
            nspec = len(spec_list)
            color = iter(cm.rainbow(np.linspace(0, 1, nspec + 1)))

            if (scale == "log"):
                if (spec == "TE") or (spec == "ET"): continue
            if (scale == "linear"):
                if (kind == "auto") or (kind == "noise"):
                    if (spec == "TT") or (spec == "EE"): continue
                
            plt.figure(figsize=(12,12))
            count = 0
            
            
            if kind == "cross":
            
                temp = "/Users/thibautlouis/Desktop/projects/so_ps_codes/pspipe_utils/data"
                fp_choi, l_choi, cl_choi, err_choi = external_data.get_choi_data(temp, spec)
                for fp in fp_choi:
                    print(fp)
                    plt.errorbar(l_choi, cl_choi[fp], err_choi[fp], fmt = ".", label=f"choi {fp}")
            
            
            for my_spec in spec_list:
                spec_name = f"{type}_{my_spec}_{kind}"
                lb, Db = so_spectra.read_ps(f"{spec_dir}/{spec_name}.dat", spectra=spectra)
                
                cov = np.load(f"{cov_dir}/analytic_cov_{my_spec}_{my_spec}.npy")
                cov = so_cov.selectblock(cov,
                                        ["TT", "TE", "ET", "EE"],
                                        n_bins = len(lb),
                                        block=spec+spec)
                                                    
                std = np.sqrt(cov.diagonal())

                        
                c=next(color)
                            
                if scale == "log":
                    plt.semilogy()
                if kind == "cross":
                    plt.errorbar(lb + count*10, Db[spec], std, fmt=".", label=f"{spec}_{spec_name}", color=c)
                else:
                    plt.errorbar(lb + count*10, Db[spec], fmt=".", label=f"{spec}_{spec_name}", color=c)
                                
                count +=1
                    
            plt.legend()
            plt.savefig(f"{plot_dir}/{scale}_{spec}_{kind}.png", bbox_inches="tight")
            plt.clf()
            plt.close()
                        
                      
