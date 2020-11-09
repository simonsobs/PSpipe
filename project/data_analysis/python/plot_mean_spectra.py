import matplotlib
matplotlib.use("Agg")
from matplotlib.pyplot import cm
from pspy import pspy_utils, so_dict, so_spectra, so_cov
import numpy as np
import pylab as plt
import sys
import data_analysis_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]
lmax = d["lmax"]

bestfit_dir = "best_fits"
cov_dir = "covariances"
specDir = "spectra"
mcm_dir = "mcms"
plot_dir = "plots/spectra/"

pspy_utils.create_directory(plot_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
n_spectra = data_analysis_utils.get_nspec(d)


for scale in ["log", "linear"]:
    for kind in ["cross", "noise", "auto"]:
        for spec in ["TT", "TE", "ET", "EE"]:
            color = iter(cm.rainbow(np.linspace(0,1,n_spectra[kind]+1)))

            if (scale == "log"):
                if (spec == "TE") or (spec == "ET"): continue
            if (scale == "linear"):
                if (kind == "auto") or (kind == "noise"):
                    if (spec == "TT") or (spec == "EE"): continue
                
            plt.figure(figsize=(12,12))
            count = 0
            
            for id_sv1, sv1 in enumerate(surveys):
                arrays_1 = d["arrays_%s" % sv1]
                for id_ar1, ar1 in enumerate(arrays_1):
                    freq1 = d["nu_eff_%s_%s" % (sv1, ar1)]

                    for id_sv2, sv2 in enumerate(surveys):
                        arrays_2 = d["arrays_%s" % sv2]
                        for id_ar2, ar2 in enumerate(arrays_2):
                            freq2 = d["nu_eff_%s_%s" % (sv2, ar2)]

                            if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                            if  (id_sv1 > id_sv2) : continue
                            if (sv1 != sv2) & (kind == "noise"): continue
                            if (sv1 != sv2) & (kind == "auto"): continue
                                
                            print (scale, spec)

                            combin = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                            spec_name = "%s_%s_%s" % (type, combin, kind)

                            lb, Db = so_spectra.read_ps("%s/%s.dat" % (specDir, spec_name), spectra=spectra)
                    
                            cov = np.load("%s/analytic_cov_%s_%s.npy"%(cov_dir, combin, combin))
                            cov = so_cov.selectblock(cov,
                                                    ["TT", "TE", "ET", "EE"],
                                                    n_bins = len(lb),
                                                    block=spec+spec)
                                                    
                            std = np.sqrt(cov.diagonal())

                            
                            lth, bfth = np.loadtxt("%s/best_fit_%sx%s_%s.dat"%(bestfit_dir, freq1, freq2, spec), unpack=True)

                            c=next(color)
                            
                            if scale == "log":
                                plt.semilogy()
                            if kind == "cross":
                                plt.errorbar(lb + count*10, Db[spec], std, fmt=".", label="%s_%s" % (spec, combin), color=c)
                            else:
                                plt.errorbar(lb + count*10, Db[spec], fmt=".", label="%s_%s" % (spec, combin), color=c)
                                
                            plt.plot(lth[:lmax], bfth[:lmax])


                            count +=1
                    
            plt.legend()
            if kind == "cross":
                range = d["range_%s"%spec]
                if (kind == "log") & (range[0]<0):
                    range[0] = 10*-3
                plt.ylim(range[0],range[1])
                
            plt.savefig("%s/%s_%s_%s.png" % (plot_dir, scale, spec, spec_name), bbox_inches="tight")
            plt.clf()
            plt.close()
                        
                      
