import matplotlib
matplotlib.use("Agg")

from pspy import pspy_utils, so_dict, so_spectra, so_cov
import numpy as np
import pylab as plt
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]
lmax = d["lmax"]

bestfit_dir = "best_fits"
specDir = "spectra"
plot_dir = "plots/all_spectra/"

pspy_utils.create_directory(plot_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


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
                        
                nsplits_1 = len(d["maps_%s_%s" % (sv1, ar1)])
                nsplits_2 = len(d["maps_%s_%s" % (sv2, ar2)])
                
                name="%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)

                for kind in ["auto", "cross"]:
                    for spec in ["TT", "TE", "ET", "EE"]:
                        for scale in ["log", "linear"]:
                        
                            if (scale == "log"):
                                if (spec == "TE") or (spec == "ET"): continue
                            if (scale == "linear"):
                                if (kind == "auto"):
                                    if (spec == "TT") or (spec == "EE"): continue
                            if (kind == "auto"):
                                if (id_sv1 != id_sv2): continue

                            plt.figure(figsize=(12, 12))
                            if scale == "log":
                                plt.semilogy()

                            lth, bfth = np.loadtxt("%s/best_fit_%sx%s_%s.dat"%(bestfit_dir, freq1, freq2, spec), unpack=True)

                            #plt.plot(lth[:lmax], bfth[:lmax],color="grey")


                            for s1 in range(nsplits_1):
                                for s2 in range(nsplits_2):
                                    if (sv1 == sv2) & (ar1 == ar2) & (s1 > s2) : continue

                                    if kind == "cross":
                                        if (s1 == s2) & (sv1 == sv2): continue
                                    elif kind == "auto":
                                        if (s1 != s2) or (sv1 != sv2): continue

                                    spec_name="%s_%s_%d%d" % (type, name, s1, s2)
                                    lb, Db = so_spectra.read_ps("%s/%s.dat" % (specDir, spec_name), spectra=spectra)
                                                    
                                    plt.errorbar(lb, Db[spec], fmt=".", label="%s" % (spec_name))
                                    
                    
                            plt.legend()
                            plt.savefig("%s/%s_%s_%s_%s.png" % (plot_dir, scale, kind, name, spec), bbox_inches="tight")
                            plt.clf()
                            plt.close()
                    
                      
