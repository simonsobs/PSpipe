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
planck_data_dir = d["planck_data_dir"]


cov_dir = "covariances"
specDir = "spectra"
plot_dir = "plots/spectra/"

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

                combin = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                spec_name = "%s_%s_%s" % (type, combin, kind)

                lb, Db = so_spectra.read_ps("%s/%s.dat" % (specDir, spec_name), spectra=spectra)
                    
                cov = np.load("%s/analytic_cov_%s_%s.npy"%(cov_dir, combin, combin))
                cov = so_cov.selectblock(cov,
                                        ["TT", "TE", "ET", "EE"],
                                        n_bins = len(lb),
                                        block="TT")
                                                    
                std = np.sqrt(cov.diagonal())

                lp, clp, errorp = np.loadtxt(planck_data_dir + "spectrum_TT_100x100.dat")
                
                plt.errorbar(lb, Db[spec], std, fmt=".", label="%s_%s" % (spec, combin), color=c)
                 
