import matplotlib
matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_spectra, so_cov
from steve_notation import *
import numpy as np
import pylab as plt
import sys, os
import data_analysis_utils
import glob

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]
lmax = d["lmax"]
data_dir = d["data_dir"]
multistep_path = d["multistep_path"]
binning_file = d["binning_file"]

specDir = "spectra"
cov_dir = "covariances"
choi_dir = "multifreq_spectra_dr4.01"
plot_dir = "plots/js/"
pspy_utils.create_directory(plot_dir)

#####
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

os.system("cp %s/multistep2.js %s/multistep2.js" % (multistep_path, plot_dir))
filename = "%s/s17-s19.html" % plot_dir
g = open(filename, mode='w')
g.write('<html>\n')
g.write('<head>\n')
g.write('<title> SO spectra </title>\n')
g.write('<script src="multistep2.js"></script>\n')
g.write('<script> add_step("sub", ["c","v"]) </script> \n')
g.write('<script> add_step("all", ["j","k"]) </script> \n')
g.write('<script> add_step("type", ["a","z"]) </script> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write('<div class=sub> \n')

_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

Cl = {}

for spec in ["TT", "TE", "ET", "EE"]:
    if spec=="TT" or spec=="EE":
        ell, Cl[spec, "f090xf090"], _, Cl[spec,"f090xf150"], _, Cl[spec,"f150xf150"], _ = np.loadtxt("%s/act_dr4.01_multifreq_wide_C_ell_%s.txt" % (choi_dir, spec), unpack=True)
        Cl[spec, "f150xf090"] = Cl[spec,"f090xf150"]
    if spec == "TE" or spec=="ET":
        ell, Cl[spec, "f090xf090"], _, Cl[spec,"f090xf150"], _, Cl[spec,"f150xf090"], _,  Cl[spec,"f150xf150"], _ = np.loadtxt("multifreq_spectra_dr4.01/act_dr4.01_multifreq_wide_C_ell_TE.txt", unpack=True)

    Cl[spec, "f220xf220"] = ell * 0
    Cl[spec, "f090xf220"] = ell * 0
    Cl[spec, "f220xf090"] = ell * 0
    Cl[spec, "f150xf220"] = ell * 0
    Cl[spec, "f220xf150"] = ell * 0


for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):

                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                
                combin = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                print("producing plots for : ", combin)
                spec_name = "%s_%s_%s" % (type, combin, "cross")

                lb, Db = so_spectra.read_ps("%s/%s.dat" % (specDir, spec_name), spectra=spectra)
                cov = np.load("%s/analytic_cov_%s_%s.npy"%(cov_dir, combin, combin))

                for spec in  ["TT", "TE", "TB", "EE", "EB", "BB"]:


                    if spec in ["TT", "TE", "ET", "EE"]:
                        cov_select = so_cov.selectblock(cov,
                                                ["TT", "TE", "ET", "EE"],
                                                n_bins = len(lb),
                                                block=spec+spec)
                        std = np.sqrt(cov_select.diagonal())

                    else:
                        std = None
         

                    if  (sv1 == sv2) & (ar1 == ar2):
                                
                        if spec == "TE":
                            Db["TE"] = (Db["TE"] + Db["ET"])/2
                        elif spec == "TB":
                            Db["TB"] = (Db["TB"] + Db["BT"])/2
                        elif spec == "EB":
                            Db["EB"] = (Db["EB"] + Db["BE"])/2

                    str = "%s_%s_cross.png" % (spec, combin)
                    
                    
                    _, f1 = ar1.split("_")
                    _, f2 = ar2.split("_")

                    plt.figure(figsize=(12,12))
                    if spec == "TT":
                        plt.semilogy()
                        
                    if spec in ["TT", "TE", "ET", "EE"]:
                        plt.plot(ell, Cl[spec,"%sx%s"%(f1,f2)]*ell**2/(2*np.pi), label="Choi wide")
                    plt.errorbar(lb, Db[spec], std, fmt=".", label=combin)
                    plt.legend()
                    plt.title(r"$D^{%s}_{\ell}$" % (spec), fontsize=20)
                    plt.xlabel(r"$\ell$", fontsize=20)
                    plt.savefig("%s/%s" % (plot_dir,str), bbox_inches="tight")
                    plt.clf()
                    plt.close()

                    g.write('<div class=type>\n')
                    g.write('<img src="' + str + '" width="50%" /> \n')
                    g.write('</div>\n')

g.write('</div> \n')
g.write('</body> \n')
g.write('</html> \n')
g.close()
