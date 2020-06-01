"""
This script compute best fit from theory and fg power spectra.
In our particular case, we use best fit foregrounds from erminia.
"""
import numpy as np, pylab as plt
from pspy import so_dict, pspy_utils
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
fg_array = np.loadtxt(d["fgfile"])

bestfit_dir = "best_fits"
plot_dir = "plots/best_fits/"

pspy_utils.create_directory(bestfit_dir)
pspy_utils.create_directory(plot_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

clth = {}
lth, clth["TT"], clth["EE"], clth["BB"], clth["TE"] = np.loadtxt(d["theoryfile"], unpack=True)

combin = ["TT_90x90", "TT_90x150", "TT_150x150",
          "TE_90x90", "TE_90x150",  "TE_150x150",
          "EE_90x90", "EE_90x150", "EE_150x150"]
        

fg_dict = {}
for c1, s1 in enumerate(combin):
    fg_dict[s1] = fg_array[:,c1+1]

fg_dict["TT_150x90"] =  fg_dict["TT_90x150"]
fg_dict["TE_150x90"] =  fg_dict["TE_90x150"]
fg_dict["EE_150x90"] =  fg_dict["EE_90x150"]

l_size = np.minimum(len(clth["TT"]),len(fg_dict["TT_90x90"]))
lth = lth[:l_size]


for spec in ["TT", "EE", "TE"]:
    plt.figure(figsize=(12,12))
    for id_sv1, sv1 in enumerate(surveys):
        arrays_1 = d["arrays_%s" % sv1]
        for id_ar1, ar1 in enumerate(arrays_1):
            nu_eff1 = d["nu_eff_%s_%s" % (sv1, ar1)]
            for id_sv2, sv2 in enumerate(surveys):
                arrays_2 = d["arrays_%s" % sv2]
                for id_ar2, ar2 in enumerate(arrays_2):
                    nu_eff2 = d["nu_eff_%s_%s" % (sv2, ar2)]
                    
                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue

                    if spec == "TT" or spec == "EE": 
                        plt.semilogy()
                    
                    name = "%s_%sx%s_%s_%s" % (sv1, ar1, sv2, ar2, spec)
                    print(name, nu_eff1, nu_eff2)
                
                    cl_th_and_fg = clth[spec][:l_size] + fg_dict["%s_%sx%s" % (spec, nu_eff1, nu_eff2)][:l_size]
                    
                    np.savetxt("%s/fg_%sx%s.dat" % (bestfit_dir, nu_eff1, nu_eff2),
                                np.transpose([lth, fg_dict["%s_%sx%s" % (spec, nu_eff1, nu_eff2)][:l_size]]))
                                
                    np.savetxt("%s/best_fit_%s.dat" % (bestfit_dir, name),
                                np.transpose([lth, cl_th_and_fg]))
        
                    plt.plot(lth, cl_th_and_fg, label="%s" % name)
        
    plt.legend()
    plt.savefig("%s/best_fit_%s.png" % (plot_dir, spec))
    plt.clf()
    plt.close()
