import matplotlib
matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_spectra, so_cov
import numpy as np
import pylab as plt
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]
lmax = d["lmax"]
data_dir = d["data_dir"]
multistep_path = d["multistep_path"]

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


Cl = {}

for spec in ["TT", "TE", "ET", "EE"]:
    if spec=="TT" or spec=="EE":
        ell, Cl[spec, "f090xf090"], _, Cl[spec,"f090xf150"], _, Cl[spec,"f150xf150"], _ = np.loadtxt("%s/act_dr4.01_multifreq_wide_C_ell_%s.txt" % (choi_dir, spec), unpack=True)
        Cl[spec, "f150xf090"] = Cl[spec,"f090xf150"]
    if spec == "TE" or spec=="ET":
        ell, Cl[spec, "f090xf090"], _, Cl[spec,"f090xf150"], _, Cl[spec,"f150xf090"], _,  Cl[spec,"f150xf150"], _ = np.loadtxt("multifreq_spectra_dr4.01/act_dr4.01_multifreq_wide_C_ell_TE.txt", unpack=True)

    Cl[spec, "f220xf220"] = Cl[spec,"f150xf150"]
    Cl[spec, "f090xf220"] = Cl[spec,"f150xf150"]
    Cl[spec, "f220xf090"] = Cl[spec,"f150xf150"]
    Cl[spec, "f150xf220"] = Cl[spec,"f150xf150"]
    Cl[spec, "f220xf150"] = Cl[spec,"f150xf150"]

ylim ={}

ylim["TT"] = [10,10**4]
ylim["TE"] = [-180,180]
ylim["ET"] = [-180,180]

ylim["EE"] = [-30,100]


arrays = d["arrays_s17"]

for spec in  ["TT", "ET", "TE", "EE"]:
    for ar in arrays:
        plt.figure(figsize=(16,12))

        for id_sv1, sv1 in enumerate(surveys):
            for id_sv2, sv2 in enumerate(surveys):
        
                if  (id_sv1 > id_sv2) : continue
                combin = "%s_%sx%s_%s" % (sv1, ar, sv2, ar)
                print("producing plots for : ", combin)
                spec_name = "%s_%s_%s" % (type, combin, "cross")

                lb, Db = so_spectra.read_ps("%s/%s.dat" % (specDir, spec_name), spectra=spectra)
                cov = np.load("%s/analytic_cov_%s_%s.npy"%(cov_dir, combin, combin))

                cov_select = so_cov.selectblock(cov,
                                                ["TT", "TE", "ET", "EE"],
                                                n_bins = len(lb),
                                                block=spec+spec)
                std = np.sqrt(cov_select.diagonal())
    
                if  (sv1 == sv2):
                    if spec == "TE":
                        Db["TE"] = (Db["TE"] + Db["ET"])/2

 
                _, f = ar.split("_")
            
                f_choi = f
                if f == "f220": f_choi = "f150"
                    
                if spec == "TT":
                    plt.semilogy()
            
                plt.errorbar(lb, Db[spec], std, fmt=".", label=combin)
        
        plt.plot(ell, Cl[spec,"%sx%s"%(f, f)] * ell**2 / (2*np.pi), color="grey", label="Choi wide %sx%s"%(f_choi, f_choi))
        plt.ylim(ylim[spec][0], ylim[spec][1])
        plt.legend(fontsize=18)
        plt.title(r"$D^{%s}_{\ell}$" % (spec), fontsize=20)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.savefig("%s/all_%s_%s" % (plot_dir, ar, spec), bbox_inches="tight")
        plt.clf()
        plt.close()

for spec in  ["TT", "ET", "TE", "EE"]:
    for spec_type in ["noise", "cross"]:
        for id_sv, sv in enumerate(surveys):
            arrays = d["arrays_%s" % sv]
            plt.figure(figsize=(16,12))
            for id_ar, ar in enumerate(arrays):
                combin = "%s_%sx%s_%s" % (sv, ar, sv, ar)
                print("producing plots for : ", combin)
                spec_name = "%s_%s_%s" % (type, combin, spec_type)

    

                lb, Db = so_spectra.read_ps("%s/%s.dat" % (specDir, spec_name), spectra=spectra)
                if spec_type == "cross":
                    cov = np.load("%s/analytic_cov_%s_%s.npy"%(cov_dir, combin, combin))
                    cov_select = so_cov.selectblock(cov,
                                                    ["TT", "TE", "ET", "EE"],
                                                    n_bins = len(lb),
                                                    block=spec+spec)
                    std = np.sqrt(cov_select.diagonal())
                else:
                    std = None
                    
                    
                if spec == "TE":
                    Db["TE"] = (Db["TE"] + Db["ET"])/2
        
                _, f1 = ar1.split("_")
                _, f2 = ar2.split("_")
                    
                f1_choi = f1
                f2_choi = f2
                if f1 == "f220":
                    f1_choi = "f150"
                if f2 == "f220":
                    f2_choi = "f150"

                if spec == "TT":
                    plt.semilogy()
        
                plt.errorbar(lb, Db[spec], yerr=std, fmt=".", label=combin)
    
                plt.plot(ell, Cl[spec,"%sx%s"%(f1, f2)] * ell**2 / (2*np.pi), color="grey")
                    
        plt.ylim(ylim[spec][0], ylim[spec][1])
        plt.legend(fontsize=18)
        plt.title(r"$D^{%s}_{\ell}$" % (spec), fontsize=20)
        plt.xlabel(r"$\ell$", fontsize=20)
        plt.savefig("%s/all_%s_%s_%s" % (plot_dir, sv, spec, spec_type), bbox_inches="tight")
        plt.clf()
        plt.close()
    
    


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

                for spec in  ["TT", "ET", "TE", "EE"]:


                    cov_select = so_cov.selectblock(cov,
                                                    ["TT", "TE", "ET", "EE"],
                                                    n_bins = len(lb),
                                                    block=spec+spec)
                    std = np.sqrt(cov_select.diagonal())

         

                    if  (sv1 == sv2) & (ar1 == ar2):
                                
                        if spec == "TE":
                            Db["TE"] = (Db["TE"] + Db["ET"])/2

                    str = "%s_%s_cross.png" % (spec, combin)
                    
                    
                    _, f1 = ar1.split("_")
                    _, f2 = ar2.split("_")
                    
                    f1_choi = f1
                    f2_choi = f2
                    if f1 == "f220":
                        f1_choi = "f150"
                    if f2 == "f220":
                        f2_choi = "f150"


                    plt.figure(figsize=(16,12))
                    if spec == "TT":
                        plt.semilogy()
                    
                    plt.plot(ell, Cl[spec,"%sx%s"%(f1,f2)]*ell**2/(2*np.pi), label="Choi wide %sx%s"%(f1_choi,f2_choi))
                    plt.errorbar(lb, Db[spec], std, fmt=".", label=combin)
                    plt.legend(fontsize=24)
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



