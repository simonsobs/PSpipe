import matplotlib
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
multistep_path = d["multistep_path"]
binning_file = d["binning_file"]
steve_dir = "%s" % d["steve_dir"]

simDir = "sim_spectra"
specDir = "spectra"
cov_dir = "covariances"

plot_dir = "plots/steve_spectra/"
pspy_utils.create_directory(plot_dir)

#####
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

os.system("cp %s/multistep2.js %s/multistep2.js" % (multistep_path, plot_dir))
filename = "%s/reproduce_steve.html" % plot_dir
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


for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    
    if d["tf_%s" % sv1] is not None:
        _, _, tf1, _ = np.loadtxt(d["tf_%s" % sv1], unpack=True)
    else:
        tf1 = np.ones(len(lb))

    for id_ar1, ar1 in enumerate(arrays_1):
    
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            
            if d["tf_%s" % sv2] is not None:
                _, _, tf2, _ = np.loadtxt(d["tf_%s" % sv2], unpack=True)
            else:
                tf2 = np.ones(len(lb))


            for id_ar2, ar2 in enumerate(arrays_2):

                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                
                combin = "%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)
                print("producing plots for : ", combin)
                spec_name = "%s_%s_%s" % (type, combin, "cross")


                tf = np.sqrt(tf1 * tf2)
                lb, Db = so_spectra.read_ps("%s/%s.dat" % (specDir, spec_name), spectra=spectra)
                cov = np.load("%s/analytic_cov_%s_%s.npy"%(cov_dir, combin, combin))

                for spec in  ["TT", "TE", "TB", "EE", "EB", "BB"]:


                    if spec in ["TT", "TE", "ET", "EE"]:
                        cov_select = so_cov.selectblock(cov,
                                                ["TT", "TE", "ET", "EE"],
                                                n_bins = len(lb),
                                                block=spec+spec)
                        std = np.sqrt(cov_select.diagonal())
                        std /= np.sqrt(tf)

                    else:
                        std = None
         
                    try:
                        name_mc = glob.glob("%s/*_%sx_%s%s_Nmc*_fsky*_sig_filtered.npy" % (simDir, mc_name["%s_%s" %(sv1,ar1)], mc_name["%s_%s" %(sv2,ar2)], spec))
                        mc_data = np.load("%s" % (name_mc[0]))
                    except:
                        name_mc = glob.glob("%s/*_%sx_%s%s_Nmc*_fsky*_sig_filtered.npy" % (simDir,mc_name["%s_%s" %(sv2,ar2)], mc_name["%s_%s" %(sv1,ar1)], spec))
                        mc_data = np.load("%s" % (name_mc[0]))
                        
                    mc_std = np.std(mc_data, axis=0)
                    mc_std /= tf


                    if  (sv1 == sv2) & (ar1 == ar2):
                                
                        if spec == "TE":
                            Db["TE"] = (Db["TE"] + Db["ET"])/2
                        elif spec == "TB":
                            Db["TB"] = (Db["TB"] + Db["BT"])/2
                        elif spec == "EB":
                            Db["EB"] = (Db["EB"] + Db["BE"])/2

                        name = glob.glob("%s/%s%s_lmax*_fsky*_cal_*.txt" % (steve_dir, sname["%s_%s" %(sv1,ar1)], spec))
                        l, cl, error = np.loadtxt("%s" % name[0], unpack=True)
                    else:
                        try:
                            name = glob.glob("%s/%sx_%s%s_lmax*_fsky*_cal_*.txt" % (steve_dir, sname["%s_%s" %(sv1,ar1)], sname["%s_%s" %(sv2,ar2)], spec))
                            l, cl, error = np.loadtxt("%s" % name[0], unpack=True)

                        except:
                            name = glob.glob("%s/%sx_%s%s_lmax*_fsky*_cal_*.txt" % (steve_dir, sname["%s_%s" %(sv2,ar2)], sname["%s_%s" %(sv1,ar1)], spec))
                            l, cl, error = np.loadtxt("%s" % name[0], unpack=True)
                            spec = "%s%s" % (spec[1],spec[0])
                            
                    
                    str = "%s_%s_cross.png" % (spec, combin)

                    data_analysis_utils.plot_vs_choi(l, cl, error, mc_std, Db, std, plot_dir, combin, spec)
                    
                    g.write('<div class=type>\n')
                    g.write('<img src="' + str + '" width="50%" /> \n')
                    g.write('<img src="' + 'error_divided_' + str + '" width="50%" /> \n')
                    g.write('<img src="' + 'frac_error_' + str + '" width="50%" /> \n')
                    g.write('</div>\n')

g.write('</div> \n')
g.write('</body> \n')
g.write('</html> \n')
g.close()
