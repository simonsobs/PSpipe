import matplotlib
matplotlib.use("Agg")
import numpy as np
import pylab as plt
from pspy import so_dict, so_spectra, pspy_utils
import os, sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

freqs = d["freqs"]
multistep_path = d["multistep_path"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

spectra_dir = "spectra"
plot_dir = "plots"

pspy_utils.create_directory(plot_dir)

pol_efficiency= {}
pol_efficiency["100"] = 0.9995
pol_efficiency["143"] = 0.999
pol_efficiency["217"] = 0.999

os.system("cp %s/multistep2.js %s/multistep2.js" % (multistep_path, plot_dir))
file = "%s/compare_spectra.html" % (plot_dir)
g = open(file, mode="w")
g.write('<html>\n')
g.write('<head>\n')
g.write('<title> covariance </title>\n')
g.write('<script src="multistep2.js"></script>\n')
g.write('<script> add_step("sub",  ["c","v"]) </script> \n')
g.write('<style> \n')
g.write('body { text-align: center; } \n')
g.write('img { width: 100%; max-width: 1200px; } \n')
g.write('</style> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write('<div class=sub>\n')


for f1, freq1 in enumerate(freqs):
    for f2, freq2 in enumerate(freqs):
        if f1 > f2: continue
        
        spec_name = "Planck_%sxPlanck_%s-%sx%s" % (freq1, freq2, "hm1", "hm2")
        file_name = "%s/spectra_%s.dat" % (spectra_dir, spec_name)
        lb, ps_dict = so_spectra.read_ps(file_name, spectra=spectra)
        
        if freq1 != freq2:
            spec_name_2 = "Planck_%sxPlanck_%s-%sx%s" % (freq1, freq2, "hm2", "hm1")
            file_name_2 = "%s/spectra_%s.dat" % (spectra_dir, spec_name_2)
            lb, ps_dict_2 = so_spectra.read_ps(file_name_2, spectra=spectra)

            for spec in ["TT", "TE", "ET", "EE"]:
                ps_dict[spec] = (ps_dict[spec]+ ps_dict_2[spec])/2


        for spec in ["TT", "TE", "EE"]:
            
            if (spec == "TT") & (freq1 == "100") & (freq2 == "143") : continue
            if (spec == "TT") & (freq1 == "100") & (freq2 == "217") : continue
            
            planck_name = "%s_%sx%s" % (spec, freq1, freq2)
            l, cl, error = np.loadtxt("data/planck_data/spectrum_" + planck_name + ".dat", unpack=True)
            id = np.where((lb >= l[0]) & (lb <= l[-1]))
            lb_sel = lb.copy()
            lb_sel = lb_sel[id]

            if spec == "TE":
                ps_dict["TE"] = (ps_dict["TE"] + ps_dict["ET"]) / 2
                ps_dict["TE"] *= np.sqrt(pol_efficiency[freq1]*pol_efficiency[freq2])
    
            if spec == "EE":
                ps_dict["EE"] *= pol_efficiency[freq1]*pol_efficiency[freq2]
        
            ps_dict[spec] = ps_dict[spec][id]
            
            fpair = "%sx%s"%(freq1,freq2)
            
            str = "redo_planck_%s_%s.png" % (spec,fpair)

            plt.figure(figsize=(14, 7))
            plt.subplot(2, 1, 1)
            plt.errorbar(lb_sel, ps_dict[spec]*lb_sel**2/(2*np.pi), label="planck pspy %s GHz" % fpair, color="black")
            plt.errorbar(l, cl*l**2/(2*np.pi), error*l**2/(2*np.pi), fmt=".", label="planck %s GHz" % fpair, color="red")
            plt.xlabel(r"$\ell$", fontsize=22)
            plt.ylabel(r"$\ell^2 C_\ell/(2 \pi)$", fontsize=22)
            plt.legend()
            plt.subplot(2, 1, 2)
            plt.plot(lb_sel, lb_sel*0, color="black")
            plt.errorbar(lb_sel, (ps_dict[spec]-cl)/error, label="frac %s" % fpair, color="red")
            plt.xlabel(r"$\ell$", fontsize=22)
            plt.ylabel(r"$\Delta C_\ell/\sigma_{\ell} $", fontsize=22)
            plt.ylim(-0.5, 0.5)
            plt.legend()
            plt.savefig("%s/%s"%(plot_dir,str))
            plt.clf()
            plt.close()

            g.write('<div class=sub>\n')
            g.write('<img src="'+str+'"  /> \n')
            g.write('</div>\n')
    

g.write('</body> \n')
g.write('</html> \n')
g.close()




