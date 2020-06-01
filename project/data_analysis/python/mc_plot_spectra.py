import matplotlib
matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_mcm, so_spectra
from matplotlib.pyplot import cm
import numpy as np
import pylab as plt
import os, sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]
iStart = d["iStart"]
iStop = d["iStop"]
lmax = d["lmax"]
clfile = d["theoryfile"]
multistep_path = d["multistep_path"]

noise_dir = "noise_model"
mcm_dir = "mcms"
plot_dir = "plots/mc_spectra/"
mc_dir = "montecarlo"
bestfit_dir = "best_fits"


pspy_utils.create_directory(plot_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
nsims = iStop - iStart


lth, Dlth = pspy_utils.ps_lensed_theory_to_dict(clfile, output_type=type, lmax=lmax, start_at_zero=False)

theory = {}
bin_theory = {}

for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            for id_ar2, ar2 in enumerate(arrays_2):

                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue

                l, bl1 = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv1, ar1)], lmax=lmax)
                l, bl2 = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv2, ar2)], lmax=lmax)

                if sv1 == sv2:
                    lb, nlth = so_spectra.read_ps("%s/mean_%sx%s_%s_noise.dat" % (noise_dir, ar1, ar2, sv1), spectra=spectra)
                    for spec in spectra:
                        nlth[spec]  /= (bl1 * bl2)
                else:
                    nlth = {}
                    for spec in spectra:
                        nlth[spec] = np.zeros(lmax)

                prefix= "%s/%s_%sx%s_%s" % (mcm_dir, sv1, ar1, sv2, ar2)
                
                mbb_inv, Bbl = so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)

                for kind in ["cross", "noise", "auto"]:
                
                    if (sv1 != sv2) & (kind == "noise"): continue
                    if (sv1 != sv2) & (kind == "auto"): continue

                    ps_th = {}
                    for spec in spectra:
                        ps=Dlth[spec].copy()
                        if spec == "TT":
                        
                            nu_eff_1 = d["nu_eff_%s_%s" % (sv1, ar1)]
                            nu_eff_2 = d["nu_eff_%s_%s" % (sv2, ar2)]
                            
                            _, flth = np.loadtxt("%s/fg_%sx%s.dat" %(bestfit_dir, nu_eff_1, nu_eff_2), unpack=True)
                            
                            ps = Dlth[spec] + flth[:lmax]
                    
                        if kind == "cross":
                            ps_th[spec] = ps
                        elif kind == "noise":
                            ps_th[spec] = nlth[spec]
                        elif kind == "auto":
                            ns = len( d["maps_%s_%s" % (sv1, ar1)])

                            ps_th[spec] = ps + nlth[spec] * ns
    
                    theory[sv1, ar1, sv2, ar2, kind] = ps_th
                    bin_theory[sv1, ar1, sv2, ar2, kind] = so_mcm.apply_Bbl(Bbl, ps_th, spectra=spectra)


os.system("cp %s/multistep2.js %s/multistep2.js" % (multistep_path, plot_dir))
filename = "%s/SO_spectra.html" % plot_dir
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

mean_dict = {}
std_dict = {}
n_spec = {}

for kind in ["cross", "noise", "auto"]:
    g.write('<div class=all>\n')
    id_spec = 0
    for spec in spectra:
        n_spec[kind] = 0
        for id_sv1, sv1 in enumerate(surveys):
            arrays_1 = d["arrays_%s" % sv1]
            for id_ar1, ar1 in enumerate(arrays_1):
                for id_sv2, sv2 in enumerate(surveys):
                    arrays_2 = d["arrays_%s" % sv2]
                    for id_ar2, ar2 in enumerate(arrays_2):

                        if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                        if  (id_sv1 > id_sv2) : continue
                        if (sv1 != sv2) & (kind == "noise"): continue
                        if (sv1 != sv2) & (kind == "auto"): continue

                        spec_name = "spectra_%s_%s_%sx%s_%s_%s" % (spec, sv1, ar1, sv2, ar2, kind)

                        lb, mean, std = np.loadtxt("%s/%s.dat" % (mc_dir, spec_name), unpack=True)
                        
                        mean_dict[kind, spec, sv1, ar1, sv2, ar2] = mean
                        std_dict[kind, spec, sv1, ar1, sv2, ar2] = std
                        
                        ps_th = theory[sv1, ar1, sv2, ar2, kind][spec]
                        ps_th_binned = bin_theory[sv1, ar1, sv2, ar2, kind][spec]
                        
                        plt.figure(figsize=(8, 7))
                
                        if spec == "TT":
                            plt.semilogy()
                
                        plt.plot(lth, ps_th, color="grey", alpha=0.4)
                        plt.plot(lb, ps_th_binned)
                        plt.errorbar(lb, mean, std, fmt=".", color="red")
                        plt.title(r"$D^{%s,%s_{%s}x%s_{%s}}_{%s,\ell}$" % (spec, sv1, ar1, sv2, ar2, kind), fontsize=20)
                        plt.xlabel(r"$\ell$", fontsize=20)
                        plt.savefig("%s/%s.png" % (plot_dir, spec_name), bbox_inches="tight")
                        plt.clf()
                        plt.close()
                        
                        plt.errorbar(lb, mean - ps_th_binned, std / np.sqrt(nsims), fmt=".", color="red")
                        plt.title(r"$\Delta D^{%s,%s_{%s}x%s_{%s}}_{%s,\ell}$" % (spec, sv1, ar1, sv2, ar2, kind), fontsize=20)
                        plt.xlabel(r"$\ell$", fontsize=20)
                        plt.savefig("%s/diff_%s.png" % (plot_dir, spec_name), bbox_inches="tight")
                        plt.clf()
                        plt.close()

                        plt.errorbar(lb, (mean - ps_th_binned) / (std / np.sqrt(nsims)), color="red")
                        plt.title(r"$\Delta D^{%s,%s_{%s}x%s_{%s}}_{%s,\ell}/\sigma$"%(spec, sv1, ar1, sv2, ar2, kind), fontsize=20)
                        plt.xlabel(r"$\ell$", fontsize=20)
                        plt.savefig("%s/frac_%s.png" % (plot_dir, spec_name), bbox_inches="tight")
                        plt.clf()
                        plt.close()
                        
                        str = "%s.png" % (spec_name)
                        
                        g.write('<div class=type>\n')
                        g.write('<img src="' + str + '" width="50%" /> \n')
                        g.write('<img src="' + 'diff_' + str + '" width="50%" /> \n')
                        g.write('<img src="' + 'frac_' + str + '" width="50%" /> \n')
                        g.write('</div>\n')

                        n_spec[kind] += 1


    g.write('</div>\n')
g.write('</div> \n')
g.write('</body> \n')
g.write('</html> \n')
g.close()



for fig in ["log", "linear"]:
    for spec in ["TT", "TE", "ET", "EE"]:

        plt.figure(figsize=(12, 12))
        color = iter(cm.rainbow(np.linspace(0, 1, n_spec["cross"] + 1)))
    
        if fig == "log":
            plt.semilogy()

        exp_name = ""
        for id_sv1, sv1 in enumerate(surveys):
            arrays_1 = d["arrays_%s" % sv1]
            for id_ar1, ar1 in enumerate(arrays_1):
                for id_sv2, sv2 in enumerate(surveys):
                    arrays_2 = d["arrays_%s" % sv2]
                    for id_ar2, ar2 in enumerate(arrays_2):
                        if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                        if  (id_sv1 > id_sv2) : continue
                
                        c = next(color)
                        
                        mean = mean_dict["cross", spec, sv1, ar1, sv2, ar2]
                        std = std_dict["cross", spec, sv1, ar1, sv2, ar2]
                        ps_th = theory[sv1, ar1, sv2, ar2, "cross"][spec]
                    
                        if (fig == "linear") and (spec == "TT"):
                            plt.errorbar(lb, mean * lb**2, std * lb**2, fmt='.', color=c, label="%s%s x %s%s" % (sv1, ar1, sv2, ar2), alpha=0.6)
                            plt.errorbar(lth, ps_th * lth**2, color=c, alpha=0.4)
                        else:
                            plt.errorbar(lb, mean, std, fmt='.', color=c, label="%s%s x %s%s" % (sv1, ar1, sv2, ar2), alpha=0.6)
                            plt.errorbar(lth, ps_th, color=c, alpha=0.4)

            exp_name += "%s_" % sv1

        if (fig == "log") and (spec == "TT"):
            plt.ylim(10, 10**4)
        if (fig == "linear") and (spec == "TT"):
            plt.ylim(0, 2 * 10**9)

        if fig == "log":
            plt.legend(fontsize=14, bbox_to_anchor=(1.4, 1.1))
        else:
            plt.legend(fontsize=14, bbox_to_anchor=(1.4, 1.))

        plt.xlabel(r"$\ell$", fontsize=20)

        if (fig == "linear") and (spec == "TT"):
            plt.ylabel(r"$\ell^{2} D^{%s}_\ell$" % spec, fontsize=20)
        else:
            plt.ylabel(r"$D^{%s}_\ell$" % spec, fontsize=20)

        plt.savefig("%s/all_%s_spectra_%s_all_%scross.png" % (plot_dir, fig, spec, exp_name), bbox_inches="tight")
        plt.clf()
        plt.close()





