import matplotlib
matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_mcm
from matplotlib.pyplot import cm
import numpy as np
import pylab as plt
import os, sys
import maps_to_params_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
experiments = d["experiments"]
iStart = d["iStart"]
iStop = d["iStop"]
lmax = d["lmax"]
clfile = d["clfile"]
lcut = d["lcut"]
multistep_path = d["multistep_path"]
include_fg = d["include_fg"]
fg_dir = d["fg_dir"]

fg_components = d["fg_components"]
fg_components["tt"].remove("tSZ_and_CIB")
for comp in ["tSZ", "cibc", "tSZxCIB"]:
    fg_components["tt"].append(comp)

specDir = "spectra"
mcm_dir = "mcms"
plot_dir = "plots/spectra/"
mc_dir = "montecarlo"

pspy_utils.create_directory(plot_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
nsims = iStop-iStart
ns = {exp: d["nsplits_%s"%exp] for exp in experiments}


lth, Dlth = pspy_utils.ps_lensed_theory_to_dict(clfile, output_type=type, lmax=lmax, start_at_zero=False)

theory = {}
bin_theory = {}

for id_exp1, exp1 in enumerate(experiments):
    freqs1=d["freqs_%s" % exp1]
    for id_f1, f1 in enumerate(freqs1):
        for id_exp2, exp2 in enumerate(experiments):
            freqs2 = d["freqs_%s"%exp2]
            for id_f2, f2 in enumerate(freqs2):

                if  (id_exp1 == id_exp2) & (id_f1 > id_f2) : continue
                if  (id_exp1 > id_exp2) : continue

                l, bl1 = np.loadtxt("sim_data/beams/beam_%s_%s.dat" % (exp1, f1), unpack=True)
                l, bl2 = np.loadtxt("sim_data/beams/beam_%s_%s.dat" % (exp2, f2), unpack=True)

                if exp1 == exp2:
                    nl_file_t = "sim_data/noise_ps/noise_t_%s_%sx%s_%s.dat" % (exp1, f1, exp2, f2)
                    nl_file_pol = "sim_data/noise_ps/noise_pol_%s_%sx%s_%s.dat" % (exp1, f1, exp2, f2)

                    nlth = maps_to_params_utils.get_effective_noise(nl_file_t,
                                                                    bl1,
                                                                    bl2,
                                                                    lmax,
                                                                    nl_file_pol=nl_file_pol,
                                                                    lcut=0)
                else:
                    nlth = {}
                    for spec in spectra:
                        nlth[spec] = np.zeros(lmax)

                prefix= "%s/%s_%sx%s_%s" % (mcm_dir, exp1, f1, exp2, f2)

                mbb_inv, Bbl = so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)

                for kind in ["cross", "noise", "auto"]:
                    ps_th = {}
                    for spec in spectra:
                        ps=Dlth[spec].copy()
                        if spec.lower() in fg_components:
                            if include_fg:
                                flth_all = 0
                                for foreground in fg_components[spec.lower()]:
                                    l, flth = np.loadtxt("%s/%s_%s_%sx%s.dat" % (fg_dir,spec.lower(), foreground, f1, f2)
                                                        ,unpack=True)
                                    flth_all += flth[:lmax]
                                ps = Dlth[spec] + flth_all

                        if kind == "cross":
                            ps_th[spec] = ps
                        elif kind == "noise":
                            ps_th[spec] = nlth[spec] * lth**2 / (2 * np.pi)
                        elif kind == "auto":
                            ps_th[spec] = ps + nlth[spec] * lth**2 / (2 * np.pi) * ns[exp1]

                    theory[exp1, f1, exp2, f2, kind] = ps_th
                    bin_theory[exp1, f1, exp2, f2, kind] = so_mcm.apply_Bbl(Bbl, ps_th, spectra=spectra)


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
        for id_exp1, exp1 in enumerate(experiments):
            freqs1 = d["freqs_%s" % exp1]
            for id_f1, f1 in enumerate(freqs1):
                for id_exp2, exp2 in enumerate(experiments):
                    freqs2 = d["freqs_%s" % exp2]
                    for id_f2, f2 in enumerate(freqs2):

                        if  (id_exp1 == id_exp2) & (id_f1 > id_f2) : continue
                        if  (id_exp1 > id_exp2) : continue
                        if (exp1 != exp2) & (kind == "noise"): continue
                        if (exp1 != exp2) & (kind == "auto"): continue

                        spec_name = "spectra_%s_%s_%sx%s_%s_%s" % (spec, exp1, f1, exp2, f2, kind)

                        lb, mean, std = np.loadtxt("%s/%s.dat" % (mc_dir, spec_name), unpack=True)

                        mean_dict[kind, spec, exp1, f1, exp2, f2] = mean
                        std_dict[kind, spec, exp1, f1, exp2, f2] = std

                        ps_th = theory[exp1, f1, exp2, f2, kind][spec]
                        ps_th_binned = bin_theory[exp1, f1, exp2, f2, kind][spec]

                        plt.figure(figsize=(8, 7))

                        if spec == "TT":
                            plt.semilogy()

                        plt.plot(lth, ps_th, color="grey", alpha=0.4)
                        plt.plot(lb, ps_th_binned)
                        plt.errorbar(lb, mean, std, fmt=".", color="red")
                        plt.title(r"$D^{%s,%s_{%s}x%s_{%s}}_{%s,\ell}$" % (spec, exp1, f1, exp2, f2, kind), fontsize=20)
                        plt.xlabel(r"$\ell$", fontsize=20)
                        plt.savefig("%s/%s.png" % (plot_dir, spec_name), bbox_inches="tight")
                        plt.clf()
                        plt.close()

                        plt.errorbar(lb, mean - ps_th_binned, std / np.sqrt(nsims), fmt=".", color="red")
                        plt.title(r"$\Delta D^{%s,%s_{%s}x%s_{%s}}_{%s,\ell}$" % (spec, exp1, f1, exp2, f2, kind), fontsize=20)
                        plt.xlabel(r"$\ell$", fontsize=20)
                        plt.savefig("%s/diff_%s.png" % (plot_dir, spec_name), bbox_inches="tight")
                        plt.clf()
                        plt.close()

                        plt.errorbar(lb, (mean - ps_th_binned) / (std / np.sqrt(nsims)), color="red")
                        plt.title(r"$\Delta D^{%s,%s_{%s}x%s_{%s}}_{%s,\ell}/\sigma$"%(spec, exp1, f1, exp2, f2, kind), fontsize=20)
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
        for id_exp1, exp1 in enumerate(experiments):
            freqs1 = d["freqs_%s" % exp1]
            for id_f1, f1 in enumerate(freqs1):
                for id_exp2, exp2 in enumerate(experiments):
                    freqs2 = d["freqs_%s" % exp2]
                    for id_f2, f2 in enumerate(freqs2):
                        if  (id_exp1 == id_exp2) & (id_f1 > id_f2) : continue
                        if  (id_exp1 > id_exp2) : continue

                        c = next(color)

                        mean = mean_dict["cross", spec, exp1, f1, exp2, f2]
                        std = std_dict["cross", spec, exp1, f1, exp2, f2]
                        ps_th = theory[exp1, f1, exp2, f2, "cross"][spec]

                        if (fig == "linear") and (spec == "TT"):
                            plt.errorbar(lb, mean * lb**2, std * lb**2, fmt='.', color=c, label="%s%s x %s%s" % (exp1, f1, exp2, f2), alpha=0.6)
                            plt.errorbar(lth, ps_th * lth**2, color=c, alpha=0.4)
                        else:
                            plt.errorbar(lb, mean, std, fmt='.', color=c, label="%s%s x %s%s" % (exp1, f1, exp2, f2), alpha=0.6)
                            plt.errorbar(lth, ps_th, color=c, alpha=0.4)

            exp_name += "%s_" % exp1

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
