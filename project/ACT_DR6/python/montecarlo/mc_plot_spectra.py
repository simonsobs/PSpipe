"""
This script plot the mean of the simulation spectra with respect to the input theory
It is there to catch bias from the pipeline
"""


import matplotlib
matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_spectra, so_mcm
from matplotlib.pyplot import cm
import numpy as np
import pylab as plt
import os, sys
from pspipe_utils import pspipe_list, best_fits, log, misc
import pspipe

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

type = d["type"]
surveys = d["surveys"]
iStart = d["iStart"]
iStop = d["iStop"]
lmax = d["lmax"]
multistep_path = os.path.join(os.path.dirname(pspipe.__file__), "js")

noise_model_dir = "noise_model"
mcm_dir = "mcms"
plot_dir = "plots/mc_spectra/"
mc_dir = "montecarlo"
bestfit_dir = "best_fits"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

_, sv_list, ar_list = pspipe_list.get_arrays_list(d)
array_list = [f"{sv}_{ar}" for (sv, ar) in zip(sv_list, ar_list)]
lth, cmb_and_fg_dict = best_fits.fg_dict_from_files(bestfit_dir + "/fg_{}x{}.dat",
                                                    array_list,
                                                    lmax + 2,
                                                    spectra,
                                                    f_name_cmb=bestfit_dir + "/cmb.dat")


pspy_utils.create_directory(plot_dir)

spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
nsims = iStop - iStart

diff_l_fac = {}
diff_l_fac["TT"] = 1
diff_l_fac["TE"] = 0
diff_l_fac["TB"] = -0.8
diff_l_fac["ET"] = 0
diff_l_fac["BT"] = -0.8
diff_l_fac["EE"] = -0.8
diff_l_fac["EB"] = -0.8
diff_l_fac["BE"] = -0.8
diff_l_fac["BB"] = -0.8


theory = {}
bin_theory = {}

for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d[f"arrays_{sv1}"]
    for id_ar1, ar1 in enumerate(arrays_1):
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d[f"arrays_{sv2}"]
            for id_ar2, ar2 in enumerate(arrays_2):

                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                
                log.info(f"prepare theory spectra {sv1} {ar1} x {sv2} {ar2}")

                l, bl1 = misc.read_beams(d[f"beam_T_{sv1}_{ar1}"], d[f"beam_pol_{sv1}_{ar1}"], lmax=lmax)
                l, bl2 = misc.read_beams(d[f"beam_T_{sv2}_{ar2}"], d[f"beam_pol_{sv2}_{ar2}"],  lmax=lmax)

                if sv1 == sv2:
                    lb, nlth = so_spectra.read_ps(f"{noise_model_dir}/mean_{ar1}x{ar2}_{sv1}_noise.dat", spectra=spectra)
                    for spec in spectra:
                        X, Y = spec
                        nlth[spec]  /= (bl1[X] * bl2[Y])
                else:
                    nlth = {}
                    for spec in spectra:
                        nlth[spec] = np.zeros(lmax)

                prefix= f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}"

                mbb_inv, Bbl = so_mcm.read_coupling(prefix=prefix,spin_pairs=spin_pairs)

                for kind in ["cross", "noise", "auto"]:

                    if (sv1 != sv2) & (kind == "noise"): continue
                    if (sv1 != sv2) & (kind == "auto"): continue

                    ps_th = {}
                    for spec in spectra:

                        if kind == "cross":
                            ps_th[spec] = cmb_and_fg_dict[f"{sv1}_{ar1}", f"{sv2}_{ar2}"][spec]
                        elif kind == "noise":
                            ps_th[spec] = nlth[spec]
                        elif kind == "auto":
                            n_splits = len(d[f"maps_{sv1}_{ar1}"])
                            ps_th[spec] = cmb_and_fg_dict[f"{sv1}_{ar1}", f"{sv2}_{ar2}"][spec] + nlth[spec] * n_splits

                    theory[sv1, ar1, sv2, ar2, kind] = ps_th
                    bin_theory[sv1, ar1, sv2, ar2, kind] = so_mcm.apply_Bbl(Bbl, ps_th, spectra=spectra)


os.system(f"cp {multistep_path}/multistep2.js {plot_dir}/multistep2.js")
filename = f"{plot_dir}/SO_spectra.html"
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
            arrays_1 = d[f"arrays_{sv1}"]
            for id_ar1, ar1 in enumerate(arrays_1):
                for id_sv2, sv2 in enumerate(surveys):
                    arrays_2 = d[f"arrays_{sv2}"]
                    for id_ar2, ar2 in enumerate(arrays_2):

                        if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                        if  (id_sv1 > id_sv2) : continue
                        if (sv1 != sv2) & (kind == "noise"): continue
                        if (sv1 != sv2) & (kind == "auto"): continue

                        spec_name = f"spectra_{spec}_{sv1}_{ar1}x{sv2}_{ar2}_{kind}"
                        
                        log.info(f"plot {spec_name}")

                        lb, mean, std = np.loadtxt(f"{mc_dir}/{spec_name}.dat", unpack=True)

                        
                        ps_th = theory[sv1, ar1, sv2, ar2, kind][spec]
                        ps_th_binned = bin_theory[sv1, ar1, sv2, ar2, kind][spec]

                        
                        # skip the first bin, we never use it and it dominated the plotting range
                        lb = lb[1:]
                        mean = mean[1:]
                        std = std[1:]
                        ps_th_binned = ps_th_binned[1:]
                        
                        
                        mean_dict[kind, spec, sv1, ar1, sv2, ar2] = mean
                        std_dict[kind, spec, sv1, ar1, sv2, ar2] = std

                        plt.figure(figsize=(8, 7))

                        if spec == "TT":
                            plt.semilogy()

                        plt.plot(lth, ps_th, color="grey", alpha=0.4)
                        plt.plot(lb, ps_th_binned)
                        plt.errorbar(lb, mean, std, fmt=".", color="red")
                        plt.title(r"$D^{%s, %s %s x %s %s}_{%s,\ell}$ (press c/v for type, j/k for spectra, a/z for residual)" % (spec, sv1, ar1, sv2, ar2, kind), fontsize=20)
                        plt.xlabel(r"$\ell$", fontsize=20)
                        plt.savefig("%s/%s.png" % (plot_dir, spec_name), bbox_inches="tight")
                        plt.clf()
                        plt.close()

                        plt.errorbar(lb,
                                     (mean - ps_th_binned) * lb ** diff_l_fac[spec],
                                     std / np.sqrt(nsims)  * lb ** diff_l_fac[spec],
                                     fmt=".",
                                     color="red")
                        plt.plot(lb, lb * 0, color="gray", alpha=0.6)
                        plt.title(r"$\ell^{%.01f} \Delta D^{%s, %s %s x %s %s}_{%s,\ell}$ (press c/v for type, j/k for spectra, a/z for residual)" % (diff_l_fac[spec], spec, sv1, ar1, sv2, ar2, kind), fontsize=20)
                        plt.xlabel(r"$\ell$", fontsize=20)
                        plt.savefig("%s/diff_%s.png" % (plot_dir, spec_name), bbox_inches="tight")
                        plt.clf()
                        plt.close()

                        plt.errorbar(lb, (mean - ps_th_binned) / (std / np.sqrt(nsims)), color="red")
                        plt.title(r"$\Delta D^{%s, %s %s x %s %s}_{%s,\ell}/\sigma$ (press c/v for type, j/k for spectra, a/z for residual)"%(spec, sv1, ar1, sv2, ar2, kind), fontsize=20)
                        plt.xlabel(r"$\ell$", fontsize=20)
                        plt.savefig("%s/frac_%s.png" % (plot_dir, spec_name), bbox_inches="tight")
                        plt.clf()
                        plt.close()



                        if kind == "noise":
                            plt.errorbar(lb, mean / ps_th_binned, color="red")
                            plt.title(r"$ (D^{\rm MC}/D^{\rm data})^{ %s %s x %s %s }_{%s, %s, \ell} $" % (sv1, ar1, sv2, ar2, spec, kind), fontsize=20)
                            plt.xlabel(r"$\ell$", fontsize=20)
                            plt.ylim(0.7, 1.3)
                            plt.savefig("%s/ratio_%s.png" % (plot_dir, spec_name), bbox_inches="tight")
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
            arrays_1 = d[f"arrays_{sv1}"]
            for id_ar1, ar1 in enumerate(arrays_1):
                for id_sv2, sv2 in enumerate(surveys):
                    arrays_2 = d[f"arrays_{sv2}"]
                    for id_ar2, ar2 in enumerate(arrays_2):
                        if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                        if  (id_sv1 > id_sv2) : continue

                        c = next(color)

                        mean = mean_dict["cross", spec, sv1, ar1, sv2, ar2]
                        std = std_dict["cross", spec, sv1, ar1, sv2, ar2]
                        ps_th = theory[sv1, ar1, sv2, ar2, "cross"][spec]

                        if (fig == "linear") and (spec == "TT"):
                            plt.errorbar(lb, mean * lb ** 2, std * lb ** 2, fmt='.', color=c, label=f"{sv1}{ar1} x {sv2}{ar2}", alpha=0.6)
                            plt.errorbar(lth, ps_th * lth ** 2, color=c, alpha=0.4)
                        else:
                            plt.errorbar(lb, mean, std, fmt='.', color=c, label=f"{sv1}{ar1} x {sv2}{ar2}", alpha=0.6)
                            plt.errorbar(lth, ps_th, color=c, alpha=0.4)

            exp_name += "%s_" % sv1

        if (fig == "log") and (spec == "TT"):
            plt.ylim(10, 10 ** 4)
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

        plt.savefig(f"{plot_dir}/all_{fig}_spectra_{spec}_all_{exp_name}cross.png", bbox_inches="tight")
        plt.clf()
        plt.close()
