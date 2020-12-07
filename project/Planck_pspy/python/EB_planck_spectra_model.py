import numpy as np
import pylab as plt
from pspy import so_dict, so_spectra, pspy_utils
import sys
import scipy.interpolate
import planck_utils

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra_dir = "spectra"
plot_dir = "plots"
bestfit_dir = "best_fits"
plot_dir = "plots"

pspy_utils.create_directory(bestfit_dir)
pspy_utils.create_directory(plot_dir)


spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
type = d["type"]
freqs = d["freqs"]
binning_file = d["binning_file"]
lmax = d["lmax"]
splits = ["hm1", "hm2"]
size = 1
exp = "Planck"


color_array = ["blue", "red", "green", "orange", "purple", "pink"]

clth = {}
lth, clth["TT"], clth["EE"], clth["BB"], clth["TE"] =np.loadtxt("data/cosmo2017_10K_acc3_lensedCls.dat", unpack=True)
clth["EB"] = 0

for spec in ["EE", "BB", "EB"]:
    clth[spec] *= 2 * np.pi/(lth * (lth + 1))

    id_color = 0
    for f1, freq1 in enumerate(freqs):
        for f2, freq2 in enumerate(freqs):
            if f1 > f2: continue
            fname = "%sx%s" % (freq1, freq2)

            spec_name = "%s_%sx%s_%s-%sx%s" % (exp, freq1, exp, freq2, "hm1", "hm2")
            l, ps = so_spectra.read_ps("%s/spectra_unbin_%s.dat" % (spectra_dir, spec_name), spectra=spectra)
            
            #ps[spec] = ps[spec][:lmax]
            if spec is not "EB":
                fg = ps[spec] - clth[:len(ps[spec])]
                fg_th = np.zeros(len(lth))
                pivot = 75
                power1 = -1.9
                power2 = -2.5
                lmin_fit = 30
                lmax_fit = 800
            
                A_amplitude = np.linspace(0.1, 100, 1000)
                chi2 = np.zeros(1000)
                for c, A in enumerate(A_amplitude):
                    fg_th[lmin_fit:pivot] = (lc[lmin_fit:pivot]/A)**(power1)*(lc[pivot]/A)**(power2-power1)
                    fg_th[pivot:] = (lc[pivot:]/A)**(power2)
                    chi2[c] = np.sum((fg_th[lmin_fit:lmax_fit]-fg[lmin_fit:lmax_fit])**2)
            
                id = np.where(chi2 == np.min(chi2))
                A_bestfit = A_amplitude[id]
                print(f1,f2,A_bestfit)
            
                fg_th[:pivot] = (lth[:pivot]/A_bestfit)**(power1)*(lth[pivot]/A_bestfit)**(power2-power1)
                fg_th[pivot:] = (lth[pivot:]/A_bestfit)**(power2)
            else:
                fg_th = np.zeros(len(lth))
            
            lth_padded = np.arange(len(lth))
            best_fit = np.zeros(len(lth))
            best_fit[2:] = clth[spec][:-2] + fg_th[:-2]
            
            np.savetxt("%s/best_fit_%s_%s.dat" % (bestfit_dir, fname, spec), np.transpose([lth_padded, best_fit]))

            model = (clth[spec] + fg_th) * l * (l +1) / (2 * np.pi)
            ps[spec] *= (l * (l +1) / (2 * np.pi))
            if spec != "EB":
                plt.semilogy()
            plt.plot(l[lmin_fit:],  model[lmin_fit:], color=color_array[id_color])
            plt.errorbar(l[lmin_fit:], ps[spec][lmin_fit:], label = "%s_%sx%s" % (spec, freq1, freq2), fmt=".", color=color_array[id_color], alpha=0.2)

            id_color += 1

    plt.legend()
    plt.savefig("%s/best_fit_%s.png" % (plot_dir,spec))
    plt.clf()
    plt.close()




