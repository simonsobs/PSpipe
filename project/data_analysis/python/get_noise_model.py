import matplotlib
matplotlib.use("Agg")
import numpy as np
import pylab as plt
from pspy import so_dict, so_spectra, pspy_utils
import sys
import scipy.interpolate

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra_dir = "spectra"
ps_model_dir = "noise_model"
plot_dir = "plots/noise_model/"

pspy_utils.create_directory(ps_model_dir)
pspy_utils.create_directory(plot_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]

lth = np.arange(2, lmax+2)

for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for id_ar1, ar1 in enumerate(arrays):
        for id_ar2, ar2 in enumerate(arrays):
            if id_ar1 > id_ar2: continue
            l, bl_ar1 = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv, ar1)])
            l, bl_ar2 = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv, ar2)])


            lb, bb_ar1 = pspy_utils.naive_binning(l, bl_ar1, binning_file, lmax)
            lb, bb_ar2 = pspy_utils.naive_binning(l, bl_ar2, binning_file, lmax)

            nsplits = len(d["maps_%s_%s" % (sv, ar1)])
        
            spec_name_noise = "%s_%s_%sx%s_%s_noise" % (type, sv, ar1, sv, ar2)
            lb, nbs = so_spectra.read_ps("%s/%s.dat" % (spectra_dir, spec_name_noise), spectra=spectra)
        
            nl_dict = {}
            for spec in spectra:
                nbs_mean = nbs[spec] * bb_ar1*bb_ar2
                plt.figure(figsize=(12,12))

                if (spec == "TT" or spec == "EE" or spec == "BB") & (ar1 == ar2):
                
                    nl = scipy.interpolate.interp1d(lb, nbs_mean, fill_value = "extrapolate")
                    nl_dict[spec] = np.array([nl(i) for i in lth])
                    id = np.where(lth <= np.min(lb))
                    nl_dict[spec][id]= nbs_mean[0]
                    nl_dict[spec] = np.abs(nl_dict[spec])
                    
                    plt.semilogy()
                    
                    plt.plot(lth,
                             nl_dict[spec],
                             label="interpolate",
                             color="lightblue")

                else:
                    nl_dict[spec] = np.zeros(len(lth))
                
                plt.plot(lb,
                         nbs_mean,
                         ".",
                         label = "%s %sx%s" % (sv, ar1, ar2),
                        color="red")
                
                plt.legend(fontsize=20)
                plt.savefig("%s/noise_interpolate_%sx%s_%s_%s.png" % (plot_dir, ar1, ar2, sv, spec), bbox_inches="tight")
                plt.clf()
                plt.close()

            spec_name_noise_mean = "mean_%sx%s_%s_noise" % (ar1, ar2, sv)

            so_spectra.write_ps(ps_model_dir + "/%s.dat" % spec_name_noise_mean, lth, nl_dict, type, spectra=spectra)
