#TO BE TESTED MORE

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pylab as plt
from pspy import so_dict, so_spectra, pspy_utils
import sys
import scipy.interpolate


def interpolate_dict(lb, cb, lth, spectra, force_positive=True, l_inf_lmin_equal_lmin=True, discard_cross=True):
       cl_dict = {}
       for spec in spectra:
            cl = scipy.interpolate.interp1d(lb, cb[spec], fill_value = "extrapolate")
            cl_dict[spec] = cl(lth)
            if l_inf_lmin_equal_lmin:
                id = np.where(lth <= np.min(lb))
                cl_dict[spec][id]= cb[spec][0]
            if force_positive:
                cl_dict[spec] = np.abs(cl_dict[spec])
            if discard_cross:
                if spec not in ["TT", "EE", "BB"]:
                    cl_dict[spec] = np.zeros(len(lth))
       return cl_dict

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

            
            lb, nbs_ar1xar1 = so_spectra.read_ps("%s/%s_%s_%sx%s_%s_noise.dat" % (spectra_dir, type, sv, ar1, sv, ar1), spectra=spectra)
            lb, nbs_ar1xar2 = so_spectra.read_ps("%s/%s_%s_%sx%s_%s_noise.dat" % (spectra_dir, type, sv, ar1, sv, ar2), spectra=spectra)
            lb, nbs_ar2xar2 = so_spectra.read_ps("%s/%s_%s_%sx%s_%s_noise.dat" % (spectra_dir, type, sv, ar2, sv, ar2), spectra=spectra)
            
            lb, bb_ar1 = pspy_utils.naive_binning(l, bl_ar1, binning_file, lmax)
            lb, bb_ar2 = pspy_utils.naive_binning(l, bl_ar2, binning_file, lmax)

            Rb = {}
            for spec in spectra:
                nbs_ar1xar1[spec] *= bb_ar1 * bb_ar1
                nbs_ar1xar2[spec] *= bb_ar1 * bb_ar2
                nbs_ar2xar2[spec] *= bb_ar2 * bb_ar2
                Rb[spec] = nbs_ar1xar2[spec] / np.sqrt(np.abs(nbs_ar1xar1[spec] * nbs_ar2xar2[spec]))
                
            if ar1 == ar2:
                nlth = interpolate_dict(lb, nbs_ar1xar1, lth, spectra)
            else:
                Rlth = interpolate_dict(lb, Rb, lth, spectra)
                nlth_ar1xar1 = interpolate_dict(lb, nbs_ar1xar1, lth, spectra)
                nlth_ar2xar2 = interpolate_dict(lb, nbs_ar2xar2, lth, spectra)
                nlth = {}
                for spec in spectra:
                    nlth[spec] = Rlth[spec] * np.sqrt(nlth_ar1xar1[spec] * nlth_ar2xar2[spec])
            
            
            for spec in spectra:
                plt.figure(figsize=(12,12))
                plt.plot(lth,
                        nlth[spec],
                        label="interpolate",
                        color="lightblue")
                if ar1 == ar2:
                    nbs= nbs_ar1xar1[spec]
                else:
                    nbs= nbs_ar1xar2[spec]

                plt.plot(lb,
                         nbs,
                         ".",
                         label = "%s %sx%s" % (sv, ar1, ar2),
                         color="red")
                plt.legend(fontsize=20)
                plt.savefig("%s/noise_interpolate_%sx%s_%s_%s.png" % (plot_dir, ar1, ar2, sv, spec), bbox_inches="tight")
                plt.clf()
                plt.close()

            spec_name_noise_mean = "mean_%sx%s_%s_noise" % (ar1, ar2, sv)
            so_spectra.write_ps(ps_model_dir + "/%s.dat" % spec_name_noise_mean, lth, nlth, type, spectra=spectra)

                

