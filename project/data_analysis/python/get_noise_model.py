#TO BE TESTED MORE

import matplotlib

matplotlib.use("Agg")
import sys

import numpy as np
import pylab as plt
import scipy.interpolate
from pspipe_utils import log, misc
from pspy import pspy_utils, so_dict, so_spectra


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
log = log.get_logger(**d)

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
    arrays = d[f"arrays_{sv}"]
    for id_ar1, ar1 in enumerate(arrays):
        for id_ar2, ar2 in enumerate(arrays):
            if id_ar1 > id_ar2: continue

            log.info(f"Computing noise for '{sv}' survey and '{ar1}x{ar2}' arrays")

            l, bl_ar1 = misc.read_beams(d[f"beam_T_{sv}_{ar1}"], d[f"beam_pol_{sv}_{ar1}"])
            l, bl_ar2 = misc.read_beams(d[f"beam_T_{sv}_{ar2}"], d[f"beam_pol_{sv}_{ar2}"])

            lb, nbs_ar1xar1 = so_spectra.read_ps(f"{spectra_dir}/{type}_{sv}_{ar1}x{sv}_{ar1}_noise.dat", spectra=spectra)
            lb, nbs_ar1xar2 = so_spectra.read_ps(f"{spectra_dir}/{type}_{sv}_{ar1}x{sv}_{ar2}_noise.dat", spectra=spectra)
            lb, nbs_ar2xar2 = so_spectra.read_ps(f"{spectra_dir}/{type}_{sv}_{ar2}x{sv}_{ar2}_noise.dat", spectra=spectra)
        
            bb_ar1, bb_ar2  = {}, {}
            for field in ["T", "E", "B"]:
                lb, bb_ar1[field] = pspy_utils.naive_binning(l, bl_ar1[field], binning_file, lmax)
                lb, bb_ar2[field] = pspy_utils.naive_binning(l, bl_ar2[field], binning_file, lmax)

            Rb = {}
            for spec in spectra:
                X,Y = spec
                nbs_ar1xar1[spec] *= bb_ar1[X] * bb_ar1[Y]
                nbs_ar1xar2[spec] *= bb_ar1[X] * bb_ar2[Y]
                nbs_ar2xar2[spec] *= bb_ar2[X] * bb_ar2[Y]
                
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
                         label = f"{sv} {ar1}x{ar2}",
                         color="red")
                plt.legend(fontsize=20)
                plt.savefig(f"{plot_dir}/noise_interpolate_{ar1}x{ar2}_{sv}_{spec}.png", bbox_inches="tight")
                plt.clf()
                plt.close()

            spec_name_noise_mean = f"mean_{ar1}x{ar2}_{sv}_noise"
            so_spectra.write_ps(ps_model_dir + f"/{spec_name_noise_mean}.dat", lth, nlth, type, spectra=spectra)

            if ar2 != ar1:
                spec_name_noise_mean = f"mean_{ar2}x{ar1}_{sv}_noise"
                TE, ET, TB, BT, EB, BE = nlth["ET"], nlth["TE"], nlth["BT"], nlth["TB"], nlth["BE"], nlth["EB"]
                nlth["TE"], nlth["ET"], nlth["TB"], nlth["BT"], nlth["EB"], nlth["BE"] = TE, ET, TB, BT, EB, BE
                so_spectra.write_ps(ps_model_dir + f"/{spec_name_noise_mean}.dat", lth, nlth, type, spectra=spectra)
