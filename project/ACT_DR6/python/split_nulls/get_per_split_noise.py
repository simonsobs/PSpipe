"""
This script is used to compute the per-split noise
from pre-computed power spectra. `write_split_spectra` has to be
set to `True` to run this script.
"""
from pspy import so_dict, pspy_utils, so_spectra
import sys
from pspipe_utils import pspipe_list
import scipy.interpolate
import numpy as np
import matplotlib.pyplot as plt

def interpolate_dict(lb, cb, lth, spectra, force_positive=True, l_inf_lmin_equal_lmin=True, discard_cross=True):
       cl_dict = {}
       for spec in spectra:
            cl = scipy.interpolate.interp1d(lb, cb[spec], fill_value="extrapolate")
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
split_noise_dir = "split_noise"
plot_dir = "plots/split_noise"

pspy_utils.create_directory(split_noise_dir)
pspy_utils.create_directory(plot_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]

lth = np.arange(2, lmax+2)

arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
n_splits = {sv: d[f"n_splits_{sv}"] for sv in surveys}

spec_name_list = pspipe_list.get_spec_name_list(d)

split_noise_dict = {}
for spec_name in spec_name_list:
    sv1ar1, sv2ar2 = spec_name.split("x")
    sv1, ar1 = sv1ar1.split("&")
    sv2, ar2 = sv2ar2.split("&")

    if sv1 != sv2: continue

    split_ps_dict = {}
    for i in range(n_splits[sv1]):
        for j in range(n_splits[sv2]):
            if i > j and sv1ar1 == sv2ar2: continue

            lb, ps = so_spectra.read_ps(f"{spectra_dir}/Dl_{sv1}_{ar1}x{sv2}_{ar2}_{i}{j}.dat",
                                        spectra=spectra)
            split_ps_dict[i, j] = ps

    for i in range(n_splits[sv1]):

        cross_ps_dict = {spec: [] for spec in spectra}

        for id1, id2 in list(split_ps_dict.keys()):
            if id1 == id2: continue

            for spec in spectra:
                cross_ps_dict[spec].append(split_ps_dict[id1, id2][spec])

        for spec in spectra:
            cross_ps_dict[spec] = np.mean(cross_ps_dict[spec], axis=0)

        split_noise = {spec: split_ps_dict[i, i][spec] - cross_ps_dict[spec] for spec in spectra}

        split_noise_dict[sv1, ar1, sv2, ar2, i] = split_noise

        spec_name = f"Dl_{sv1}_{ar1}x{sv2}_{ar2}_{i}{i}_noise.dat"
        so_spectra.write_ps(f"{split_noise_dir}/{spec_name}", lb, split_noise, type, spectra=spectra)

for (sv1, ar1, sv2, ar2, split), split_noise in split_noise_dict.items():

    # Noise model
    l, bl1 = pspy_utils.read_beam_file(d[f"beam_{sv1}_{ar1}"])
    l, bl2 = pspy_utils.read_beam_file(d[f"beam_{sv2}_{ar2}"])

    lb, bb1 = pspy_utils.naive_binning(l, bl1, binning_file, lmax)
    lb, bb2 = pspy_utils.naive_binning(l, bl2, binning_file, lmax)

    if ar1 == ar2:
        for spec in spectra:
            split_noise[spec] = split_noise[spec] * bb1 * bb2
        nlth = interpolate_dict(lb, split_noise, lth, spectra)

    else:
        nb_ar1xar1 = {spec: split_noise_dict[sv1, ar1, sv1, ar1, split][spec] * bb1 * bb1 for spec in spectra}
        nb_ar2xar2 = {spec: split_noise_dict[sv2, ar2, sv2, ar2, split][spec] * bb2 * bb2 for spec in spectra}
        nb_ar1xar2 = {spec: split_noise_dict[sv1, ar1, sv2, ar2, split][spec] * bb1 * bb2 for spec in spectra}

        Rb = {spec : nb_ar1xar2[spec] / np.sqrt(np.abs(nb_ar1xar1[spec] * nb_ar2xar2[spec])) for spec in spectra}

        Rlth = interpolate_dict(lb, Rb, lth, spectra)
        nlth_ar1xar1 = interpolate_dict(lb, nb_ar1xar1, lth, spectra)
        nlth_ar2xar2 = interpolate_dict(lb, nb_ar2xar2, lth, spectra)

        nlth = {spec: Rlth[spec] * np.sqrt(nlth_ar1xar1[spec] * nlth_ar2xar2[spec]) for spec in spectra}

    spec_model_name = f"Dl_{ar1}_{split}x{ar2}_{split}_{sv1}_noise_model.dat"
    so_spectra.write_ps(f"{split_noise_dir}/{spec_model_name}", lth, nlth, type, spectra=spectra)

    for split2 in range(n_splits[sv1]):
        if split == split2: continue
        nl_xsplit = {spec: np.zeros_like(nlth[spec]) for spec in spectra}
        spec_name = f"Dl_{ar1}_{split}x{ar2}_{split2}_{sv1}_noise_model.dat"
        so_spectra.write_ps(f"{split_noise_dir}/{spec_name}", lth, nl_xsplit, type, spectra=spectra)

# Plots
for spec_name in spec_name_list:
    sv1ar1, sv2ar2 = spec_name.split("x")
    sv1, ar1 = sv1ar1.split("&")
    sv2, ar2 = sv2ar2.split("&")

    if sv1 != sv2: continue

    noise_model_file = f"{ps_model_dir}/mean_{ar1}x{ar2}_{sv1}_noise.dat"
    ell, nl_mean = so_spectra.read_ps(noise_model_file, spectra=spectra)

    for spec in ["TT", "EE", "BB"]:
        plt.figure(figsize = (13, 8))
        for i in range(n_splits[sv1]):
            split_noise_model_file = f"{split_noise_dir}/Dl_{ar1}_{i}x{ar2}_{i}_{sv1}_noise_model.dat"
            ell, nl = so_spectra.read_ps(split_noise_model_file, spectra=spectra)
            plt.plot(ell, nl[spec], label=f"Split {i}")

        plt.plot(ell, nl_mean[spec] * n_splits[sv1], label="Mean noise", color="k", ls="--")
        plt.title(f"{sv1ar1}x{sv2ar2}", fontsize=16.5)
        plt.xlabel(r"$\ell$", fontsize=15)
        plt.ylabel(f"$N_\ell^{{{spec}}}$", fontsize=15)
        plt.yscale("log")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/noise_{sv1ar1}x{sv2ar2}_{spec}.png", dpi=300)

        if sv1ar1 == sv2ar2:
            plt.figure(figsize=(13,8))
            plt.axhline(1, color="k", ls="--")
            for i in range(n_splits[sv1]):
                split_noise_model_file = f"{split_noise_dir}/Dl_{ar1}_{i}x{ar2}_{i}_{sv1}_noise_model.dat"
                ell, nl = so_spectra.read_ps(split_noise_model_file, spectra = spectra)
                plt.plot(ell, nl[spec]/nl_mean[spec]/n_splits[sv1], label = f"Split {i}")
            plt.title(f"{ar1}x{ar2}", fontsize = 16.5)
            plt.xlabel(r"$\ell$", fontsize = 15)
            plt.ylabel(r"$N_\ell^{%s}/N_\ell^{%s,\mathrm{mean}}$" % (spec, spec), fontsize = 15)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/noise_ratio_{sv1ar1}x{sv2ar2}_{spec}.png", dpi = 300)
