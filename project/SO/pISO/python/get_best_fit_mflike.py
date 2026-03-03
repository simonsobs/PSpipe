description = """
This script compute best fit from theory and fg power spectra.
It uses camb and the foreground model of mflike based on fgspectra
"""
import matplotlib

matplotlib.use("Agg")
from os.path import join as opj
import argparse

import numpy as np
from matplotlib import pyplot as plt
from pspipe_utils import best_fits, log, pspipe_list, beam_chromaticity
from pspy import pspy_utils, so_dict, so_spectra

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

# first let's get a list of all frequency we plan to study
surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

# let's create the directories to write best fit to disk and for plotting purpose

# tag = d.get("best_fit_tag", "")

bestfit_dir = d["best_fits_dir"]
components_dir = f"{bestfit_dir}/components"

plot_dir = opj(d['plots_dir'], 'best_fits')



log.info(f"save best fits in {bestfit_dir} folder")

pspy_utils.create_directory(bestfit_dir)
pspy_utils.create_directory(components_dir)
pspy_utils.create_directory(plot_dir)

cosmo_params = d["cosmo_params"]
log.info(f"Computing lensed CMB spectra with cosmological parameters: {cosmo_params}")
l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, type, lmax + 500, **d["accuracy_params"])

f_name = f"{bestfit_dir}/cmb.dat"
so_spectra.write_ps(f_name, l_th, ps_dict, type, spectra=spectra)

# also compute unlensed spectra with lensing spectra in case needed for sims.
# this will include PP, PT, PE spectra so we need the keys of the dict.
log.info(f"Computing unlensed CMB spectra and lensing spectra with cosmological parameters: {cosmo_params}")
_, unlensed_ps_dict = pspy_utils.unlensed_ps_from_params(cosmo_params, type, lmax + 500, **d["accuracy_params"])

f_name = f"{bestfit_dir}/unlensed_cmb_and_lensing.dat"
so_spectra.write_ps(f_name, l_th, unlensed_ps_dict, type, spectra=unlensed_ps_dict.keys())


fg_norm = d["fg_norm"]
fg_params = d["fg_params"]
fg_components = d["fg_components"]
log.info(f"Computing fg spectra with components: {fg_components}")
log.info(f"Computing fg spectra with params: {fg_params}")

passbands, band_shift_dict = {}, {}
do_bandpass_integration = d["do_bandpass_integration"]

if do_bandpass_integration:
    log.info("Doing bandpass integration")

narrays, _, _ = pspipe_list.get_arrays_list(d)
map_set_list = pspipe_list.get_map_set_list(d)

for map_set in map_set_list:
    freq_info = d[f"freq_info_{map_set}"]
    if do_bandpass_integration:
        nu_ghz, pb = np.loadtxt(freq_info["passband"]).T
        
        # delete any 0-freq entries
        good_idxs = nu_ghz > 0
        nu_ghz = nu_ghz[good_idxs]
        pb = pb[good_idxs]
    else:
        nu_ghz, pb = np.array([freq_info["freq_tag"]]), np.array([1.])

    passbands[f"{map_set}"] = [nu_ghz, pb]
    band_shift_dict[f"bandint_shift_{map_set}"] = d[f"bandpass_shift_{map_set}"]
    log.info(f"bandpass shift: {map_set} {band_shift_dict[f'bandint_shift_{map_set}']}")

beams = None
if d["include_beam_chromaticity_effect_in_best_fit"]:
    log.info(f"include beam array accounting for beam chromaticity \n")
    # Get beam chromaticity
    alpha_dict, nu_ref_dict = beam_chromaticity.act_dr6_beam_scaling()
    beams = {}
    for map_set in map_set_list:
        bl_mono_file_name = d[f"beam_mono_{map_set}"]
        l, bl = pspy_utils.read_beam_file(bl_mono_file_name, lmax=10000)
        l, nu_array, bl_nu = beam_chromaticity.get_multifreq_beam(l,
                                                                  bl,
                                                                  passbands[map_set],
                                                                  nu_ref_dict[map_set],
                                                                  alpha_dict[map_set])
                                                                  
        beams[map_set + "_s0"] = {"nu": nu_array, "beams": bl_nu.T}
        beams[map_set + "_s2"] = {"nu": nu_array, "beams": bl_nu.T}


log.info("Getting foregrounds contribution")

fg_dict = best_fits.get_foreground_dict(l_th,
                                        passbands,
                                        fg_components,
                                        fg_params,
                                        fg_norm,
                                        band_shift_dict=band_shift_dict,
                                        beams=beams)

log.info("Writing best fit spectra")
spectra_list = pspipe_list.get_spec_name_list(d, delimiter = "_")
best_fit_dict = {}
for ps_name in spectra_list:
    fg = {}
    best_fit_dict[ps_name] = {}
    name1, name2 = ps_name.split("x")
    for spec in spectra:
        if spec.lower() in d["fg_components"].keys():
            fg[spec] = fg_dict[spec.lower(), "all", name1, name2]
        else:
            fg[spec] = fg_dict[spec.lower()[::-1], "all", name2, name1] # NOTE: differs from dr6, doesn't rely on T and P freq SED being same for a TP cross if also a name cross
        best_fit_dict[ps_name][spec] = ps_dict[spec] + fg[spec]
    so_spectra.write_ps(f"{bestfit_dir}/fg_{ps_name}.dat", l_th, fg, type, spectra=spectra)
    so_spectra.write_ps(f"{bestfit_dir}/cmb_and_fg_{ps_name}.dat", l_th, best_fit_dict[ps_name], type, spectra=spectra)

# check that fg and cmb correlation matrices are valid, since possible to enter
# invalid fgparams in paramfile (and maybe theory code for cmb can mess up)
ps_mat = np.zeros((1, 3, 1, 3, max(l_th) + 1))
fg_mat = np.zeros((len(map_set_list), 3, len(map_set_list), 3, max(l_th) + 1))
lidx = l_th.astype(int)
for p1, pol1 in enumerate('TEB'):
    for p2, pol2 in enumerate('TEB'):
        spec = pol1 + pol2 
        ps_mat[0, p1, 0, p2, lidx] = ps_dict[spec]
        for n1, name1 in enumerate(map_set_list):
            for n2, name2 in enumerate(map_set_list):
                try:
                    fg_mat[n1, p1, n2, p2, lidx] = fg_dict[spec.lower(), "all", name1, name2]
                except KeyError:
                    fg_mat[n1, p1, n2, p2, lidx] = fg_dict[spec.lower()[::-1], "all", name2, name1]

def is_mat_ok(mat):
    mat = mat.reshape(mat.shape[0] * mat.shape[1], mat.shape[2] * mat.shape[3], -1)

    # am i symmetric
    assert np.max(np.abs(np.nan_to_num(2 * (mat - np.moveaxis(mat, (0, 1), (1, 0))) / (mat + np.moveaxis(mat, (0, 1), (1, 0)))))) < 1e-12, \
        'Matrix is not symmetric'

    D = np.diagonal(mat).T**0.5
    assert not np.any(np.isnan(D))

    # am i a valid correlation matrix
    # NOTE: doesn't necessarily mean valid covariance matrix, would need to look
    # at evals for that. unfort they can be numerically negative...
    corr = np.nan_to_num((1/D[:, None]/D[None, :]) * mat)
    if not np.max(np.abs(corr)) < 1:
        assert np.allclose(np.max(np.abs(corr)), 1, rtol=0, atol=1e-12), \
            'Correlation matrix is not valid'

is_mat_ok(ps_mat)
is_mat_ok(fg_mat)

log.info("Plotting best fit spectra")
for spec in spectra:
    plt.figure(figsize=(12, 12))
    if spec == "TT":
        plt.semilogy()
    for ps_name in spectra_list:
        plt.plot(l_th, best_fit_dict[ps_name][spec], label=ps_name)
    plt.legend()
    plt.savefig(f"{plot_dir}/best_fit_{spec}.png")
    plt.clf()
    plt.close()

fg_components["tt"].remove("tSZ_and_CIB")
for comp in ["tSZ", "cibc", "tSZxCIB"]:
    fg_components["tt"].append(comp)

for mode in ["tt", "te", "tb", "ee", "eb", "bb"]:
    fig, axes = plt.subplots(narrays, narrays, sharex=True, sharey=True, figsize=(16, 16))
    axes = np.atleast_2d(axes)
    indices = np.triu_indices(narrays)[::-1]
    for i, cross in enumerate(spectra_list):
        name1, name2 = cross.split("x")
        idx = (indices[0][i], indices[1][i])
        ax = axes[idx]

        for comp in fg_components[mode]:
            ax.plot(l_th, fg_dict[mode, comp, name1, name2])
            np.savetxt(f"{components_dir}/{mode}_{comp}_{cross}.dat", np.transpose([l_th, fg_dict[mode, comp, name1, name2]]))
        ax.plot(l_th, fg_dict[mode, "all", name1, name2], color="k")
        ax.plot(l_th, ps_dict[mode.upper()], color="gray")
        ax.set_title(cross)
        if mode == "tt":
            ax.set_yscale("log")
            ax.set_ylim(1e-1, 1e4)
        if mode == "ee":
            ax.set_yscale("log")
        if mode == "bb":
            ax.set_yscale("log")

        if (mode[0] != mode[1]) and (name1 != name2):
            ax = axes[idx[::-1]]
            for comp in fg_components[mode]:
                ax.plot(l_th, fg_dict[mode, comp, name2, name1])
            ax.plot(l_th, fg_dict[mode, "all", name2, name1])
            ax.plot(l_th, ps_dict[mode.upper()], color="gray")
            ax.set_title(f"{name2}x{name1}")

    if mode[0] == mode[1]:
        for idx in zip(*np.triu_indices(narrays, k=1)):
            ax = axes[idx]
            fig.delaxes(ax)

    for i in range(narrays):
        axes[-1, i].set_xlabel(r"$\ell$")
        axes[i, 0].set_ylabel(r"$D_\ell$")
    fig.legend(fg_components[mode] + ["all"], title=mode.upper(), bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/foregrounds_all_comps_{mode}.png", dpi=300)
    plt.close()
