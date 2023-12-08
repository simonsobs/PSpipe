"""
This script generates the data needed for the simple map2parameter simulation.
It generates the expected noise power spectra of the Simons Observatory
large aperture telescope and the expected Planck white noise power spectra and write them to disk
in formal: ell, n_ell. Note that SO has correlated noise between frequency channels.
It also generates beam files for SO and Planck with format: ell, b_ell.
It generates a theoretical power spectra from camb and compare it to noise power spectra and foreground power spectra.
Finally it generates a binning_file for SO with format : bin_min, bin_max, bin_mean.

The code makes use of the SO noise calculator: so_noise_calculator_public_20180822.
"""
from pspy import pspy_utils, so_dict, so_spectra, so_map, so_window
from itertools import combinations_with_replacement as cwr
import so_noise_calculator_public_20180822 as noise_calc
from pspipe_utils import pspipe_list, best_fits
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import camb
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

type = d["type"]
surveys = d["surveys"]

# Define and create the directories to save inputs for the simulations
windows_dir = d["windows_dir"]
bestfit_dir = "best_fits"
noise_model_dir = "noise_model"
plot_dir = "plots"
beam_dir = d["beam_dir"]
binning_dir = d["binning_dir"]
passband_dir = d["passband_dir"]

pspy_utils.create_directory(windows_dir)
pspy_utils.create_directory(bestfit_dir)
pspy_utils.create_directory(noise_model_dir)
pspy_utils.create_directory(beam_dir)
pspy_utils.create_directory(binning_dir)
pspy_utils.create_directory(passband_dir)

# Linestyles used to make the difference between LAT and Planck in the plots
linestyle = {
    "LAT": "solid",
    "Planck": "dashed"
}

# Get survey, array lists & spectra name list
n_arrays, sv_list, ar_list = pspipe_list.get_arrays_list(d)
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter = "_")
sv_ar_list = [f"{sv}_{ar}" for sv, ar in zip(sv_list, ar_list)]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

# We start with SO, we have to specify a sensitivity mode (2: goal, 1: baseline), and f_sky
# both parameters are specified in the dictionnary
sensitivity_mode = d["sensitivity_mode"]
f_sky_LAT = d["f_sky_LAT"]

ell_min, ell_max = 2, d["lmax"] + 500
delta_ell = 1
# We use the SO noise calculator to compute the expected noise in temperature and polarisation.
ell, n_ell_t_LAT, n_ell_pol_LAT, map_wn = noise_calc.Simons_Observatory_V3_LA_noise(sensitivity_mode,
                                                                                    f_sky_LAT,
                                                                                    ell_min,
                                                                                    ell_max,
                                                                                    delta_ell,
                                                                                    N_LF=1.,
                                                                                    N_MF=4.,
                                                                                    N_UHF=2.,
                                                                                    apply_beam_correction=False,
                                                                                    apply_kludge_correction=True)

# Initialize the dictionnaries with zeros
n_ell_t = {spec_name: np.zeros_like(ell, dtype = float) for spec_name in spec_name_list}
n_ell_pol = deepcopy(n_ell_t)

# Create a dict storing the noise computed with the SO noise calculator
f_pairs_LAT = [
    "LAT_27xLAT_27",
    "LAT_39xLAT_39",
    "LAT_93xLAT_93",
    "LAT_145xLAT_145",
    "LAT_225xLAT_225",
    "LAT_280xLAT_280",
    "LAT_27xLAT_39",
    "LAT_93xLAT_145",
    "LAT_225xLAT_280"
]
n_ell_t_LAT_dict = {f_pair: n_ell_t_LAT[i] for i,f_pair in enumerate(f_pairs_LAT)}
n_ell_pol_LAT_dict = {f_pair: n_ell_pol_LAT[i] for i,f_pair in enumerate(f_pairs_LAT)}

# Fill the dict with non-zero noise power
for f_pair_LAT in f_pairs_LAT:
    if f_pair_LAT in spec_name_list:
        n_ell_t[f_pair_LAT] = n_ell_t_LAT_dict[f_pair_LAT]
        n_ell_pol[f_pair_LAT] = n_ell_pol_LAT_dict[f_pair_LAT]

# Now let's go to Planck we will use information from the Table 4 of
# https://arxiv.org/pdf/1807.06205.pdf
# Planck noise will be assumed to be white for these simulations
# we give Planck standard deviations in uk.arcmin
f_pairs_Planck = [
    "Planck_100xPlanck_100",
    "Planck_143xPlanck_143",
    "Planck_217xPlanck_217",
    "Planck_353xPlanck_353"
]

sigma_t = {
    "Planck_100xPlanck_100": 77.4,
    "Planck_143xPlanck_143": 33.0,
    "Planck_217xPlanck_217": 46.80,
    "Planck_353xPlanck_353": 153.6
}

sigma_pol = {
    "Planck_100xPlanck_100": 117.6,
    "Planck_143xPlanck_143": 70.2,
    "Planck_217xPlanck_217": 105.0,
    "Planck_353xPlanck_353": 438.6
}

n_ell_t_Planck_dict = {}
n_ell_pol_Planck_dict = {}
for f_pair_Planck in f_pairs_Planck:
    sigma_t_rad = np.deg2rad(sigma_t[f_pair_Planck]) / 60
    sigma_pol_rad = np.deg2rad(sigma_pol[f_pair_Planck]) / 60

    n_ell_t_Planck_dict[f_pair_Planck] = np.full_like(ell, sigma_t_rad ** 2, dtype = float)
    n_ell_pol_Planck_dict[f_pair_Planck] = np.full_like(ell, sigma_pol_rad ** 2, dtype = float)

# Fill the dict with non-zero noise power
for f_pair_Planck in f_pairs_Planck:
    if f_pair_Planck in spec_name_list:
        n_ell_t[f_pair_Planck] = n_ell_t_Planck_dict[f_pair_Planck]
        n_ell_pol[f_pair_Planck] = n_ell_pol_Planck_dict[f_pair_Planck]

# Now let's write the n_ell_t and n_ell_pol dictionnary to disk
# Note that we are creating a lot of small files, we could use another data format
fac = ell * (ell + 1) / (2 * np.pi) if d["type"] == "Dl" else 1.

nlth_dict = {}
for sv in surveys:
    for id1, ar1 in enumerate(d[f"arrays_{sv}"]):
        for id2, ar2 in enumerate(d[f"arrays_{sv}"]):
            if id1 > id2: continue
            spec_name = f"{sv}_{ar1}x{sv}_{ar2}"
            mean_noise_t = n_ell_t[spec_name] * fac
            mean_noise_p = n_ell_pol[spec_name] * fac
            mean_noise = {"TT": mean_noise_t, "EE": mean_noise_p, "BB": mean_noise_p}

            # apply a regularization to the noise power spectrum at very low ell
            l_cut_noise = d["l_cut_noise_LAT"]
            for spec in spectra:
                if spec in ["TT", "EE", "BB"]:
                    mean_noise[spec] = np.where(ell <= l_cut_noise, 0., mean_noise[spec])
                else:
                    mean_noise[spec] = np.zeros_like(ell, dtype = float)
            nlth_dict[spec_name] = mean_noise

            so_spectra.write_ps(f"{noise_model_dir}/mean_{ar1}x{ar2}_{sv}_noise.dat", ell, mean_noise, type, spectra = spectra)
            so_spectra.write_ps(f"{noise_model_dir}/mean_{ar2}x{ar1}_{sv}_noise.dat", ell, mean_noise, type, spectra = spectra)

# Finally let's generate the beam harmonic transform for Planck and SO LAT
# For Planck we  we will use information from the Table 4 of
# https://arxiv.org/pdf/1807.06205.pdf
# For SO we use info from Table 1 of
# https://arxiv.org/pdf/1808.07445.pdf
beam_plot_dir = f"{plot_dir}/beams"
pspy_utils.create_directory(beam_plot_dir)

beam_fwhm = {
    "LAT_27": 7.4,
    "LAT_39": 5.1,
    "LAT_93": 2.2,
    "LAT_145": 1.4,
    "LAT_225": 1.0,
    "LAT_280": 0.9,
    "Planck_100": 9.68,
    "Planck_143": 7.30,
    "Planck_217": 5.02,
    "Planck_353": 4.94
}

l = np.arange(ell_max + 1000)
bl = {sv_ar: np.ones(ell_max + 1000) for sv_ar in sv_ar_list}
plt.figure(figsize=(12, 12))
for sv_ar in sv_ar_list:
    _, bl_sv_ar = pspy_utils.beam_from_fwhm(beam_fwhm[sv_ar], ell_max+1000)
    bl[sv_ar][2:] = bl_sv_ar

    np.savetxt(d[f"beam_{sv_ar}"], np.array([l, bl[f"{sv_ar}"]]).T)

    plt.plot(l, bl[f"{sv_ar}"], ls = linestyle[sv], label = f"{sv_ar}")
    plt.xlabel(r"$\ell$", fontsize = 22)
    plt.ylabel(r"$b_{\ell}$", fontsize = 22)
plt.legend()
plt.savefig(f"{beam_plot_dir}/beams_plot.pdf")
plt.clf()
plt.close()

# Let's compare the signal power spectra with the noise power spectra
# We generate the signal power spectra using camb and write it to disk
cosmo_params =  d["cosmo_params"]
l_th, ps_dict = pspy_utils.ps_from_params(cosmo_params, d["type"], ell_max, **d["accuracy_params"])
f_name = f"{bestfit_dir}/cmb.dat"
so_spectra.write_ps(f_name, l_th, ps_dict, type, spectra=spectra)

# Plot noise power spectra
output_plot_noise = f"{plot_dir}/noise_model"
pspy_utils.create_directory(output_plot_noise)
for spec in ["TT", "EE"]:
    plt.figure(figsize = (10, 8))
    plt.plot(l_th, ps_dict[spec], color = "k")
    plt.xlabel(r"$\ell$")
    plt.ylabel(r"$N_\ell^\mathrm{%s}$" % spec)
    for sv in surveys:
        for id1, ar1 in enumerate(d[f"arrays_{sv}"]):
            for id2, ar2 in enumerate(d[f"arrays_{sv}"]):
                if id1 > id2: continue
                spec_name = f"{sv}_{ar1}x{sv}_{ar2}"
                n_ell = nlth_dict[spec_name]

                plt.plot(ell, n_ell[spec] / (bl[f"{sv}_{ar1}"] * bl[f"{sv}_{ar2}"])[2:ell_max], ls = linestyle[sv],
                         label = spec_name)
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_plot_noise}/noise_ps_all_{spec}.pdf")

# We now compare the signal power spectra with foreground power spectra
fg_norm = d["fg_norm"]
fg_components = d["fg_components"]
fg_params =  d["fg_params"]
do_bandpass_integration = d["do_bandpass_integration"]

res = 1. #GHz
passbands = {}

for sv_ar in sv_ar_list:
    freq_info = d[f"freq_info_{sv_ar}"]
    if do_bandpass_integration:
        central_nu, delta_nu_rel = freq_info["freq_tag"], d[f"bandwidth_{sv_ar}"]
        nu_min, nu_max = central_nu * (1 - delta_nu_rel), central_nu * (1 + delta_nu_rel)
        nu_ghz = np.linspace(nu_min, nu_max, int((nu_max-nu_min)/res))

        bp = np.where(nu_ghz > central_nu * (1 + delta_nu_rel * 0.5), 0.,
                      np.where(nu_ghz < central_nu * (1 - delta_nu_rel * 0.5), 0., 1.))

        # remove the frequencies for which the passband is zero
        # this is useful in the case of top-hat passbands in particular
        nu_ghz, bp = nu_ghz[bp != 0.], bp[bp != 0.]
        np.savetxt(freq_info["passband"], np.array([nu_ghz, bp]).T)

    else:
        nu_ghz, bp = np.array([freq_info["freq_tag"]]), np.array([1])

    passbands[f"{sv_ar}"] = [nu_ghz, bp]

fg_dict = best_fits.get_foreground_dict(ell, passbands, fg_components,
                                        fg_params, fg_norm=fg_norm)

fg = {}
for sv1_ar1 in sv_ar_list:
    for sv2_ar2 in sv_ar_list:
        fg[sv1_ar1, sv2_ar2] = {}
        for spec in spectra:
            fg[sv1_ar1, sv2_ar2][spec] = fg_dict[spec.lower(), "all", sv1_ar1, sv2_ar2]

        so_spectra.write_ps(f"{bestfit_dir}/fg_{sv1_ar1}x{sv2_ar2}.dat", ell, fg[sv1_ar1, sv2_ar2], type, spectra=spectra)

# Create binning file
n_bins = 200
bin_size = np.zeros(n_bins)
bin_size[0] = 50
bin_size[1:80] = 35
bin_size[80:100] = 60
bin_size[100:200] = 100

bin_min = 2
binning = np.zeros((n_bins, 3))

for i in range(n_bins):
    bin_max = bin_min + bin_size[i]
    bin_mean = (bin_min + bin_max) / 2
    binning[i, :] = bin_min, bin_max, bin_mean
    bin_min += bin_size[i] + 1
np.savetxt(d["binning_file"], binning)

# Generate windows functions
windows_plot_dir = f"{plot_dir}/windows"
pspy_utils.create_directory(windows_plot_dir)

for sv in surveys:
    if d[f"pixel_{sv}"] == "CAR":
        binary = so_map.car_template(ncomp = 1,
                                     ra0 = d[f"ra0_{sv}"],
                                     ra1 = d[f"ra1_{sv}"],
                                     dec0 = d[f"dec0_{sv}"],
                                     dec1 = d[f"dec1_{sv}"],
                                     res = d[f"res_{sv}"])
        binary.data[:] = 1
        if d[f"binary_is_survey_mask_{sv}"]:
            binary.data[:] = 0
            binary.data[1:-1, 1:-1] = 1

    elif d[f"pixel_{sv}"] == "HEALPIX":
        binary = so_map.healpix_template(ncomp = 1,
                                         nside = d[f"nside_{sv}"])
        binary.data[:] = 1

    for ar in d[f"arrays_{sv}"]:
        window = binary.copy()

        if d[f"include_galactic_mask_{sv}"]:
            gal_mask = so_map.read_map(d[f"gal_mask_{sv}_{ar}"])
            gal_mask.plot(file_name = f"{windows_plot_dir}/galactic_mask_{sv}_{ar}")
            window.data[:] *= gal_mask.data[:]

        if not d[f"binary_is_survey_mask_{sv}"]:
            survey_mask = so_map.read_map(d[f"survey_mask_{sv}_{ar}"])
            survey_mask.plot(file_name = f"{windows_plot_dir}/survey_mask_{sv}_{ar}")
            window.data[:] *= survey_mask.data[:]

        apod_survey_degree = d[f"apod_survey_degree_{sv}"]
        apod_type_survey = d[f"apod_type_survey_{sv}"]

        window = so_window.create_apodization(window, apo_type = apod_type_survey,
                                              apo_radius_degree = apod_survey_degree)

        if d[f"include_pts_source_mask_{sv}"]:
            hole_radius_arcmin = d[f"pts_source_mask_radius_{sv}"]
            n_holes = d[f"pts_source_mask_nholes_LAT"]
            apod_type_pts_source_mask = d[f"apod_type_pts_source_mask_{sv}"]
            apod_pts_source_mask_degree = d[f"apod_pts_source_mask_degree_{sv}"]

            mask = so_map.simulate_source_mask(binary, n_holes = n_holes, hole_radius_arcmin = hole_radius_arcmin)
            mask = so_window.create_apodization(mask, apo_type = apod_type_pts_source_mask,
                                                apo_radius_degree = apod_pts_source_mask_degree)
            window.data[:] *= mask.data[:]

        window.write_map(d[f"window_T_{sv}_{ar}"])
        window.plot(file_name = f"{windows_plot_dir}/window_{sv}_{ar}")
