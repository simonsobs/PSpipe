"""
This script generates the data needed for the simple map2parameter simulation.
It generates the expected noise power spectra of the Simons Observatory
large aperture telescope and the expeted Planck white noise power spectra and write them to disk
in formal: ell, n_ell. Note that SO has correlated noise between frequency channels.
It also generates beam files for SO and Planck with format: ell, b_ell.
It generates a theoretical power spectra from camb and compare it to noise power spectra and foreground power spectra.
Finally it generates a binning_file for SO with format : bin_min, bin_max, bin_mean.

The code makes use of the SO noise calculator: so_noise_calculator_public_20180822.
"""
#import matplotlib
#matplotlib.use("Agg")
from pspy import pspy_utils, so_dict
import numpy as np
import pylab as plt
from itertools import combinations_with_replacement as cwr
import os
import sys
import so_noise_calculator_public_20180822 as noise_calc
from copy import deepcopy

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

pspy_utils.create_directory("sim_data")

plot_dir = "plots/model/"
pspy_utils.create_directory(plot_dir)

linestyle = {}
linestyle["LAT"] = "solid"
linestyle["Planck"] = "dashed"

# We start with SO, we have to specify a sensitivity mode (2: goal, 1: baseline), and f_sky
# both parameters are specified in the dictionnary

sensitivity_mode = d["sensitivity_mode"]
f_sky_LAT = d["f_sky_LAT"]
freqs = {}
freqs["LAT"] = ["27", "39", "93", "145", "225", "280"]
ell_min, ell_max = 2, 9000
delta_ell = 1

pspy_utils.create_directory("sim_data/noise_ps")

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
n_ell_t = {"LAT_{}xLAT_{}".format(*cross): ell * 0. for cross in cwr(freqs["LAT"], 2)}
n_ell_pol = deepcopy(n_ell_t)

# We fill with non zeros he different frequency pairs considered in the SO noise calculator

f_pairs_LAT = ["LAT_27xLAT_27",
               "LAT_39xLAT_39",
               "LAT_93xLAT_93",
               "LAT_145xLAT_145",
               "LAT_225xLAT_225",
               "LAT_280xLAT_280",
               "LAT_27xLAT_39",
               "LAT_93xLAT_145",
               "LAT_225xLAT_280"]

for i, f_pair in enumerate(f_pairs_LAT):
    n_ell_t[f_pair] = n_ell_t_LAT[i]
    n_ell_pol[f_pair] = n_ell_pol_LAT[i]

# Now let's go to Planck we will use information from the Table 4 of
# https://arxiv.org/pdf/1807.06205.pdf
# Planck noise will be assumed to be white for these simulations
# we give Planck standard deviations in uk.arcmin

freqs["Planck"] = ["100", "143", "217", "353"]
f_pairs_Planck = ["Planck_{}xPlanck_{}".format(*cross) for cross in cwr(freqs["Planck"], 2)]
sigma = {"Planck_{}xPlanck_{}".format(*cross): 0. for cross in cwr(freqs["Planck"], 2)}
sigma_pol = deepcopy(sigma)

sigma["Planck_100xPlanck_100"] = 77.4
sigma["Planck_143xPlanck_143"] = 33.0
sigma["Planck_217xPlanck_217"] = 46.80
sigma["Planck_353xPlanck_353"] = 153.6

sigma_pol["Planck_100xPlanck_100"] = 117.6
sigma_pol["Planck_143xPlanck_143"] = 70.2
sigma_pol["Planck_217xPlanck_217"] = 105.0
sigma_pol["Planck_353xPlanck_353"] = 438.6

for f_pair in f_pairs_Planck:
    sigma_rad = np.deg2rad(sigma[f_pair]) / 60
    n_ell_t[f_pair] = ell * 0 + sigma_rad**2
    sigma_pol_rad = np.deg2rad(sigma_pol[f_pair]) / 60
    n_ell_pol[f_pair] = ell * 0 + sigma_pol_rad**2

# Now let's write the n_ell_t and n_ell_pol dictionnary to disk
# Note that we are creating a lot of small files, we could use another data format

for exp in ["LAT", "Planck"]:
    my_freqs = freqs[exp]
    for cross in cwr(my_freqs, 2):
        f1, f2 = cross
        name = "%s_%sx%s_%s"%(exp,f1,exp,f2)
        print (name)
        np.savetxt("sim_data/noise_ps/noise_t_%s.dat"%(name), np.transpose([ell,n_ell_t[name]]))
        np.savetxt("sim_data/noise_ps/noise_pol_%s.dat"%(name), np.transpose([ell,n_ell_pol[name]]))


# Finally let's generate the beam harmonic transform for Planck and SO LAT
# For Planck we  we will use information from the Table 4 of
# https://arxiv.org/pdf/1807.06205.pdf
# For SO we use info from Table 1 of
# https://arxiv.org/pdf/1808.07445.pdf

pspy_utils.create_directory("sim_data/beams")

beam_fwhm = {}
beam_fwhm["LAT_27"] = 7.4
beam_fwhm["LAT_39"] = 5.1
beam_fwhm["LAT_93"] = 2.2
beam_fwhm["LAT_145"] = 1.4
beam_fwhm["LAT_225"] = 1.0
beam_fwhm["LAT_280"] = 0.9

beam_fwhm["Planck_100"] = 9.68
beam_fwhm["Planck_143"] = 7.30
beam_fwhm["Planck_217"] = 5.02
beam_fwhm["Planck_353"] = 4.94

bl = {}
plt.figure(figsize=(12, 12))
for exp in ["LAT", "Planck"]:
    my_freqs = freqs[exp]
    for f in my_freqs:
        l, bl[exp + f] = pspy_utils.beam_from_fwhm(beam_fwhm[exp + '_' + f], ell_max)
        np.savetxt("sim_data/beams/beam_%s_%s.dat" % (exp,f), np.transpose([l, bl[exp + f] ]))

        plt.plot(l, bl[exp + f] , linestyle=linestyle[exp], label="%s_%s" % (exp, f))
        plt.xlabel(r"$\ell$", fontsize=22)
        plt.ylabel(r"$b_{\ell}$", fontsize=22)
plt.legend()
plt.savefig("%s/beams_plot.pdf"%plot_dir)
plt.clf()
plt.close()

# Let's compare the signal power spectra with the noise power spectra
# We generate the signal power spectra using camb and write it to disk

import camb

# Some standard cosmo parameters
cosmo_params =  d["cosmo_params"]
camb_cosmo = {k: v for k, v in cosmo_params.items()
              if k not in ["logA", "As"]}
camb_cosmo.update({"As": 1e-10*np.exp(cosmo_params["logA"]),
                   "lmax": ell_max,
                   "lens_potential_accuracy": d["lens_potential_accuracy"],
                   "lens_margin": d["lens_margin"],
                   "AccuracyBoost": d["AccuracyBoost"],
                   "lSampleBoost": d["lSampleBoost"],
                   "lAccuracyBoost": d["lAccuracyBoost"]})
pars = camb.set_params(**camb_cosmo)
results = camb.get_results(pars)


powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
dls = {}
dls["tt"] = powers["total"][ell_min:ell_max][:,0]
dls["ee"] = powers["total"][ell_min:ell_max][:,1]
dls["bb"] = powers["total"][ell_min:ell_max][:,2]
dls["te"] = powers["total"][ell_min:ell_max][:,3]


cosmo_fg_dir=("sim_data/cosmo_and_fg")
pspy_utils.create_directory(cosmo_fg_dir)
np.savetxt("%s/cosmo_spectra.dat"% cosmo_fg_dir, np.transpose([ell, dls["tt"], dls["ee"], dls["bb"], dls["te"]]))

fig, ax = plt.subplots(2, 1, figsize = (12, 12))
for exp in ["LAT", "Planck"]:
    my_freqs = freqs[exp]
    for cross in cwr(my_freqs, 2):
        f1, f2 = cross
        name = "%s_%sx%s_%s"%(exp,f1,exp,f2)

        fac = ell * (ell + 1) / (2 * np.pi)

        if (len(np.nonzero(n_ell_t[name])[0])) != 0:  # plot only the spectra that are non zero

            ax[0].set_yscale("symlog")
            ax[0].set_ylim(1, 10**5)
            ax[0].plot(ell, dls["tt"], color = "black")
            ax[0].plot(ell, n_ell_t[name] * fac / (bl[exp + f1] * bl[exp + f2]),
                     linestyle=linestyle[exp],
                     label="%s" % (name))
            ax[0].set_xlabel(r"$\ell$", fontsize=22)
            ax[0].set_ylabel(r"$N^{T}_{\ell}$", fontsize=22)
            ax[1].set_yscale("symlog")
            ax[1].set_ylim(5e-2, 1e3)
            ax[1].plot(ell, dls["ee"], color = "black")
            ax[1].plot(ell, n_ell_pol[name] * fac / (bl[exp + f1] * bl[exp + f2]),
                     linestyle=linestyle[exp],
                     label="%s" % (name))
            ax[1].set_xlabel(r"$\ell$", fontsize=22)
            ax[1].set_ylabel(r"$N^{P}_{\ell}$", fontsize=22)


ax[0].legend()
plt.savefig("%s/noise_ps_plot.pdf"%plot_dir)
plt.close()


# We now compare the signal power spectra with foreground power spectra

experiments = d["experiments"]

fg_norm = d["fg_norm"]
fg_components = d["fg_components"]
fg_model =  {"normalisation":  fg_norm, "components": fg_components}
fg_params =  d["fg_params"]
nuis_params = d["nuisance_params"]
band_integration = d["band_integration"]

all_freqs = np.array([int(freq) for exp in experiments for freq in d["freqs_%s" % exp]])
nfreqs = len(all_freqs)

from mflike import theoryforge_MFLike as th_mflike

ThFo = th_mflike.TheoryForge_MFLike()
ThFo.freqs = all_freqs
ThFo.bandint_nsteps = band_integration["nsteps"]
ThFo.bandint_width = band_integration["bandwidth"]
ThFo.bandint_external_bandpass = band_integration["external_bandpass"]

ThFo.foregrounds = fg_model
ThFo._init_foreground_model()

#perform band integration if needed (only if bandpass not external)
ThFo.bandint_freqs = ThFo._bandpass_construction(**nuis_params)

fg_dict = ThFo._get_foreground_model(ell = ell, **fg_params)



# Plot separated components for tSZ and CIB
fg_components["tt"].remove("tSZ_and_CIB")
for comp in ["tSZ", "cibc", "tSZxCIB"]:
    fg_components["tt"].append(comp)

for mode in ["tt", "te", "ee"]:
    fig, axes = plt.subplots(nfreqs, nfreqs, sharex=True, sharey=True, figsize=(10, 10))
    from itertools import product
    for i, cross in enumerate(product(all_freqs, all_freqs)):
        f0, f1 = cross
        f0, f1 = int(f0), int(f1)
        idx = (i%nfreqs, i//nfreqs)
        ax = axes[idx]
        if idx in zip(*np.triu_indices(nfreqs, k=1)) and mode != "te":
            fig.delaxes(ax)
            continue
        for compo in fg_model["components"][mode]:
            ax.plot(l, fg_dict[mode, compo, f0, f1])
            np.savetxt("%s/%s_%s_%dx%d.dat" % (cosmo_fg_dir, mode, compo, f0, f1),
                       np.transpose([l,fg_dict[mode, compo, f0, f1]]))

        ax.plot(l, fg_dict[mode, "all", f0, f1], color="k")
        ax.plot(l, dls[mode], color="gray")
        np.savetxt("%s/%s_%s_%dx%d.dat" % (cosmo_fg_dir, mode, "all", f0, f1),
                   np.transpose([l,fg_dict[mode, "all", f0, f1]]))

        ax.legend([], title="{}x{} GHz".format(*cross))
        if mode == "tt":
            ax.set_yscale("log")
            ax.set_ylim(10**-1, 10**4)
        if mode == "ee":
            ax.set_yscale("log")

    for i in range(nfreqs):
        axes[-1, i].set_xlabel("$\ell$")
        axes[i, 0].set_ylabel("$D_\ell$")
    fig.legend(fg_model["components"][mode] + ["all"], title=mode.upper(), bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.savefig("%s/foregrounds_%s.pdf"%(plot_dir,mode))
    plt.close()



# Create binning file

pspy_utils.create_directory("sim_data/binning")
n_bins = 200
bin_size = np.zeros(n_bins)
bin_size[0] = 50
bin_size[1:80] = 35
bin_size[80:100] = 60
bin_size[100:200] = 100

bin_min = 2

g = open("sim_data/binning/binning.dat", mode="w")
for i in range(n_bins):
    bin_max = bin_min + bin_size[i]
    bin_mean = (bin_min + bin_max) / 2
    g.write("%f %f %f\n" % (bin_min, bin_max, bin_mean))
    bin_min += bin_size[i] + 1
g.close()
