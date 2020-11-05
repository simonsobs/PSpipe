"""
This script compute best fit from theory and fg power spectra.
It uses camb and the foreground model of mflike based on fgspectra
"""
import sys

import numpy as np
import pylab as plt
from pspy import pspy_utils, so_dict

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# first let's get a list of all frequency we plan to study
surveys = d["surveys"]
lmax = d["lmax"]

freq_list = []
for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for ar in arrays:
        freq_list += [d["nu_eff_%s_%s" % (sv, ar)]]

# remove doublons
freq_list = list(dict.fromkeys(freq_list))

# let's create the directories to write best fit to disk and for plotting purpose
bestfit_dir = "best_fits"
plot_dir = "plots/best_fits/"

pspy_utils.create_directory(bestfit_dir)
pspy_utils.create_directory(plot_dir)

# now we use camb to produce best fit power spectrum, we will use CAMB to do so with standard LCDM params
import camb
ell_min, ell_max = 2, lmax + 500
cosmo_params = d["cosmo_params"]
camb_cosmo = {k: v for k, v in cosmo_params.items() if k not in ["logA", "As"]}
camb_cosmo.update({"As": 1e-10*np.exp(cosmo_params["logA"]), "lmax": ell_max, "lens_potential_accuracy": 1})
pars = camb.set_params(**camb_cosmo)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
clth = {}
for count, spec in enumerate(["TT", "EE", "BB", "TE" ]):
    clth[spec] = powers["total"][ell_min:ell_max][:,count]
clth["ET"] = clth["TE"]
for spec in ["TB", "BT", "EB", "BE" ]:
    clth[spec] = clth["TT"] * 0
    
    
# we will now use mflike (and in particular the fg module) to get the best fit foreground model
# we will only include foreground in tt, note that for now only extragalactic foreground are present
import mflike as mfl

fg_norm = d["fg_norm"]
fg_components =  d["fg_components"]
components = {"tt": fg_components, "ee": [], "te": []}
fg_model =  {"normalisation":  fg_norm, "components": components}
fg_params = d["fg_params"]
ell = np.arange(ell_min, ell_max)
fg_dict = mfl.get_foreground_model(fg_params, fg_model, freq_list, ell)

spectra = ["TT", "TE", "EE", "EB", "BE", "BB"]

for spec in spectra:
    plt.figure(figsize=(12,12))
    for id_sv1, sv1 in enumerate(surveys):
        arrays_1 = d["arrays_%s" % sv1]
        for id_ar1, ar1 in enumerate(arrays_1):
            nu_eff1 = d["nu_eff_%s_%s" % (sv1, ar1)]
            for id_sv2, sv2 in enumerate(surveys):
                arrays_2 = d["arrays_%s" % sv2]
                for id_ar2, ar2 in enumerate(arrays_2):
                    nu_eff2 = d["nu_eff_%s_%s" % (sv2, ar2)]
                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue
                    
                    cl_th_and_fg = clth[spec]
                    if spec == "TT":
                        plt.semilogy()
                        cl_th_and_fg = cl_th_and_fg + fg_dict["tt", "all", nu_eff1, nu_eff2]
                        
                    name = "%s_%sx%s_%s_%s" % (sv1, ar1, sv2, ar2, spec)
                    
                    print(name)
                    
                    np.savetxt("%s/best_fit_%s.dat" % (bestfit_dir, name),
                                np.transpose([ell, cl_th_and_fg]))

                    
                    plt.plot(ell, cl_th_and_fg, label= "%s %s x %s %s" %(sv1, ar1, sv2, ar2) )
    plt.legend()
    plt.savefig("%s/best_fit_%s.png" % (plot_dir, spec))
    plt.clf()
    plt.close()
