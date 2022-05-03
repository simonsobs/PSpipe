# for reference the script to pickle the data 
import pickle
import numpy as np
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils

surveys = ["sv1", "sv2"]
arrays = {}
arrays["sv1"] = ["pa1", "pa2"]
arrays["sv2"]= ["pa3"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


spec_name = []
for id_sv1, sv1 in enumerate(surveys):
    for id_ar1, ar1 in enumerate(arrays[sv1]):
        for id_sv2, sv2 in enumerate(surveys):
            for id_ar2, ar2 in enumerate(arrays[sv2]):
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                spec_name += ["%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)]
data = {}
for sid1, spec1 in enumerate(spec_name):

    for spin in spin_pairs:
        data["mcm", spec1, spin] = np.load("%s_mbb_inv_%s.npy" % (spec1, spin))

    l_ref, data["spectra", spec1]  = so_spectra.read_ps("Dl_%s_cross.dat" % spec1, spectra=spectra)

    for sid2, spec2 in enumerate(spec_name):
        if sid1 > sid2: continue

        data["analytic_cov", spec1, spec2] = np.load("analytic_cov_%s_%s.npy" % (spec1, spec2))
    
f = open("trial_data.pkl","wb")
pickle.dump(data,f)

f1 = open("trial_data.pkl", "rb")
data2 = pickle.load(f1)

print(data2)
