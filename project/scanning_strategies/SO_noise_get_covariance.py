"""
This script compute the covariance matrix corresponding to
the different scanning strategies.
We use both a master analytical computation and a simple approximation
"""

import pylab as plt
import numpy as np
from pspy import so_spectra, so_cov, so_mcm, pspy_utils, so_map, so_dict
import scipy.interpolate
import sys
import SO_noise_utils


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


scan_list = d["scan_list"]
lmax = d["lmax"]
niter = d["niter"]
spectra = d["spectra"]
split_list = d["split_list"]
runs = d["runs"]
spin_pairs = d["spin_pairs"]
binning_file = d["binning_file_name"]
clfile = d["clfile"]

lth, ps_theory = pspy_utils.ps_lensed_theory_to_dict(clfile, "Dl", lmax=lmax)

spectra_dir = "spectra"
plot_dir = "plot/covariance"
cov_dir = "covariance"
mcm_dir = "mcms"
window_dir = "windows"


pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(cov_dir)

fsky = {}
for scan in scan_list:
    for run in runs:
        print(scan, run)
    
        nl_th = {}
        
        for spec in ["TT", "EE"]:

            spec_name_00 = "%s_%sx%s_%s" % (scan, "split0", "split0", run)
            spec_name_11 = "%s_%sx%s_%s" % (scan, "split1", "split1", run)
            spec_name_10 = "%s_%sx%s_%s" % (scan, "split1", "split0", run)

            lb, Db_dict_00 = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name_00), spectra=spectra)
            lb, Db_dict_11 = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name_11), spectra=spectra)
            lb, Db_dict_10 = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name_10), spectra=spectra)

            nb = (Db_dict_00[spec] + Db_dict_11[spec])/2 - Db_dict_10[spec]
            
            #nb /= 2 # because we have two splits the effective noise is half the mean
            
            nl_th[spec] = scipy.interpolate.interp1d(lb, nb, fill_value = "extrapolate")
            nl_th[spec] = np.array([nl_th[spec](i) for i in lth])
            id = np.where(lth <= np.min(lb))
            nl_th[spec][id]= nl_th[spec][0]
            
            plt.figure()
            plt.semilogy()
            plt.plot(lb, nb, ".")
            plt.plot(lth, nl_th[spec], label="interpolate", color="lightblue")
            plt.legend(fontsize=20)
            plt.savefig("%s/noise_interpolate_%s_%s_%s.png" % (plot_dir, scan, run, spec), bbox_inches="tight")
            plt.clf()
            plt.close()
        
        nl_th["TE"] = np.zeros(len(lth))
        nl_th["ET"] = nl_th["TE"]
        
        survey_id = ["a", "b", "c", "d"]
        survey_name = ["split_0", "split_1", "split_0", "split_1"]

        name_list = []
        id_list = []
        for field in ["T", "E"]:
            for s, id in zip(survey_name, survey_id):
                name_list += ["%s%s" % (field, s)]
                id_list += ["%s%s" % (field, id)]

        Clth_dict = {}
        for name1, id1 in zip(name_list, id_list):
            for name2, id2 in zip(name_list, id_list):
                spec = id1[0] + id2[0]
                Clth_dict[id1 + id2] = ps_theory[spec] + nl_th[spec] * so_cov.delta2(name1, name2)
        
        window = so_map.read_map("%s/window_%s_%s.fits" % (window_dir, scan, run))
        
        mbb_inv, Bbl = so_mcm.read_coupling(prefix="%s/%s_%s" % (mcm_dir, scan, run), spin_pairs=spin_pairs)


        coupling_dict = so_cov.cov_coupling_spin0and2_simple(window, lmax, niter=niter, planck=False)
        analytic_cov = so_cov.cov_spin0and2(Clth_dict, coupling_dict, binning_file, lmax, mbb_inv, mbb_inv)
        
        fsky[scan, run], quick_cov = SO_noise_utils.quick_analytic_cov(lth, Clth_dict, window, binning_file, lmax)
        
        
        np.save("%s/analytic_cov_%s_%s.npy" % (cov_dir, scan, run), analytic_cov)
        np.save("%s/quick_cov_%s_%s.npy" % (cov_dir, scan, run), quick_cov)

for run in runs:
    print("")
    for scan in scan_list:
        print(run, scan, "%0.3f" % fsky[scan, run])
