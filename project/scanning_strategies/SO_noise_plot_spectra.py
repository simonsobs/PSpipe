"""
This script plot and compare the different noise power spectra corresponding to
the different scanning strategies
"""

import matplotlib
matplotlib.use("Agg")
import pylab as plt
import numpy as np
import sys
from pspy import so_spectra, pspy_utils, so_dict

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

scan_list = d["scan_list"]
spectra = d["spectra"]
split_list = d["split_list"]
runs = d["runs"]
lmax = d["lmax"]
clfile = d["clfile"]

            
lth, ps_theory = pspy_utils.ps_lensed_theory_to_dict(clfile, "Dl", lmax=lmax)

spectra_dir = "spectra"
plot_dir = "plot/spectra"

pspy_utils.create_directory(plot_dir)

Db_dict = {}
Db_auto_dict = {}
Db_cross_dict = {}

for scan in scan_list:
    for run in runs:
        for spec in spectra:
            Db_auto_dict[scan, run, spec] = []
            Db_cross_dict[scan, run, spec] = []

        for c0, s0 in enumerate(split_list):
            for c1, s1 in enumerate(split_list):
                if c1 > c0: continue
                spec_name = "%s_%sx%s_%s" % (scan, s0, s1, run)
                
                lb, Db_dict[spec_name] = so_spectra.read_ps("%s/spectra_%s.dat" % (spectra_dir, spec_name), spectra=spectra)
                
                for spec in spectra:
                    if s0 == s1:
                        Db_auto_dict[scan, run, spec] += [Db_dict[spec_name][spec]]
                    else:
                        Db_cross_dict[scan, run, spec] += [Db_dict[spec_name][spec]]
                        
        for spec in spectra:
            Db_auto_dict[scan, run, spec] = np.mean(Db_auto_dict[scan, run, spec], axis=0)
            Db_cross_dict[scan, run, spec] = np.mean(Db_cross_dict[scan, run, spec], axis=0)

# look at the split per split properties
colors = ["blue", "green", "orange", "red", "grey", "lightblue", "magenta"]
for run in runs:
    plt.figure(figsize = (25, 10))
    for c, spec in enumerate(["TT", "EE", "BB"]):
        plt.subplot(1, 4, c + 1)
        plt.semilogy()
        plt.plot(lth, ps_theory[spec])
        for c_scan, scan in enumerate(scan_list):
            fmt = ["-", "--"]
            for count, split in enumerate(split_list):
            
                spec_name = "%s_%sx%s_%s" % (scan, split, split, run)
                spec_name_cross = "%s_%sx%s_%s" % (scan, "split1", "split0", run)
                
                nb = Db_dict[spec_name][spec] - Db_dict[spec_name_cross][spec]
                
                plt.plot(lb, nb, fmt[count], color=colors[c_scan], label = r" %s %s" % (scan, split))
        if c == 2:
            plt.legend(bbox_to_anchor=(1.05, 1))
        
        plt.title(r"$D^{%s}_{\ell}$" % spec, fontsize=14)
        plt.xlabel(r"$\ell$", fontsize=14)
    plt.suptitle(run)
    plt.savefig("%s/noise_spec_%s.png" % (plot_dir, run), bbox_inches="tight")
    plt.clf()
    plt.close()
    
    
# look at weight vs unweight
for scan in scan_list:
    plt.figure(figsize = (25, 10))
    for c, spec in enumerate(["TT", "EE", "BB"]):
        plt.subplot(1, 4, c + 1)
        plt.semilogy()
        plt.plot(lth, ps_theory[spec])
        for run in runs:
            noise_ps = (Db_auto_dict[scan, run, spec] - Db_cross_dict[scan, run, spec])/2
            plt.plot(lb, noise_ps, label = r" %s %s" % (scan, run))
            np.savetxt("%s/noise_%s_%s_%s.dat" % (spectra_dir, scan, run, spec), np.transpose([lb, noise_ps]))
    plt.legend()
    plt.savefig("%s/noise_spec_%s.png" % (plot_dir, scan), bbox_inches="tight")
    plt.clf()
    plt.close()
    
# Check if the cross is unbiased
for scan in scan_list:
    plt.figure(figsize = (25, 10))
    for c, spec in enumerate(["TT", "EE", "BB"]):
        plt.subplot(1, 4, c + 1)
        plt.semilogy()
        plt.plot(lth, ps_theory[spec])
        for run in runs:
            plt.plot(lb, Db_cross_dict[scan, run, spec], "o", label = r" %s %s" % (scan, run))
    plt.legend()
    plt.savefig("%s/signal_spec_%s.png" % (plot_dir, scan), bbox_inches="tight")
    plt.clf()
    plt.close()

