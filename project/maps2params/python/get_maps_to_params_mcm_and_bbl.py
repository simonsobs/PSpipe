"""
This script computes the window functions, the mode coupling matrices and the binning matrices Bbl
for  different frequency channels of different CMB experiments and write them to disk.
"""
from pspy import so_map, so_window, so_mcm, pspy_utils, so_dict
import healpy as hp
import numpy as np
import pylab as plt
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

window_dir = "windows"
mcm_dir = "mcms"
plot_dir = "plots/windows/"

pspy_utils.create_directory(window_dir)
pspy_utils.create_directory(mcm_dir)
pspy_utils.create_directory(plot_dir)

experiments = d["experiments"]
lmax = d["lmax"]
lmax_mcm = d["lmax_mcm"]

# The first step of this code is to generate the window functions for the different frequency channels
# of the different experiments.

print("Geneating window functions")

for exp in experiments:
    freqs = d["freqs_%s" % exp]

    if d["pixel_%s" % exp] == "CAR":
        binary = so_map.car_template(ncomp=1,
                                     ra0=d["ra0_%s" % exp],
                                     ra1=d["ra1_%s" % exp],
                                     dec0=d["dec0_%s" % exp],
                                     dec1=d["dec1_%s" % exp],
                                     res=d["res_%s" % exp])
                                     
        binary.data[:] = 1
        if d["binary_is_survey_mask"] == True:
            binary.data[:] = 0
            binary.data[1:-1, 1:-1] = 1
    
    elif d["pixel_%s" % exp] == "HEALPIX":
        binary=so_map.healpix_template(ncomp=1, nside=d["nside_%s" % exp])
        binary.data[:] = 1

    for freq in freqs:
        window = binary.copy()
        
        if d["galactic_mask_%s" % exp] == True:
            gal_mask = so_map.read_map(d["galactic_mask_%s_file_%s" % (exp, freq)])
            gal_mask.plot(file_name="%s/gal_mask_%s_%s" % (plot_dir, exp, freq))
            window.data[:] *= gal_mask.data[:]
        
        if d["survey_mask_%s" % exp] == True:
            survey_mask = so_map.read_map(d["survey_mask_%s_file_%s" % (exp, freq)])
            survey_mask.plot(file_name="%s/survey_mask_mask_%s_%s" % (plot_dir, exp, freq))
            window.data[:] *= survey_mask.data[:]

        apo_radius_degree = (d["apo_radius_survey_%s" % exp])
        window = so_window.create_apodization(window, apo_type=d["apo_type_survey_%s" % exp], apo_radius_degree=apo_radius_degree)

        if d["pts_source_mask_%s" % exp] == True:
            hole_radius_arcmin=(d["source_mask_radius_%s" % exp])
            mask = so_map.simulate_source_mask(binary, n_holes=d["source_mask_nholes_%s" % exp], hole_radius_arcmin=hole_radius_arcmin)
            mask = so_window.create_apodization(mask, apo_type=d["apo_type_mask_%s" % exp], apo_radius_degree=d["apo_radius_mask_%s" % exp])
            window.data[:] *= mask.data[:]

        window.write_map("%s/window_%s_%s.fits" % (window_dir, exp, freq))
        window.plot(file_name="%s/window_%s_%s" % (plot_dir, exp, freq))

# We then compute the mode coupling matrices and binning matrices BBl for the different cross spectra
# that we will form, the code print the cross spectra to be considered in this computation.

print("Computing mode coupling matrices for the cross spectra:")

for id_exp1, exp1 in enumerate(experiments):
    freqs1 = d["freqs_%s" % exp1]
    
    for id_f1, freq1 in enumerate(freqs1):
        l, bl1 = np.loadtxt("sim_data/beams/beam_%s_%s.dat" % (exp1,freq1), unpack=True)
        window1 = so_map.read_map("%s/window_%s_%s.fits" % (window_dir, exp1, freq1))
        
        for id_exp2, exp2 in enumerate(experiments):
            freqs2 = d["freqs_%s" % exp2]
            
            for id_f2, freq2 in enumerate(freqs2):
                # This ensures that we do not repeat equivalent computation
                if  (id_exp1 == id_exp2) & (id_f1 > id_f2) : continue
                if  (id_exp1 > id_exp2) : continue
                
                print("%s_%s x %s_%s" % (exp1, freq1, exp2, freq2))

                l, bl2 = np.loadtxt("sim_data/beams/beam_%s_%s.dat" % (exp2, freq2), unpack=True)
                window2 = so_map.read_map("%s/window_%s_%s.fits" % (window_dir,exp2,freq2))
                
                mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(win1=(window1, window1),
                                                            win2=(window2, window2),
                                                            bl1=(bl1, bl1),
                                                            bl2=(bl2, bl2),
                                                            binning_file=d["binning_file"],
                                                            niter=d["niter"],
                                                            lmax=d["lmax"],
                                                            type=d["type"],
                                                            save_file="%s/%s_%sx%s_%s"%(mcm_dir, exp1, freq1, exp2, freq2))



