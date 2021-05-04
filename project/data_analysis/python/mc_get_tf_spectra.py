"""
This script generate simplistic signa-only simulations of the actpol data that will be used to measure the transfer function
This is essentially a much simpler version of mc_get_spectra.py, since it doesn't include noise on the simulation and thus does not require
different splits of the data
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi
import numpy as np
import sys
import data_analysis_utils
import time
from pixell import curvedsky, powspec


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
type = d["type"]
binning_file = d["binning_file"]
sim_alm_dtype = d["sim_alm_dtype"]

if sim_alm_dtype == "complex64":
    sim_alm_dtype = np.complex64
elif sim_alm_dtype == "complex128":
    sim_alm_dtype = np.complex128

window_dir = "windows"
mcm_dir = "mcms"
tf_dir = "spectra_for_tf"
bestfit_dir = "best_fits"
ps_model_dir = "noise_model"

pspy_utils.create_directory(tf_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

# let's list the different frequencies used in the code
freq_list = []
for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for ar in arrays:
        freq_list += [d["nu_eff_%s_%s" % (sv, ar)]]
freq_list = list(dict.fromkeys(freq_list)) # this bit removes doublons

id_freq = {}
# create a list assigning an integer index to each freq (used later in the code to generate fg simulations)
for count, freq in enumerate(freq_list):
    id_freq[freq] = count
    
# we read cmb and fg best fit power spectrum
# we put the best fit power spectrum in a matrix [nfreqs, nfreqs, lmax]
# taking into account the correlation of the fg between different frequencies

ncomp = 3
ps_cmb = powspec.read_spectrum("%s/lcdm.dat" % bestfit_dir)[:ncomp, :ncomp]
l, ps_fg = data_analysis_utils.get_foreground_matrix(bestfit_dir, freq_list, lmax)

# the template for the simulations
template = d["maps_%s_%s" % (surveys[0], arrays[0])][0]
template = so_map.read_map(template)

# we will use mpi over the number of simulations
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    t0 = time.time()
 
    # generate cmb alms and foreground alms
    # cmb alms will be of shape (3, lm) 3 standing for T,E,B
    # fglms will be of shape (nfreq, lm) and is T only
    
    alms = curvedsky.rand_alm(ps_cmb, lmax=lmax, dtype=sim_alm_dtype)
    fglms = curvedsky.rand_alm(ps_fg, lmax=lmax, dtype=sim_alm_dtype)
    
    master_alms = {}

    for sv in surveys:
    
        arrays = d["arrays_%s" % sv]
        
        for ar_id, ar in enumerate(arrays):
        
            win_T = so_map.read_map(d["window_T_%s_%s" % (sv, ar)])
            win_pol = so_map.read_map(d["window_pol_%s_%s" % (sv, ar)])
    
            window_tuple = (win_T, win_pol)
            
            del win_T, win_pol
        
            # we add fg alms to cmb alms in temperature
            alms_beamed = alms.copy()
            alms_beamed[0] += fglms[id_freq[d["nu_eff_%s_%s" % (sv, ar)]]]
            
            # we convolve signal + foreground with the beam of the array
            l, bl = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv, ar)])
            alms_beamed = curvedsky.almxfl(alms_beamed, bl)

            # generate our signal only sim
            split = sph_tools.alm2map(alms_beamed, template)
            
            # compute the alms of the sim
                
            master_alms[sv, ar, "nofilter"] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)
            
            # apply the k-space filter

            binary = so_map.read_map("%s/binary_%s_%s.fits" % (window_dir, sv, ar))
            split = data_analysis_utils.get_filtered_map(split,
                                                         binary,
                                                         vk_mask=d["vk_mask"],
                                                         hk_mask=d["hk_mask"],
                                                         normalize=False)
                                                         
            # compute the alms of the filtered sim

            master_alms[sv, ar, "filter"] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)
            master_alms[sv, ar, "filter"] /= (split.data.shape[1]*split.data.shape[2])


    ps_dict = {}
    _, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

    for id_sv1, sv1 in enumerate(surveys):
        arrays_1 = d["arrays_%s" % sv1]
        for id_ar1, ar1 in enumerate(arrays_1):
            for id_sv2, sv2 in enumerate(surveys):
                arrays_2 = d["arrays_%s" % sv2]
                for id_ar2, ar2 in enumerate(arrays_2):

                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue

                    spec_name="%s_%s_%sx%s_%s" % (type, sv1, ar1, sv2, ar2)

                    
                    mbb_inv, Bbl = so_mcm.read_coupling(prefix="%s/%s_%sx%s_%s" % (mcm_dir, sv1, ar1, sv2, ar2),
                                                        spin_pairs=spin_pairs)
                                                        
                    # we  compute the power spectra of the sim (with and without the k-space filter applied)
                    
                    for filt in ["filter", "nofilter"]:
                    
                        l, ps_master = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, filt],
                                                                     master_alms[sv2, ar2, filt],
                                                                     spectra=spectra)
                                                                                      
                        lb, ps = so_spectra.bin_spectra(l,
                                                        ps_master,
                                                        binning_file,
                                                        lmax,
                                                        type=type,
                                                        mbb_inv=mbb_inv,
                                                        spectra=spectra)
                                        

                        so_spectra.write_ps(tf_dir + "/%s_%s_%05d.dat" % (spec_name, filt, iii), lb, ps, type, spectra=spectra)

              

