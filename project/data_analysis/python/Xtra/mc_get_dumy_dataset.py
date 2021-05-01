"""
This script generate a very simplistic dataset (to test the pipeline in a quick way)
The data set will have the same complexity in term of combinatoric, but the maps themselves will be much smaller
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi
import numpy as np
import sys
import data_analysis_utils
import time
from pixell import curvedsky, powspec


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


bestfit_dir = "best_fits"
ps_model_dir = "noise_model"
dummy_data_dir = "dummy_data"

lmax = 2000
ra0, ra1, dec0, dec1 = -10, 10, -10, 10
res = 3

pspy_utils.create_directory(dummy_data_dir)

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
template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)

for iii in subtasks:
    t0 = time.time()
 
    # generate cmb alms and foreground alms
    # cmb alms will be of shape (3, lm) 3 standing for T,E,B
    # fglms will be of shape (nfreq, lm) and is T only
    
    alms = curvedsky.rand_alm(ps_cmb, lmax=lmax)
    fglms = curvedsky.rand_alm(ps_fg, lmax=lmax)
    
    master_alms = {}
    nsplits = {}
    
    for sv in surveys:
    
        arrays = d["arrays_%s" % sv]
        nsplits[sv] = len(d["maps_%s_%s" % (sv, arrays[0])])
        
        # for each survey, we read the mesasured noise power spectrum from the data
        # since we want to allow for array x array noise correlation this is an
        # (narrays, narrays, lmax) matrix
        # the pol noise is taken as the arithmetic mean of E and B noise

        l, nl_array_t, nl_array_pol = data_analysis_utils.get_noise_matrix_spin0and2(ps_model_dir,
                                                                                     sv,
                                                                                     arrays,
                                                                                     lmax,
                                                                                     nsplits[sv])
                                                                                     
        # we generate noise alms from the matrix, resulting in a dict with entry ["T,E,B", "0,...nsplit-1"]
        # each element of the dict is a [narrays,lm] array
        
        nlms = data_analysis_utils.generate_noise_alms(nl_array_t,
                                                       lmax,
                                                       nsplits[sv],
                                                       ncomp,
                                                       nl_array_pol=nl_array_pol)

        for ar_id, ar in enumerate(arrays):
                
            # we add fg alms to cmb alms in temperature
            alms_beamed = alms.copy()
            alms_beamed[0] += fglms[id_freq[d["nu_eff_%s_%s" % (sv, ar)]]]
            
            # we convolve signal + foreground with the beam of the array
            l, bl = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv, ar)])
            alms_beamed = data_analysis_utils.multiply_alms(alms_beamed, bl, ncomp)

            print("%s split of survey: %s, array %s" % (nsplits[sv], sv, ar))
            maps = d["maps_%s_%s" % (sv, ar)]

            for k, map in enumerate(maps):

                # finally we add the noise alms for each split
                noisy_alms = alms_beamed.copy()
                noisy_alms[0] +=  nlms["T", k][ar_id]
                noisy_alms[1] +=  nlms["E", k][ar_id]
                noisy_alms[2] +=  nlms["B", k][ar_id]

                split = sph_tools.alm2map(noisy_alms, template)
                print(map)
