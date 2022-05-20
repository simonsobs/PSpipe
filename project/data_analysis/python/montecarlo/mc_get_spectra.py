"""
This script generate simplistic simulations of the actpol data
it generates gaussian simulations of cmb, fg and noise
the fg is based on fgspectra, and the noise is based on the 1d noise power spectra measured on the data
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi, so_map_preprocessing
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
write_all_spectra = d["write_splits_spectra"]
sim_alm_dtype = d["sim_alm_dtype"]

if sim_alm_dtype == "complex64":
    sim_alm_dtype = np.complex64
elif sim_alm_dtype == "complex128":
    sim_alm_dtype = np.complex128

window_dir = "windows"
mcm_dir = "mcms"
spec_dir = "sim_spectra"
bestfit_dir = "best_fits"
ps_model_dir = "noise_model"

pspy_utils.create_directory(spec_dir)

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

# prepare the filter and the transfer function
template = {}
filter = {}
tf_survey = {}
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

for sv in surveys:

    template_name = d["maps_%s_%s" % (sv, arrays[0])][0]
    template[sv] = so_map.read_map(template_name)
    ks_f = d["k_filter_%s" % sv]
    
    tf_survey[sv] = np.ones(len(lb))

    if template[sv].pixel == "CAR":
    
        if ks_f["apply"]:
            shape, wcs = template[sv].data.shape, template[sv].data.wcs
            if ks_f["type"] == "binary_cross":
                filter[sv] = so_map_preprocessing.build_std_filter(shape, wcs, vk_mask=ks_f["vk_mask"], hk_mask=ks_f["hk_mask"], dtype=np.float64)
            elif ks_f["type"] == "gauss":
                filter[sv] = so_map_preprocessing.build_sigurd_filter(shape, wcs, ks_f["lbounds"], dtype=np.float64)
            else:
                print("you need to specify a valid filter type")

            if ks_f["tf"] == "analytic":
                print("compute analytic kspace tf %s" % sv)
                _, kf_tf = so_map_preprocessing.analytical_tf(template[sv], filter[sv], binning_file, lmax)
            else:
                print("use kspace tf from file %s" % sv)
                _, _, kf_tf, _ = np.loadtxt(ks_f["tf"], unpack=True)
            
            tf_survey[sv] *= np.sqrt(np.abs(kf_tf[:len(lb)]))



# we will use mpi over the number of simulations
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    t0 = time.time()
 
    # generate cmb alms and foreground alms
    # cmb alms will be of shape (3, lm) 3 standing for T,E,B
    # fglms will be of shape (nfreq, lm) and is T only
    
    alms = curvedsky.rand_alm(ps_cmb, lmax=lmax, seed=(iii, 101), dtype=sim_alm_dtype)
    fglms = curvedsky.rand_alm(ps_fg, lmax=lmax, seed=(iii, 102), dtype=sim_alm_dtype)
    
    master_alms = {}
    nsplits = {}
    
    for sv in surveys:
    
        ks_f = d["k_filter_%s" % sv]
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
                                                       nl_array_pol=nl_array_pol,
                                                       dtype=sim_alm_dtype)
        print(nl_array_t.shape)
        del nl_array_t, nl_array_pol

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

            print("%s split of survey: %s, array %s" % (nsplits[sv], sv, ar))

            t1 = time.time()

            for k in range(nsplits[sv]):
            
                # finally we add the noise alms for each split
                noisy_alms = alms_beamed.copy()
                noisy_alms[0] +=  nlms["T", k][ar_id]
                noisy_alms[1] +=  nlms["E", k][ar_id]
                noisy_alms[2] +=  nlms["B", k][ar_id]

                split = sph_tools.alm2map(noisy_alms, template[sv])
                
                # from now on the simulation pipeline is done
                # and we are back to the get_spectra algorithm
                                
                if window_tuple[0].pixel == "CAR":
                    if ks_f["apply"]:
                        binary = so_map.read_map("%s/binary_%s_%s.fits" % (window_dir, sv, ar))
                        split = data_analysis_utils.get_filtered_map(split, binary, filter[sv], weighted_filter=ks_f["weighted"])

                        del binary
                
                if d["remove_mean"] == True:
                    split = data_analysis_utils.remove_mean(split, window_tuple, ncomp)
                
                master_alms[sv, ar, k] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)
                print("m_alms_dtype", master_alms[sv, ar, k].dtype)

            print(time.time()-t1)

    ps_dict = {}
    _, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

    for id_sv1, sv1 in enumerate(surveys):
        arrays_1 = d["arrays_%s" % sv1]
        nsplits_1 = nsplits[sv1]
    
        for id_ar1, ar1 in enumerate(arrays_1):
    
            for id_sv2, sv2 in enumerate(surveys):
                arrays_2 = d["arrays_%s" % sv2]
                nsplits_2 = nsplits[sv2]
            
                for id_ar2, ar2 in enumerate(arrays_2):

                    if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                    if  (id_sv1 > id_sv2) : continue

                    for spec in spectra:
                        ps_dict[spec, "auto"] = []
                        ps_dict[spec, "cross"] = []
                
                    for s1 in range(nsplits_1):
                        for s2 in range(nsplits_2):
                            if (sv1 == sv2) & (ar1 == ar2) & (s1>s2) : continue
                    
                            mbb_inv, Bbl = so_mcm.read_coupling(prefix="%s/%s_%sx%s_%s" % (mcm_dir, sv1, ar1, sv2, ar2),
                                                                spin_pairs=spin_pairs)

                            l, ps_master = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, s1],
                                                                         master_alms[sv2, ar2, s2],
                                                                         spectra=spectra)
                                                              
                            spec_name="%s_%s_%sx%s_%s_%d%d" % (type, sv1, ar1, sv2, ar2, s1, s2)
                        
                            lb, ps = so_spectra.bin_spectra(l,
                                                            ps_master,
                                                            binning_file,
                                                            lmax,
                                                            type=type,
                                                            mbb_inv=mbb_inv,
                                                            spectra=spectra)
                                                        
                            data_analysis_utils.deconvolve_tf(lb, ps, tf_survey[sv1], tf_survey[sv2], ncomp, lmax)

                            if write_all_spectra:
                                so_spectra.write_ps(spec_dir + "/%s_%05d.dat" % (spec_name,iii), lb, ps, type, spectra=spectra)

                            for count, spec in enumerate(spectra):
                                if (s1 == s2) & (sv1 == sv2):
                                    if count == 0:
                                        print("auto %s_%s X %s_%s %d%d" % (sv1, ar1, sv2, ar2, s1, s2))
                                    ps_dict[spec, "auto"] += [ps[spec]]
                                else:
                                    if count == 0:
                                        print("cross %s_%s X %s_%s %d%d" % (sv1, ar1, sv2, ar2, s1, s2))
                                    ps_dict[spec, "cross"] += [ps[spec]]

                    ps_dict_auto_mean = {}
                    ps_dict_cross_mean = {}
                    ps_dict_noise_mean = {}

                    for spec in spectra:
                        ps_dict_cross_mean[spec] = np.mean(ps_dict[spec, "cross"], axis=0)
                        spec_name_cross = "%s_%s_%sx%s_%s_cross_%05d" % (type, sv1, ar1, sv2, ar2, iii)
                    
                        if ar1 == ar2 and sv1 == sv2:
                            # Average TE / ET so that for same array same season TE = ET
                            ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec, "cross"], axis=0) + np.mean(ps_dict[spec[::-1], "cross"], axis=0)) / 2.

                        if sv1 == sv2:
                            ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
                            spec_name_auto = "%s_%s_%sx%s_%s_auto_%05d" % (type, sv1, ar1, sv2, ar2, iii)
                            ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / nsplits[sv1]
                            spec_name_noise = "%s_%s_%sx%s_%s_noise_%05d" % (type, sv1, ar1, sv2, ar2, iii)

                    so_spectra.write_ps(spec_dir + "/%s.dat" % spec_name_cross, lb, ps_dict_cross_mean, type, spectra=spectra)
                
                    if sv1 == sv2:
                        so_spectra.write_ps(spec_dir+"/%s.dat" % spec_name_auto, lb, ps_dict_auto_mean, type, spectra=spectra)
                        so_spectra.write_ps(spec_dir+"/%s.dat" % spec_name_noise, lb, ps_dict_noise_mean, type, spectra=spectra)

    print("sim number %05d done in %.02f s" % (iii, time.time()-t0))
