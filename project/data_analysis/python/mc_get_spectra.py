"""
This script compute all power spectra and write them to disk.
"""
from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi
import numpy as np
import sys
import data_analysis_utils
from pixell import powspec, curvedsky
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
type = d["type"]
binning_file = d["binning_file"]
write_all_spectra = d["write_splits_spectra"]

window_dir = "windows"
mcm_dir = "mcms"
spec_dir = "sim_spectra"
bestfit_dir = "best_fits"
ps_model_dir = "noise_model"

pspy_utils.create_directory(spec_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
ncomp = 3

ps_cmb = powspec.read_spectrum(d["theoryfile"])[:ncomp, :ncomp]
all_freqs = list(dict.fromkeys([d["nu_eff_%s_%s" % (sv,ar)] for sv in surveys for ar in d["arrays_%s" % sv]]))
l, ps_fg = data_analysis_utils.get_foreground_matrix(bestfit_dir, all_freqs, lmax + 1)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    t0 = time.time()
    
    master_alms = {}
    nsplits = {}

    alms = curvedsky.rand_alm(ps_cmb, lmax=lmax)
    fglms = curvedsky.rand_alm(ps_fg, lmax=lmax)
    fglms_dict = {freq: fglms[i] for i, freq in enumerate(all_freqs)}

    for sv in surveys:
        arrays = d["arrays_%s" % sv]
        nsplits[sv] = len( d["maps_%s_%s" % (sv, arrays[0])])

        l, nl_t, nl_pol = data_analysis_utils.get_noise_matrix_spin0and2(ps_model_dir,
                                                                         sv,
                                                                         arrays,
                                                                         lmax,
                                                                         nsplits[sv])

        nlms = data_analysis_utils.generate_noise_alms(nl_t,
                                                       lmax,
                                                       nsplits[sv],
                                                       ncomp,
                                                       nl_array_pol=nl_pol)

        for ar_id, ar in enumerate(arrays):
        
            alms_ar = alms.copy()
            # Add fg (only in T)
            alms_ar[0] += fglms_dict[d["nu_eff_%s_%s" % (sv, ar)]]
            
            l, bl = pspy_utils.read_beam_file(d["beam_%s_%s" % (sv, ar)], lmax=lmax)

            alms_ar = data_analysis_utils.multiply_alms(alms_ar, bl, ncomp)
        
            win_T = so_map.read_map("%s/window_T_%s_%s.fits" % (window_dir, sv, ar))
            win_pol = so_map.read_map("%s/window_pol_%s_%s.fits" % (window_dir, sv, ar))
            window_tuple = (win_T, win_pol)
            box = so_map.bounding_box_from_map(win_T)
            dec0, ra0, dec1, ra1 = box.flatten()*180/np.pi
        
            for k in range(nsplits[sv]):
            
                noisy_alms = alms_ar.copy()
                noisy_alms[0] +=  nlms["T",k][ar_id]
                noisy_alms[1] +=  nlms["E",k][ar_id]
                noisy_alms[2] +=  nlms["B",k][ar_id]
                
                template = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, 0.5)

                split = sph_tools.alm2map(noisy_alms, template)

                split = data_analysis_utils.remove_mean(split, window_tuple, ncomp)
                master_alms[sv, ar, k] = sph_tools.get_alms(split, window_tuple, niter, lmax)


    ps_dict = {}

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

                            l, ps_master = so_spectra.get_spectra(master_alms[sv1, ar1, s1],
                                                                  master_alms[sv2, ar2, s2],
                                                                  spectra=spectra)
                                                              
                        
                            lb, ps = so_spectra.bin_spectra(l,
                                                            ps_master,
                                                            binning_file,
                                                            lmax,
                                                            type=type,
                                                            mbb_inv=mbb_inv,
                                                            spectra=spectra)
                                                        

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
                    
                        if sv1 == sv2:
                            ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
                            spec_name_auto = "%s_%s_%sx%s_%s_auto_%05d" % (type, sv1, ar1, sv2, ar2, iii)
                            ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / nsplits[sv1]
                            spec_name_noise = "%s_%s_%sx%s_%s_noise_%05d" % (type, sv1, ar1, sv2, ar2, iii)

                    so_spectra.write_ps(spec_dir + "/%s.dat" % spec_name_cross, lb, ps_dict_cross_mean, type, spectra=spectra)
                
                    if sv1 == sv2:
                        so_spectra.write_ps(spec_dir + "/%s.dat" % spec_name_auto, lb, ps_dict_auto_mean, type, spectra=spectra)
                        so_spectra.write_ps(spec_dir + "/%s.dat" % spec_name_noise, lb, ps_dict_noise_mean, type, spectra=spectra)


