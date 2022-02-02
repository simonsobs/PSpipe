"""
This script compute all power spectra and write them to disk.
It uses the window function provided in the dictionnary file.
Optionally, it applies a calibration to the maps, a kspace filter and deconvolve the pixel window function.
The spectra are then combined in mean auto, cross and noise power spectrum and written to disk.
If write_all_spectra=True, each individual spectrum is also written to disk.
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_map_preprocessing
from pixell import enmap
import numpy as np
import healpy as hp
import sys
import data_analysis_utils
import time


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
write_all_spectra = d["write_splits_spectra"]
deconvolve_pixwin = d["deconvolve_pixwin"]

mcm_dir = "mcms"
spec_dir = "spectra"
alms_dir = "alms"

pspy_utils.create_directory(spec_dir)

master_alms = {}
nsplit = {}

# read the alms

for sv in surveys:
    for ar in d["arrays_%s" % sv]:
        maps = d["maps_%s_%s" % (sv, ar)]
        nsplit[sv] = len(maps)
        print("%s split of survey: %s, array %s"%(nsplit[sv], sv, ar))
        for k, map in enumerate(maps):
            master_alms[sv, ar, k] = np.load("%s/alms_%s_%s_%d.npy" % (alms_dir, sv, ar, k))
                
# compute the transfer functions
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
tf_array = {}

for sv in surveys:

    tf_survey = np.ones(len(lb))
    ks_f = d["k_filter_%s" % sv]
    template = so_map.read_map(d["window_T_%s_%s" % (sv, d["arrays_%s" % sv][0])])

    if ks_f["apply"]:
        if ks_f["tf"] == "analytic":
            print("compute analytic kspace tf %s" % sv)
            shape, wcs = template.data.shape, template.data.wcs
            if ks_f["type"] == "binary_cross":
                filter = so_map_preprocessing.build_std_filter(shape, wcs, vk_mask=ks_f["vk_mask"], hk_mask=ks_f["hk_mask"], dtype=np.float32)
            elif ks_f["type"] == "gauss":
                filter = so_map_preprocessing.build_sigurd_filter(shape, wcs, ks_f["lbounds"], dtype=np.float32)
            else:
                print("you need to specify a valid filter type")
                sys.exit()

            _, kf_tf = so_map_preprocessing.analytical_tf(template, filter, binning_file, lmax)
        else:
            print("use kspace tf from file %s" % sv)
            _, _, kf_tf, _ = np.loadtxt(ks_f["tf"], unpack=True)
            
        tf_survey *= np.sqrt(np.abs(kf_tf[:len(lb)]))

    if deconvolve_pixwin:
        # extra pixel window function deconvolution for healpix and planck projected on CAR
        pixwin_l = np.ones(2 * lmax)
        if sv == "Planck":
            print("Deconvolve Planck pixel window function")
            pixwin_l = hp.pixwin(2048)
        if template.pixel == "HEALPIX":
            pixwin_l = hp.pixwin(template.nside)

        # this should be checked with simulations since maybe this should be done at the mcm level
        _, pw = pspy_utils.naive_binning(np.arange(len(pixwin_l)),  pixwin_l, binning_file, lmax)
        tf_survey *= pw

    for id_ar, ar in enumerate(d["arrays_%s" % sv]):
        tf_array[sv, ar] = tf_survey.copy()
        
        if d["deconvolve_map_maker_tf_%s" % sv]:
            print("deconvolve map maker tf %s %s" % (sv, ar))
            _, mm_tf = np.loadtxt("mm_tf_%s_%s.dat" % (sv, ar), unpack=True)
            tf_array[sv, ar] *= mm_tf[:len(lb)]
            
        np.savetxt(spec_dir + "/tf_%s_%s.dat" % (sv, ar),
                   np.transpose([lb, tf_array[sv, ar]]))

# compute the power spectra

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

ps_dict = {}

for id_sv1, sv1 in enumerate(surveys):
    
    for id_ar1, ar1 in enumerate(d["arrays_%s" % sv1]):
    
        for id_sv2, sv2 in enumerate(surveys):
                    
            for id_ar2, ar2 in enumerate(d["arrays_%s" % sv2]):
            
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
            
                for spec in spectra:
                    ps_dict[spec, "auto"] = []
                    ps_dict[spec, "cross"] = []
                    
                nsplits_1 = nsplit[sv1]
                nsplits_2 = nsplit[sv2]
                
                for s1 in range(nsplits_1):
                    for s2 in range(nsplits_2):
                        if (sv1 == sv2) & (ar1 == ar2) & (s1>s2) : continue
                    
                        spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
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
                                                        
                        data_analysis_utils.deconvolve_tf(lb, ps, tf_array[sv1, ar1], tf_array[sv2, ar2], 3, lmax)

                        if write_all_spectra:
                            so_spectra.write_ps(spec_dir + "/%s.dat" % spec_name, lb, ps, type, spectra=spectra)

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
                    spec_name_cross = "%s_%s_%sx%s_%s_cross" % (type, sv1, ar1, sv2, ar2)
                    
                    if ar1 == ar2 and sv1 == sv2:
                        # Average TE / ET so that for same array same season TE = ET
                        ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec, "cross"], axis=0) + np.mean(ps_dict[spec[::-1], "cross"], axis=0)) / 2.

                    if sv1 == sv2:
                        ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
                        spec_name_auto = "%s_%s_%sx%s_%s_auto" % (type, sv1, ar1, sv2, ar2)
                        ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / nsplit[sv1]
                        spec_name_noise = "%s_%s_%sx%s_%s_noise" % (type, sv1, ar1, sv2, ar2)

                so_spectra.write_ps(spec_dir + "/%s.dat" % spec_name_cross, lb, ps_dict_cross_mean, type, spectra=spectra)
                if sv1 == sv2:
                    so_spectra.write_ps(spec_dir + "/%s.dat" % spec_name_auto, lb, ps_dict_auto_mean, type, spectra=spectra)
                    so_spectra.write_ps(spec_dir + "/%s.dat" % spec_name_noise, lb, ps_dict_noise_mean, type, spectra=spectra)


