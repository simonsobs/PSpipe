"""
This script compute all power spectra and write them to disk.
It uses the window function provided in the dictionnary file.
Optionally, it applies a calibration to the maps, a kspace filter and deconvolve the pixel window function.
The spectra are then combined in mean auto, cross and noise power spectrum and written to disk.
If write_all_spectra=True, each individual spectrum is also written to disk.
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra
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
niter = d["niter"]
type = d["type"]
binning_file = d["binning_file"]
write_all_spectra = d["write_splits_spectra"]
deconvolve_pixwin = d["deconvolve_pixwin"]

window_dir = "windows"
mcm_dir = "mcms"
specDir = "spectra"
plot_dir = "plots/maps/"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(specDir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

ncomp = 3

master_alms = {}
nsplit = {}
pixwin_l = {}

for sv in surveys:
    arrays = d["arrays_%s" % sv]

    for ar in arrays:
        win_T = so_map.read_map(d["window_T_%s_%s" % (sv, ar)])
        win_pol = so_map.read_map(d["window_pol_%s_%s" % (sv, ar)])

        window_tuple = (win_T, win_pol)
        
        maps = d["maps_%s_%s" % (sv, ar)]
        nsplit[sv, ar] = len(maps)

        cal = d["cal_%s_%s" % (sv, ar)]
        print("%s split of survey: %s, array %s"%(nsplit[sv, ar], sv, ar))
        
        if deconvolve_pixwin:
            # ok so this is a bit overcomplicated because we need to take into account CAR and HEALPIX
            # for CAR the pixel window function deconvolution is done in Fourier space and take into account
            # the anisotropy if the pixwin
            # In HEALPIX it's a simple 1d function in multipole space
            # we also need to take account the case where we have projected Planck into a CAR pixellisation since
            # the native pixel window function of Planck need to be deconvolved
            if win_T.pixel == "CAR":
                wy, wx = enmap.calc_window(win_T.data.shape)
                inv_pixwin_lxly = (wy[:,None] * wx[None,:]) ** (-1)
                pixwin_l[sv] = np.ones(2 * lmax)
                if sv == "Planck":
                    print("Deconvolve Planck pixel window function")
                    # we include this special case for Planck projected in CAR taking into account the Planck native pixellisation
                    # we should check if the projection doesn't include an extra pixel window
                    inv_pixwin_lxly = None
                    pixwin_l[sv] = hp.pixwin(2048)

            elif win_T.pixel == "HEALPIX":
                pixwin_l[sv] = hp.pixwin(win_T.nside)
                
        else:
            inv_pixwin_lxly = None


        t = time.time()
        for k, map in enumerate(maps):
        
            if win_T.pixel == "CAR":
                split = so_map.read_map(map, geometry=win_T.data.geometry)
                
                if d["src_free_maps_%s" % sv] == True:
                    point_source_map_name = map.replace("srcfree.fits", "model.fits")
                    if point_source_map_name == map:
                        raise ValueError("No model map is provided! Check map names!")
                    point_source_map = so_map.read_map(point_source_map_name)
                    point_source_mask = so_map.read_map(d["ps_mask_%s_%s" % (sv, ar)])
                    split = data_analysis_utils.get_coadded_map(split, point_source_map, point_source_mask)

                if d["use_kspace_filter"]:
                    print("apply kspace filter on %s" %map)
                    binary = so_map.read_map("%s/binary_%s_%s.fits" % (window_dir, sv, ar))
                    split = data_analysis_utils.get_filtered_map(
                        split, binary, vk_mask=d["vk_mask"], hk_mask=d["hk_mask"], normalize=False, inv_pixwin_lxly=inv_pixwin_lxly)
                else:
                    print("WARNING: no kspace filter is applied")
                    if deconvolve_pixwin:
                        print("WARNING: pixwin deconvolution in CAR WITHOUT kspace filter is not implemented")
                        sys.exit()
                        
            elif win_T.pixel == "HEALPIX":
                split = so_map.read_map(map)
                
            split.data *= cal
            if d["remove_mean"] == True:
                split = data_analysis_utils.remove_mean(split, window_tuple, ncomp)
                

            master_alms[sv, ar, k] = sph_tools.get_alms(split, window_tuple, niter, lmax)
            
            if d["use_kspace_filter"]:
                # there is an extra normalisation for the FFT/IFFT bit
                # note that we apply it here rather than at the FFT level because correcting the alm is faster than correcting the maps
                master_alms[sv, ar, k] /= (split.data.shape[1]*split.data.shape[2])
                
        print(time.time()- t)

ps_dict = {}
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    
    if d["tf_%s" % sv1] is not None:
        print("will deconvolve tf of %s" %sv1)
        _, _, tf1, _ = np.loadtxt(d["tf_%s" % sv1], unpack=True)
        tf1 = tf1[:len(lb)]
    else:
        tf1 = np.ones(len(lb))
    
    if deconvolve_pixwin:
        # we have an extra correction for the 1d healpix pixel window function
        # this should be checked with simulations since maybe this
        # step should be done at the mcm level
        l_pw = np.arange(len(pixwin_l[sv1]))
        _, pw1 = pspy_utils.naive_binning(l_pw,  pixwin_l[sv1], binning_file, lmax)
        tf1 *= pw1
        
    for id_ar1, ar1 in enumerate(arrays_1):
    
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            
            if d["tf_%s" % sv2] is not None:
                print("will deconvolve tf of %s" %sv2)
                _, _, tf2, _ = np.loadtxt(d["tf_%s" % sv2], unpack=True)
                tf2 = tf2[:len(lb)]
            else:
                tf2 = np.ones(len(lb))
                
            if deconvolve_pixwin:
                l_pw = np.arange(len(pixwin_l[sv2]))
                _, pw2 = pspy_utils.naive_binning(l_pw,  pixwin_l[sv2], binning_file, lmax)
                tf2 *= pw2

            for id_ar2, ar2 in enumerate(arrays_2):
            
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
            
                for spec in spectra:
                    ps_dict[spec, "auto"] = []
                    ps_dict[spec, "cross"] = []
                    
                    
                nsplits_1 = nsplit[sv1, ar1]
                nsplits_2 = nsplit[sv2, ar2]
                
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
                                                        
                        data_analysis_utils.deconvolve_tf(lb, ps, tf1, tf2, ncomp, lmax)

                        if write_all_spectra:
                            so_spectra.write_ps(specDir + "/%s.dat" % spec_name, lb, ps, type, spectra=spectra)

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

                so_spectra.write_ps(specDir + "/%s.dat" % spec_name_cross, lb, ps_dict_cross_mean, type, spectra=spectra)
                
                if sv1 == sv2:
                    so_spectra.write_ps(specDir+"/%s.dat" % spec_name_auto, lb, ps_dict_auto_mean, type, spectra=spectra)
                    so_spectra.write_ps(specDir+"/%s.dat" % spec_name_noise, lb, ps_dict_noise_mean, type, spectra=spectra)


