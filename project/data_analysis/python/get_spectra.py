"""
This script compute all power spectra and write them to disk.
"""
from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra
import numpy as np
import sys
import data_analysis_utils

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
specDir = "spectra"
plot_dir = "plots/maps/"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(specDir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

ncomp = 3

master_alms = {}
nsplit = {}
for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for ar in arrays:
        win_T = so_map.read_map("%s/window_T_%s_%s.fits" % (window_dir, sv, ar))
        win_pol = so_map.read_map("%s/window_pol_%s_%s.fits" % (window_dir, sv, ar))

        window_tuple = (win_T, win_pol)
        
        maps = d["maps_%s_%s" % (sv, ar)]
        nsplit[sv] = len(maps)

        cal = d["cal_%s_%s" % (sv, ar)]
        print("%s split of survey: %s, array %s"%(nsplit[sv], sv, ar))
        
        for k, map in enumerate(maps):
            if win_T.pixel == "CAR":
                split = so_map.read_map(map, geometry=win_T.data.geometry)

            elif win_T.pixel == "HEALPIX":
                split = so_map.read_map(map)

            split.data *= cal
            if d["remove_mean"] == True:
                split = data_analysis_utils.remove_mean(split, window_tuple, ncomp)
                
            split.plot(file_name="%s/split_%d_%s_%s" % (plot_dir, k, sv, ar), color_range=[250, 100, 100])

            master_alms[sv, ar, k] = sph_tools.get_alms(split, window_tuple, niter, lmax)


ps_dict = {}
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

for id_sv1, sv1 in enumerate(surveys):
    arrays_1 = d["arrays_%s" % sv1]
    nsplits_1 = nsplit[sv1]
    
    if d["tf_%s" % sv1] is not None:
        _, _, tf1, _ = np.loadtxt(d["tf_%s" % sv1], unpack=True)
    else:
        tf1 = np.ones(len(lb))


    for id_ar1, ar1 in enumerate(arrays_1):
    
        for id_sv2, sv2 in enumerate(surveys):
            arrays_2 = d["arrays_%s" % sv2]
            nsplits_2 = nsplit[sv2]
            
            if d["tf_%s" % sv2] is not None:
                _, _, tf2, _ = np.loadtxt(d["tf_%s" % sv2], unpack=True)
            else:
                tf2 = np.ones(len(lb))

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
                    
                    if sv1 == sv2:
                        ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
                        spec_name_auto = "%s_%s_%sx%s_%s_auto" % (type, sv1, ar1, sv2, ar2)
                        ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / nsplit[sv1]
                        spec_name_noise = "%s_%s_%sx%s_%s_noise" % (type, sv1, ar1, sv2, ar2)

                so_spectra.write_ps(specDir + "/%s.dat" % spec_name_cross, lb, ps_dict_cross_mean, type, spectra=spectra)
                
                if sv1 == sv2:
                    so_spectra.write_ps(specDir+"/%s.dat" % spec_name_auto, lb, ps_dict_auto_mean, type, spectra=spectra)
                    so_spectra.write_ps(specDir+"/%s.dat" % spec_name_noise, lb, ps_dict_noise_mean, type, spectra=spectra)


