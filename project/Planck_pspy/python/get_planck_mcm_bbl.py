'''
This script is used to compute the mode coupling matrices of the Planck data.
The inputs for the script are the Planck beam and likelihood masks.
To run it:
python get_planck_mcm_Bbl.py global.dict
'''
import numpy as np
import healpy as hp
from pspy import so_dict, so_map, so_mcm, pspy_utils
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

windows_dir = "windows"
mcm_dir = "mcms"

pspy_utils.create_directory(windows_dir)
pspy_utils.create_directory(mcm_dir)

freqs = d["freqs"]
niter = d["niter"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
pixwin = d["pixwin"]
splits = d["splits"]
experiment = "Planck"

print("Compute Planck 2018 mode coupling matrices")

for f1, freq1 in enumerate(freqs):
    
    window_t_1 = d["window_T_%s" % freq1]
    window_pol_1 = d["window_pol_%s" % freq1]

    for count1, hm1 in enumerate(splits):
        
        win_t1 = so_map.read_map(window_t_1[count1])
        win_pol1 = so_map.read_map(window_pol_1[count1])
        
        win_t1.write_map("%s/window_T_%s_%s-%s.fits" % (windows_dir, experiment, freq1, hm1))
        win_pol1.write_map("%s/window_P_%s_%s-%s.fits" % (windows_dir, experiment, freq1, hm1))

        window_tuple1 = (win_t1, win_pol1)
        
        del win_t1, win_pol1
    
        l, bl1_t = np.loadtxt(d["beam_%s_%s_T" % (freq1, hm1)], unpack=True)
        l, bl1_pol = np.loadtxt(d["beam_%s_%s_pol" % (freq1, hm1)], unpack=True)

        if pixwin == True:
            bl1_t *= hp.pixwin(window_tuple1[0].nside)[:len(bl1_t)]
            bl1_pol *= hp.pixwin(window_tuple1[0].nside)[:len(bl1_pol)]
        
        bl_tuple1 = (bl1_t, bl1_pol)

        for f2, freq2 in enumerate(freqs):
            if f1 > f2: continue
            
            window_t_2 = d["window_T_%s" % freq2]
            window_pol_2 = d["window_pol_%s" % freq2]

            for count2, hm2 in enumerate(splits):
                if (count1 > count2) & (f1 == f2): continue
                
                print(freq1, freq2)

                win_t2 = so_map.read_map(window_t_2[count2])
                win_pol2 = so_map.read_map(window_pol_2[count2])

                window_tuple2 = (win_t2, win_pol2)

                del win_t2, win_pol2
                
                l, bl2_t = np.loadtxt(d["beam_%s_%s_T" % (freq2, hm2)], unpack=True)
                l, bl2_pol = np.loadtxt(d["beam_%s_%s_pol" % (freq2, hm2)], unpack=True)

                if pixwin == True:
                    bl2_t *= hp.pixwin(window_tuple2[0].nside)[:len(bl2_t)]
                    bl2_pol *= hp.pixwin(window_tuple2[0].nside)[:len(bl2_pol)]

                bl_tuple2 = (bl2_t, bl2_pol)
                
                mcm_inv, mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(win1=window_tuple1,
                                                                     win2=window_tuple2,
                                                                     binning_file=binning_file,
                                                                     bl1=bl_tuple1,
                                                                     bl2=bl_tuple2,
                                                                     lmax=lmax,
                                                                     niter=niter,
                                                                     type=type,
                                                                     unbin=True,
                                                                     save_file="%s/%s_%sx%s_%s-%sx%s" % (mcm_dir, experiment, freq1, experiment, freq2, hm1, hm2))




