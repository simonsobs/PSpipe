"""
This script compute the auto and cross power spectra for the different scanning strategy
"""

def downgraded_plot(map, file_name, color_range=None, down_factor=4):
    map_low_res = map.copy()
    map_low_res = map_low_res.downgrade(down_factor)
    map_low_res.plot(file_name=file_name, color_range=color_range)
    
import healpy as hp
import pylab as plt
import numpy as np
import sys
from pspy import so_map, so_window, sph_tools, so_mcm, pspy_utils, so_spectra, so_dict, so_mpi
import SO_noise_utils

from soapack import interfaces
from mnms import noise_models as nm

# map regions to qids
from astropy.io import ascii
strats = ascii.read("so_scan_strategies.csv")
strats.add_index("region")
def getqid(region):
    return strats.loc[region]["qid"]


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

scan_list = d["scan_list"]
lmax = d["lmax"]
apo_type = d["apo_type"]
apo_radius_degree = d["apo_radius_degree"]
niter = d["niter"]
spectra = d["spectra"]
split_list = d["split_list"]
runs = d["runs"]
clfile = d["clfile"]
binning_file = d["binning_file_name"]
bin_size = d["bin_size"]
color_range = d["map_plot_range"]
skip_from_edges = d["skip_from_edges"]
cross_linking_threshold = d["cross_linking_threshold"]
K_to_muK = 10**6
include_cmb = d["include_CMB"]

if len(sys.argv) > 2:
    sim_idx = int(sys.argv[2])
else:
    sim_idx = int(d["sim_idx"])
print("SIMULATING   ", sim_idx)

dg = d["downgrade"]

lth, ps_theory = pspy_utils.ps_lensed_theory_to_dict(clfile, "Dl", lmax=lmax)

pspy_utils.create_binning_file(bin_size=bin_size, n_bins=300, file_name=binning_file )

map_dir = d["map_dir"]
mcm_dir = "mcms"
spectra_dir = "spectra"
sim_spectra_dir = "sim_spectra"
plot_dir = "plot/map"
window_dir = "windows"

pspy_utils.create_directory(window_dir)
pspy_utils.create_directory(mcm_dir)
pspy_utils.create_directory(spectra_dir)
pspy_utils.create_directory(sim_spectra_dir)
pspy_utils.create_directory(plot_dir)

template = so_map.read_map("%s/lat01_s25_small_tiles_fullfp_f150_1pass_2way_set00_map.fits" % (map_dir)) #downgrade(dg)
np.random.seed((sim_idx, 2022))  # set seed for cmb
cmb = template.synfast(clfile)

n_scans = len(scan_list)
print("number of scan to compute : %s" % n_scans)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_scans - 1)

for task in subtasks:

    task = int(task)
    scan = scan_list[task]

    data_model = interfaces.SOScanS0002()
    downgrade = dg
    sim_type = "tile"
    qid = getqid(scan)
    print("QID: ", qid)

    if sim_type == 'wav':
        model = nm.WaveletNoiseModel(qid, data_model=data_model, downgrade=downgrade)
    elif sim_type == 'tile':
        model = nm.TiledNoiseModel(qid, data_model=data_model, downgrade=downgrade,
                                fwhm_ivar=1, delta_ell_smooth=100, 
                                width_deg=3., height_deg=3.)

    binary = so_map.read_map("%s/lat01_s25_%s_fullfp_f150_1pass_2way_set00_div.fits" % (map_dir, scan)) #.downgrade(dg)
    binary.data[:] = 1
    
    sim = {}
    for count, split in enumerate(split_list):

        model.get_model(check_on_disk=True, keep_model=True, verbose=True)
        alm = model.get_sim(count, sim_idx, check_on_disk=False, alm=True)[0,0,:,:]
        noise_map = template.copy()
        noise_map = sph_tools.alm2map(alm, noise_map)

        # noise_map = so_map.read_map("%s/lat01_s25_%s_fullfp_f150_1pass_2way_set%02d_map.fits" % (map_dir, scan, count)).downgrade(dg)
        binary.data[noise_map.data[0] == 0] = 0
        noise_map.data[:] *= K_to_muK # should it be * np.sqrt(2) ?
        
        
        if include_cmb == True:
            sim[split] = cmb.copy()
            sim[split].data[:] += noise_map.data[:]
        else:
            sim[split] = noise_map.copy()
    

    for run in runs:
                
        # window.write_map("%s/window_%s_%s.fits" % (window_dir, scan, run))
        window = so_map.read_map("%s/window_%s_%s_xlink%s.fits" % (window_dir, scan, run, str(cross_linking_threshold))) #.downgrade(dg)


    
        window = (window, window)
    
        mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(window,
                                                    binning_file,
                                                    lmax=lmax,
                                                    type="Dl",
                                                    niter=niter,
                                                    save_file="%s/%s_%s_%s"%(mcm_dir, scan, run, str(cross_linking_threshold)))
    
        alm = {}
        for split in split_list:
            alm[split] = sph_tools.get_alms(sim[split], window, niter, lmax)
            
        for c0, s0 in enumerate(split_list):
            for c1, s1 in enumerate(split_list):
                if c1 > c0: continue

                spec_name = "%s_%sx%s_%s_xlink%s" % (scan, s0, s1, run, str(cross_linking_threshold))

                l, ps = so_spectra.get_spectra(alm[s0], alm[s1], spectra=spectra)
                lb, Db_dict = so_spectra.bin_spectra(l,
                                                     ps,
                                                     binning_file,
                                                     lmax,
                                                     type="Dl",
                                                     mbb_inv=mbb_inv,
                                                     spectra=spectra)

                so_spectra.write_ps("%s/spectra_%s_%s.dat" % (sim_spectra_dir, spec_name, sim_idx),
                                    lb,
                                    Db_dict,
                                    type="Dl",
                                    spectra=spectra)

        
