"""
This script computes the one dimensional power spectra of spt simulations.
Let me summarize our current understanding here.

SPT produce 500 simulations, and process them with the map maker filter, here we compute the power spectra of both input and output simulations.
In addition we also compute the effect of using the alm_mask while computing the spectra.
The description of simulations is provided in section 3 of https://pole.uchicago.edu/public/data/quan26/index.html.

Some important things:
1) The beams used for these simulations are not the final ones, they are the ones called input_beam_bl_{freq}ghz.txt.
2) There are two types of filtered simulations: "masking_no" and "masking_yes".

a) "masking_yes": is the standard set, which comprises the output maps produced with the masked timestream high-pass filter (this was the setting used for the real-data timestreams)
b) "masking_no": corresponds to the output maps produced with the masking turned off in the filtering process.

The baseline is therefore "masking_yes", "masking_no" can be used to understand filtering arfefacts.

3) Due to the TOD binning and processing, the out (filtered) spectra are expected to be convolved by piwin ** 6, this is taken into account in the mcms computation.

"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi, so_window
from pspipe_utils import pspipe_list, log
import numpy as np
import healpy as hp
import sys
import time
import gc
import sph_tools_mod

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

survey = "spt"
lmax = d["lmax"]
niter = d["niter"]
type = d["type"]
binning_file = d["binning_file"]
binned_mcm = d["binned_mcm"]
release_dir = d["release_dir"]
alm_conv = d[f"alm_conv_{survey}"]
also_masking_no = True
pure = d["pure"]

mcm_dir = "mcms"
sim_spec_dir = "sim_spectra_for_tf"

pspy_utils.create_directory(sim_spec_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
arrays_spt = d["arrays_spt"]


cases = ["nofilter", "filter_masking_yes", "nofilter_alm_mask", "filter_masking_yes_alm_mask"]
if also_masking_no == True:
    cases += ["filter_masking_no", "filter_masking_no_alm_mask"]


so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])


if pure == True:
    # precompute spinned_windows
    spinned_windows = {}
    for ar in arrays_spt:
        win_pol = so_map.read_map(d[f"window_pol_{survey}_{ar}"])
        spinned_windows[ar] = so_window.get_spinned_windows(win_pol, lmax, niter=niter)



for iii in subtasks:
    log.info(f"Simulation n° {iii:05d}/{d['iStop']:05d}")
    log.info(f"-------------------------")

    t0 = time.time()
    
    master_alms = {}
    
    for ar in arrays_spt:
    
    
        sim_in = so_map.read_map(f"{release_dir}/simulated_maps/input_maps/input_maps_realization{iii:03d}_{ar}ghz.fits")
        sim_out_masking_yes = so_map.read_map(f"{release_dir}/simulated_maps/output_maps/masking_yes/output_maps_masking_yes_realization{iii:03d}_{ar}ghz.fits")
        
        cal, pol_eff = d[f"cal_{survey}_{ar}_sim"], d[f"pol_eff_{survey}_{ar}_sim"]

        sim_in = sim_in.calibrate(cal=cal, pol_eff=pol_eff)
        sim_out_masking_yes = sim_out_masking_yes.calibrate(cal=cal, pol_eff=pol_eff)


        win_T = so_map.read_map(d[f"window_T_{survey}_{ar}"])
        win_pol = so_map.read_map(d[f"window_pol_{survey}_{ar}"])

        window_tuple = (win_T, win_pol)


        if pure == False:
            master_alms[survey, ar, "nofilter"] = sph_tools.get_alms(sim_in, window_tuple, niter, lmax, alm_conv=alm_conv)
            master_alms[survey, ar, "filter_masking_yes"] = sph_tools.get_alms(sim_out_masking_yes, window_tuple, niter, lmax, alm_conv=alm_conv)
        else:
            master_alms[survey, ar, "nofilter"] = sph_tools_mod.get_pure_alms(sim_in, window_tuple, spinned_windows[ar], niter, lmax, alm_conv=alm_conv)
            master_alms[survey, ar, "filter_masking_yes"] = sph_tools_mod.get_pure_alms(sim_out_masking_yes, window_tuple, spinned_windows[ar], niter, lmax, alm_conv=alm_conv)

    
        alm_mask = hp.read_alm(d[f"alm_mask_{survey}_{ar}"], hdu=1)
        alm_mask = hp.sphtfunc.resize_alm(alm_mask, d["lmax_mask"], d["lmax_mask"], lmax, lmax)

        master_alms[survey, ar, "nofilter_alm_mask"] = master_alms[survey, ar, "nofilter"] * alm_mask
        master_alms[survey, ar, "filter_masking_yes_alm_mask"] = master_alms[survey, ar, "filter_masking_yes"] * alm_mask
        
        del sim_in, sim_out_masking_yes
        gc.collect()

        if also_masking_no == True:
            sim_out_masking_no = so_map.read_map(f"{release_dir}/simulated_maps/output_maps/masking_no/output_maps_masking_no_realization{iii:03d}_{ar}ghz.fits")
            sim_out_masking_no = sim_out_masking_no.calibrate(cal=cal, pol_eff=pol_eff)
        
            if pure == False:
                master_alms[survey, ar, "filter_masking_no"] = sph_tools.get_alms(sim_out_masking_no, window_tuple, niter, lmax, alm_conv=alm_conv)
            else:
                master_alms[survey, ar, "filter_masking_no"] = sph_tools_mod.get_pure_alms(sim_out_masking_no, window_tuple, spinned_windows[ar], niter, lmax, alm_conv=alm_conv)

            master_alms[survey, ar, "filter_masking_no_alm_mask"] = master_alms[survey, ar, "filter_masking_no"] * alm_mask


    _, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
        
    n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)
        
    for i_spec in range(n_spec):
        sv1, ar1, sv2, ar2 = sv1_list[i_spec], ar1_list[i_spec], sv2_list[i_spec], ar2_list[i_spec]
        spec_name = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}"
        
        mbb_inv_dict = {}

        mbb_inv_dict["nofilter"], _ = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}_sim_in", spin_pairs=spin_pairs)
        mbb_inv_dict["nofilter_alm_mask"] = mbb_inv_dict["nofilter"]
        
        mbb_inv_dict["filter_masking_yes"], _ = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}_sim_out", spin_pairs=spin_pairs)
        mbb_inv_dict["filter_masking_yes_alm_mask"]  = mbb_inv_dict["filter_masking_yes"]
        mbb_inv_dict["filter_masking_no"]  = mbb_inv_dict["filter_masking_yes"]
        mbb_inv_dict["filter_masking_no_alm_mask"]  = mbb_inv_dict["filter_masking_yes"]

        for case in cases:

            l, ps_master = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, case], master_alms[sv2, ar2, case], spectra=spectra)

            lb, ps = so_spectra.bin_spectra(l,
                                            ps_master,
                                            binning_file,
                                            lmax,
                                            type=type,
                                            mbb_inv=mbb_inv_dict[case],
                                            spectra=spectra,
                                            binned_mcm=binned_mcm)

            so_spectra.write_ps(sim_spec_dir + f"/{spec_name}_{case}_{iii:05d}.dat", lb, ps, type, spectra=spectra)

    log.info(f"[{iii}]  Simulation n° {iii:05d} done in {time.time()-t0:.02f} s")
