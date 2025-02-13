"""
This script generate simplistic signa-only simulations of the actpol data that will be used to measure the effect of aberration
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi, so_map_preprocessing
from pspipe_utils import pspipe_list, kspace, simulation, misc, log
import numpy as np
import sys
import time
from pixell import curvedsky, powspec, aberration

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
niter = d["niter"]
type = d["type"]
binning_file = d["binning_file"]
sim_alm_dtype = d["sim_alm_dtype"]
binned_mcm = d["binned_mcm"]

if sim_alm_dtype == "complex64":
    sim_alm_dtype = np.complex64
elif sim_alm_dtype == "complex128":
    sim_alm_dtype = np.complex128

mcm_dir = "mcms"
ab_dir = "sim_spectra_aberration"
bestfit_dir = "best_fits"
plot_dir = "plots/aberration"

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(ab_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

arrays, templates = {}, {}
log.info(f"build template and filter")
for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    template_name = d[f"maps_{sv}_{arrays[sv][0]}"][0]
    templates[sv] = so_map.read_map(template_name)
    assert (templates[sv].pixel == "CAR"), "we only compute aberation in CAR pixellisation"

f_name_cmb = bestfit_dir + "/cmb.dat"
ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)

f_name_fg = bestfit_dir + "/fg_{}x{}.dat"
map_set_list = [f"{sv}_{array}" for sv in surveys for array in arrays[sv]]
l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, map_set_list, lmax, spectra)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

plot = True

color_range = [500, 150, 150]

for iii in subtasks:
    log.info(f"Simulation n° {iii:05d}/{d['iStop']:05d}")
    log.info(f"-------------------------")

    t0 = time.time()

    alms = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
    fglms = simulation.generate_fg_alms(fg_mat, map_set_list, lmax)

    master_alms = {}

    for sv in surveys:
        for ar_id, ar in enumerate(arrays[sv]):

            win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
            win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])
            
            freq = float(d[f"freq_info_{sv}_{ar}"]["freq_tag"] * 10 ** 9)

            window_tuple = (win_T, win_pol)

            # we add fg alms to cmb alms in temperature
            alms_beamed = alms.copy()
            alms_beamed += fglms[f"{sv}_{ar}"]

            # we convolve signal + foreground with the beam of the array
            l, bl = misc.read_beams(d[f"beam_T_{sv}_{ar}"], d[f"beam_pol_{sv}_{ar}"])
            alms_beamed = misc.apply_beams(alms_beamed, bl)

            # generate our signal only sim
            split = sph_tools.alm2map(alms_beamed, templates[sv])

            # compute the alms of the sims pre aberation
            master_alms[sv, ar, "no_aberration"] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)

            t1 = time.time()

            split_ab = split.copy()
            split_ab.data = aberration.boost_map(split_ab.data, beta=aberration.beta, freq=freq)
        
            log.info(f"[n° {iii:05d}] {sv} {ar} aberration took {time.time() - t1} s")

            # compute the alms of the aberated sims
            master_alms[sv, ar, "aberration"] = sph_tools.get_alms(split_ab, window_tuple, niter, lmax, dtype=sim_alm_dtype)
            
            if (plot == True) & (iii == 0):
                split = split.downgrade(4)
                split_ab = split_ab.downgrade(4)
                split.plot(file_name=f"{plot_dir}/split_{sv}_{ar}", color_range=color_range)
                split_ab.plot(file_name=f"{plot_dir}/split_{sv}_{ar}_aberrated", color_range=color_range)
                split.data -= split_ab.data
                split.plot(file_name=f"{plot_dir}/diff_split_{sv}_{ar}", color_range=np.array(color_range)/10)

        ps_dict = {}
        _, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
        
        n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)
        
        for i_spec in range(n_spec):
            sv1, ar1, sv2, ar2 = sv1_list[i_spec], ar1_list[i_spec], sv2_list[i_spec], ar2_list[i_spec]

            spec_name = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}"

            mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}",
                                                spin_pairs=spin_pairs)

            # we  compute the power spectra of the sim (with and without the aberration applied)

            for ab in ["no_aberration", "aberration"]:

                l, ps_master = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, ab],
                                                             master_alms[sv2, ar2, ab],
                                                             spectra=spectra)

                lb, ps = so_spectra.bin_spectra(l,
                                                ps_master,
                                                binning_file,
                                                lmax,
                                                type=type,
                                                mbb_inv=mbb_inv,
                                                spectra=spectra,
                                                binned_mcm=binned_mcm)

                so_spectra.write_ps(ab_dir + f"/{spec_name}_{ab}_{iii:05d}.dat", lb, ps, type, spectra=spectra)

    log.info(f"[{iii}]  Simulation n° {iii:05d} done in {time.time()-t0:.02f} s")
