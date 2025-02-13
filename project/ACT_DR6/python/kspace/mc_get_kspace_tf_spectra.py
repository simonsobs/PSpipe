"""
This script generate simplistic signa-only simulations of the actpol data that will be used to measure the transfer function
This is essentially a much simpler version of mc_get_spectra.py, since it doesn't include noise on the simulation and thus does not require
different splits of the data
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi, so_map_preprocessing
from pspipe_utils import pspipe_list, kspace, simulation, misc, log
import numpy as np
import sys
import time
from pixell import curvedsky, powspec

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
apply_kspace_filter = d["apply_kspace_filter"]

if sim_alm_dtype == "complex64":
    sim_alm_dtype = np.complex64
elif sim_alm_dtype == "complex128":
    sim_alm_dtype = np.complex128

mcm_dir = "mcms"
tf_dir = "sim_spectra_for_tf"
bestfit_dir = "best_fits"
ps_model_dir = "noise_model"

pspy_utils.create_directory(tf_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

arrays, templates, filters, filter_dicts = {}, {}, {}, {}
log.info(f"build template and filter")
for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    template_name = d[f"maps_{sv}_{arrays[sv][0]}"][0]
    templates[sv] = so_map.read_map(template_name)
    assert (templates[sv].pixel == "CAR"), "we only compute kspace tf in CAR pixellisation"
    filter_dicts[sv] = d[f"k_filter_{sv}"]
    filters[sv] = kspace.get_kspace_filter(templates[sv], filter_dicts[sv], dtype=np.float32)

f_name_cmb = bestfit_dir + "/cmb.dat"
ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)

f_name_fg = bestfit_dir + "/fg_{}x{}.dat"
array_list = [f"{sv}_{array}" for sv in surveys for array in arrays[sv]]
l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, array_list, lmax, spectra)

assert (apply_kspace_filter == True), "this has to be set to True"

scenarios = ["standard", "noE", "noB"]

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    log.info(f"Simulation n° {iii:05d}/{d['iStop']:05d}")
    log.info(f"-------------------------")

    t0 = time.time()

    alms = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
    fglms = simulation.generate_fg_alms(fg_mat, array_list, lmax)

    for scenario in scenarios:
        log.info(f"[{iii}]  scenario {scenario}")

        master_alms = {}

        for sv in surveys:
            for ar_id, ar in enumerate(arrays[sv]):

                win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
                win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])

                window_tuple = (win_T, win_pol)

                # we add fg alms to cmb alms in temperature
                alms_beamed = alms.copy()
                alms_beamed += fglms[f"{sv}_{ar}"]

                # we convolve signal + foreground with the beam of the array
                l, bl = misc.read_beams(d[f"beam_T_{sv}_{ar}"], d[f"beam_pol_{sv}_{ar}"])
                alms_beamed = misc.apply_beams(alms_beamed, bl)

                if scenario == "noE": alms_beamed[1] *= 0
                if scenario == "noB": alms_beamed[2] *= 0

                # generate our signal only sim
                split = sph_tools.alm2map(alms_beamed, templates[sv])

                # compute the alms of the sim

                master_alms[sv, ar, "nofilter"] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)

                # apply the k-space filter

                win_kspace = so_map.read_map(d[f"window_kspace_{sv}_{ar}"])

                split = kspace.filter_map(split,
                                          filters[sv],
                                          win_kspace,
                                          weighted_filter=filter_dicts[sv]["weighted"],
                                          use_ducc_rfft=True)

                # compute the alms of the filtered sim
                master_alms[sv, ar, "filter"] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)


        ps_dict = {}
        _, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
        
        n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)
        
        for i_spec in range(n_spec):
            sv1, ar1, sv2, ar2 = sv1_list[i_spec], ar1_list[i_spec], sv2_list[i_spec], ar2_list[i_spec]

            spec_name = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}"

            mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}",
                                                spin_pairs=spin_pairs)

            # we  compute the power spectra of the sim (with and without the k-space filter applied)

            for filt in ["filter", "nofilter"]:

                l, ps_master = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, filt],
                                                             master_alms[sv2, ar2, filt],
                                                             spectra=spectra)

                lb, ps = so_spectra.bin_spectra(l,
                                                ps_master,
                                                binning_file,
                                                lmax,
                                                type=type,
                                                mbb_inv=mbb_inv,
                                                spectra=spectra,
                                                binned_mcm=binned_mcm)

                so_spectra.write_ps(tf_dir + f"/{spec_name}_{filt}_{scenario}_{iii:05d}.dat", lb, ps, type, spectra=spectra)

    log.info(f"[{iii}]  Simulation n° {iii:05d} done in {time.time()-t0:.02f} s")
