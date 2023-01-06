"""
This script generate simplistic signa-only simulations of the actpol data that will be used to measure the transfer function
This is essentially a much simpler version of mc_get_spectra.py, since it doesn't include noise on the simulation and thus does not require
different splits of the data
"""

from pspy import pspy_utils, so_dict, so_map, sph_tools, so_mcm, so_spectra, so_mpi, so_map_preprocessing
from pspipe_utils import pspipe_list, kspace, simulation, misc
import numpy as np
import sys
import time
from pixell import curvedsky, powspec

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
if len(sys.argv) > 2: np.random.seed(int(sys.argv[2]))

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

window_dir = "windows"
mcm_dir = "mcms"
tf_dir = "sim_spectra_for_tf"
bestfit_dir = "best_fits"
ps_model_dir = "noise_model"

pspy_utils.create_directory(tf_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

# let's list the different frequencies used in the code
#freq_list = pspipe_list.get_freq_list(d)

# we read cmb and fg best fit power spectrum
# we put the best fit power spectrum in a matrix [nfreqs, nfreqs, lmax]
# taking into account the correlation of the fg between different frequencies

f_name_cmb = bestfit_dir + "/cmb.dat"
f_name_fg = bestfit_dir + "/fg_{}x{}.dat"

ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)
fg_mat = {}

arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
array_list = [f"{sv}_{array}" for sv in surveys for array in arrays[sv]]
l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, array_list, lmax, spectra)

assert (apply_kspace_filter == True), "this has to be set to True"

# prepare the filters

template = {}
filter = {}
for sv in surveys:
    arrays = d[f"arrays_{sv}"]
    template_name = d[f"maps_{sv}_{arrays[0]}"][0]
    template[sv] = so_map.read_map(template_name)
    assert (template[sv].pixel == "CAR"), "we only compute kspace tf in CAR pixellisation"
    ks_f = d[f"k_filter_{sv}"]
    filter[sv] = kspace.get_kspace_filter(template[sv], ks_f)

# the filter also introduce E->B leakage, in order to measure it we run the scenario where there
# is no E or B modes
scenarios = ["standard", "noE", "noB"]

# we will use mpi over the number of simulations
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    t0 = time.time()

    alms = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
    fglms = simulation.generate_fg_alms(fg_mat, array_list, lmax)

    for scenario in scenarios:

        master_alms = {}

        for sv in surveys:

            arrays = d[f"arrays_{sv}"]
            ks_f = d[f"k_filter_{sv}"]

            for ar_id, ar in enumerate(arrays):

                win_T = so_map.read_map(d[f"window_T_{sv}_{ar}"])
                win_pol = so_map.read_map(d[f"window_pol_{sv}_{ar}"])

                window_tuple = (win_T, win_pol)

                del win_T, win_pol

                # we add fg alms to cmb alms in temperature
                alms_beamed = alms.copy()
                alms_beamed += fglms[f"{sv}_{ar}"]

                # we convolve signal + foreground with the beam of the array
                l, bl = pspy_utils.read_beam_file(d[f"beam_{sv}_{ar}"])
                alms_beamed = curvedsky.almxfl(alms_beamed, bl)

                if scenario == "noE": alms_beamed[1] *= 0
                if scenario == "noB": alms_beamed[2] *= 0

                # generate our signal only sim
                split = sph_tools.alm2map(alms_beamed, template[sv])

                # compute the alms of the sim

                master_alms[sv, ar, "nofilter"] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)

                # apply the k-space filter

                binary_file = misc.str_replace(d[f"window_T_{sv}_{ar}"], "window_", "binary_")
                binary = so_map.read_map(binary_file)

                split = kspace.filter_map(split,
                                          filter[sv],
                                          binary,
                                          weighted_filter=ks_f["weighted"])

                # compute the alms of the filtered sim

                master_alms[sv, ar, "filter"] = sph_tools.get_alms(split, window_tuple, niter, lmax, dtype=sim_alm_dtype)

                print(scenario, sv, ar, time.time()-t0)

        ps_dict = {}
        _, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)

        for id_sv1, sv1 in enumerate(surveys):
            arrays_1 = d[f"arrays_{sv1}"]
            for id_ar1, ar1 in enumerate(arrays_1):
                for id_sv2, sv2 in enumerate(surveys):
                    arrays_2 = d[f"arrays_{sv2}"]
                    for id_ar2, ar2 in enumerate(arrays_2):

                        if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                        if  (id_sv1 > id_sv2) : continue

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


                            so_spectra.write_ps(tf_dir + f"/{spec_name}_{filt}_{scenario}_%05d.dat" % iii, lb, ps, type, spectra=spectra)

    print("sim number %05d done in %.02f s" % (iii, time.time()-t0))
