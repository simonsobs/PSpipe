"""
This script compute all power spectra and write them to disk.
look similar to old pspipe with two diff, first we don't have anything related to kspace filter, second we do
the pixwin deconvolution at the mcms level (so it is not in this script).
"""

import sys

import healpy as hp
import numpy as np
from pspipe_utils import log, pspipe_list
from pspy import pspy_utils, so_dict, so_mcm, so_mpi, so_spectra, so_map

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
type = d["type"]
binning_file = d["binning_file"]
write_all_spectra = d["write_splits_spectra"]
binned_mcm = d["binned_mcm"]

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

mcm_dir = "mcms"
spec_dir = "spectra"
alms_dir = "alms"

pspy_utils.create_directory(spec_dir)

master_alms, nsplit, arrays, templates, filter_dicts = {}, {}, {}, {}, {}
# read the alms

for sv in surveys:
    arrays[sv] = d[f"arrays_{sv}"]
    templates[sv] = so_map.read_map(d[f"window_T_{sv}_{arrays[sv][0]}"])

    for ar in arrays[sv]:
        maps = d[f"maps_{sv}_{ar}"]
        nsplit[sv] = len(maps)
        log.info(f"{nsplit[sv]} split of survey: {sv}, array {ar}")
        for k, map in enumerate(maps):
            master_alms[sv, ar, k] = np.load(f"{alms_dir}/alms_{sv}_{ar}_{k}.npy")
            log.debug(f"{alms_dir}/alms_{sv}_{ar}_{k}.npy")

# compute the transfer functions
_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(lb)
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")


n_spec, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)
log.info(f"number of spectra to compute : {n_spec}")
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_spec - 1)

for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = sv1_list[task], ar1_list[task], sv2_list[task], ar2_list[task]

    log.info(f"[{task}] Computing spectra for {sv1}_{ar1} x {sv2}_{ar2}")

    ps_dict = {}
    for spec in spectra:
        ps_dict[spec, "auto"] = []
        ps_dict[spec, "cross"] = []

    nsplits_1 = nsplit[sv1]
    nsplits_2 = nsplit[sv2]

    spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
    mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}",
                                        spin_pairs=spin_pairs)


    for s1 in range(nsplits_1):
        for s2 in range(nsplits_2):
            if (sv1 == sv2) & (ar1 == ar2) & (s1>s2) : continue

            l, ps_master = so_spectra.get_spectra_pixell(master_alms[sv1, ar1, s1],
                                                         master_alms[sv2, ar2, s2],
                                                         spectra=spectra)

            spec_name=f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_{s1}{s2}"

            lb, ps = so_spectra.bin_spectra(l,
                                            ps_master,
                                            binning_file,
                                            lmax,
                                            type=type,
                                            mbb_inv=mbb_inv,
                                            spectra=spectra,
                                            binned_mcm=binned_mcm)


            if write_all_spectra:
                so_spectra.write_ps(spec_dir + f"/{spec_name}.dat", lb, ps, type, spectra=spectra)

            for count, spec in enumerate(spectra):
                if (s1 == s2) & (sv1 == sv2):
                    if count == 0: log.debug(f"[{task}] auto {sv1}_{ar1} x {sv2}_{ar2} {s1}{s2}")
                    ps_dict[spec, "auto"] += [ps[spec]]
                else:
                    if count == 0: log.debug(f"[{task}] cross {sv1}_{ar1} x {sv2}_{ar2} {s1}{s2}")
                    ps_dict[spec, "cross"] += [ps[spec]]

    ps_dict_auto_mean = {}
    ps_dict_cross_mean = {}
    ps_dict_noise_mean = {}

    for spec in spectra:
        ps_dict_cross_mean[spec] = np.mean(ps_dict[spec, "cross"], axis=0)
        spec_name_cross = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_cross"

        if ar1 == ar2 and sv1 == sv2:
            # Average TE / ET so that for same array same season TE = ET
            ps_dict_cross_mean[spec] = (np.mean(ps_dict[spec, "cross"], axis=0) + np.mean(ps_dict[spec[::-1], "cross"], axis=0)) / 2.

        if sv1 == sv2:
            ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
            spec_name_auto = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_auto"
            ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / nsplit[sv1]
            spec_name_noise = f"{type}_{sv1}_{ar1}x{sv2}_{ar2}_noise"

    so_spectra.write_ps(spec_dir + f"/{spec_name_cross}.dat", lb, ps_dict_cross_mean, type, spectra=spectra)
    if sv1 == sv2:
        so_spectra.write_ps(spec_dir + f"/{spec_name_auto}.dat", lb, ps_dict_auto_mean, type, spectra=spectra)
        so_spectra.write_ps(spec_dir + f"/{spec_name_noise}.dat", lb, ps_dict_noise_mean, type, spectra=spectra)
