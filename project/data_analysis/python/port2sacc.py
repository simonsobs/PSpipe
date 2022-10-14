"""
This script store data such as spectra, covmat, bandpasses into a sacc file
"""

import os
import re
import sys

import numpy as np
import sacc
from pspipe_utils import covariance, external_data, pspipe_list
from pspy import pspy_utils, so_dict

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spec_dir = "sim_spectra"
spec_name_list = pspipe_list.get_spec_name_list(d, char="_")
spectra_order = ["TT", "TE", "ET", "EE"]
cov_order = pspipe_list.x_ar_cov_order(spec_name_list, spectra_order=spectra_order)

# Regex to filter cov order
regex = re.compile("([TEB]{2})_(.*)x(.*)")
blocks = [regex.match(block).groups() for block in cov_order]

# Retrieving the wafer names
*_, wafers = pspipe_list.get_arrays_list(d)

# Reading binning file
lmax = d["lmax"]
binning_file = d["binning_file"]
bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(bin_hi)

# Reading beams
beams = {wafer: pspy_utils.read_beam_file(d[f"beam_dr6_{wafer}"]) for wafer in wafers}

# Reading passbands : the passband file should be within the dict file
passbands = external_data.get_passband_dict_dr6(wafers)

# Reading covariance
cov_dir = "covariances"
analytic_cov = np.load(os.path.join(cov_dir, "x_ar_analytic_cov_with_beam.npy"))

iStart = d["iStart"]
iStop = d["iStop"]
for i in range(iStart, iStop):
    print(f"Storing simulation n°{i}...")

    # Saving into sacc format
    act_sacc = sacc.Sacc()
    for wafer in wafers:
        for spin, quantity in zip([0, 2], ["temperature", "polarization"]):

            nus, passband = passbands.get(wafer)
            ell, beam = beams.get(wafer)

            act_sacc.add_tracer(
                "NuMap",
                f"dr6_{wafer}_s{spin}",
                quantity=f"cmb_{quantity}",
                spin=spin,
                nu=nus,
                bandpass=passband,
                ell=ell,
                beam=beam,
            )

    # Reading the flat data vector
    data_vec = covariance.read_x_ar_spectra_vec(
        spec_dir, spec_name_list, f"cross_{i:05d}", spectra_order=spectra_order, type=d["type"]
    )

    for count, (spec, s1, s2) in enumerate(blocks):
        print(f"Adding {s1}x{s2}, {spec} spectrum")

        # Get Bbl
        mcm_dir = "mcms"
        Bbl = np.load(os.path.join(mcm_dir, f"{s1}x{s2}_Bbl_spin0xspin0.npy"))
        ls_w = np.arange(2, Bbl.shape[-1] + 2)
        bp_window = sacc.BandpowerWindow(ls_w, Bbl.T)

        # Define tracer names and cl type
        pa, pb = spec
        ta_name = f"{s1}_s0" if pa == "T" else f"{s1}_s2"
        tb_name = f"{s2}_s0" if pb == "T" else f"{s2}_s2"

        map_types = {"T": "0", "E": "e", "B": "b"}
        if pb == "T":
            cl_type = "cl_" + map_types[pb] + map_types[pa]
        else:
            cl_type = "cl_" + map_types[pa] + map_types[pb]

        # Add ell/cl to sacc
        Db = data_vec[count * n_bins : (count + 1) * n_bins]
        act_sacc.add_ell_cl(cl_type, ta_name, tb_name, lb, Db, window=bp_window)

    print("Adding covariance")
    act_sacc.add_covariance(analytic_cov)

    print("Writing sacc file")
    act_sacc.save_fits(f"act_simu_sacc_{i:05d}.fits", overwrite=True)
