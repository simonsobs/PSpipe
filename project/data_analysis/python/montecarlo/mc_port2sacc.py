"""
This script store data such as spectra, covmat, bandpasses into a sacc file
"""

import os
import re
import sys

import numpy as np
import sacc
from pspipe_utils import external_data, pspipe_list, covariance
from pspy import pspy_utils, so_dict

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

use_mc_corrected_cov = True
only_diag_corrections = False

cov_name = "analytic_cov"
if use_mc_corrected_cov:
    if only_diag_corrections:
        cov_name += "_with_diag_mc_corrections"
    else:
        cov_name += "_with_mc_corrections"

mcm_dir = "mcms"
bestfit_dir = "best_fits"
sim_spec_dir = "sim_spectra"
sim_sacc_dir = "sim_sacc"


pspy_utils.create_directory(sim_sacc_dir)


#spec_name_list, nu_eff_list = pspipe_list.get_spec_name_list(d, char="_", return_nueff=True)
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
passbands = {}
do_bandpass_integration = d["do_bandpass_integration"]
for wafer in wafers:
    freq_info = d[f"freq_info_dr6_{wafer}"]
    if do_bandpass_integration:
        nu_ghz, passband = np.loadtxt(freq_info["passband"]).T
    else:
        nu_ghz, passband = np.array([freq_info["freq_tag"]]), np.array([1.])

    passbands[wafer] = [nu_ghz, passband]

# Reading covariance
like_product_dir = "like_product"
analytic_cov = np.load(os.path.join(like_product_dir, f"x_ar_{cov_name}_with_beam.npy"))
inv_analytic_cov = np.linalg.inv(analytic_cov)


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
        sim_spec_dir, spec_name_list, f"cross_{i:05d}", spectra_order=spectra_order, type=d["type"]
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
    act_sacc.save_fits(f"{sim_sacc_dir}/act_simu_sacc_{i:05d}.fits", overwrite=True)

    print("Check chi2")
    theory_vec = covariance.read_x_ar_theory_vec(bestfit_dir,
                                                 mcm_dir,
                                                 spec_name_list,
                                                 lmax,
                                                 spectra_order = ["TT", "TE", "ET", "EE"])

    res = data_vec - theory_vec
    chi2 = res @ inv_analytic_cov @ res
    print(r"$\chi^{2}$/DoF = %.2f/%d" % (chi2, len(data_vec)))
