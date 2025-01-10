"""
This script store data such as spectra, covmat, bandpasses into a sacc file
"""

import importlib
import os
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pspipe import conventions as cvt
from pspipe_utils import beam_chromaticity, covariance, io, log, misc, pspipe_list
from pspy import pspy_utils, so_dict

matplotlib.use("Agg")

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

# choose which type of sacc you to create: data_sacc, ng_simu_sacc, simu_sacc, simu_and_syst_sacc
sacc_fname = "data_sacc"

# Set covariance file name
mcm_dir = cvt.get_mcms_dir()
cov_dir = cvt.get_covariances_dir()
sacc_dir = cvt.get_sacc_dir(create=True)

spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, delimiter="_", return_nu_tag=True)

# Get spectra/cov orders
spectra_order = cvt.get_spectra_order()

cov_order = pspipe_list.x_ar_cov_order(spec_name_list, nu_tag_list, spectra_order=spectra_order)

# Build list of the different map_set
map_set_list = pspipe_list.get_map_set_list(d)

# Reading passbands : the passband file should be within the dict file
passbands = {}
for map_set in map_set_list:
    freq_info = d[f"freq_info_{map_set}"]
    if d["do_bandpass_integration"]:
        nu_ghz, passband = np.loadtxt(freq_info["passband"]).T
    else:
        nu_ghz, passband = np.array([freq_info["freq_tag"]]), np.array([1.0])

    passbands[f"{map_set}"] = [nu_ghz, passband]

log.debug(f"Passband information: {passbands} \n")

# Get Bbl : same bbl for every cross so load one and then assign it to every cross (to make the code
# as versatile as possible)
bbl = np.load(os.path.join(mcm_dir, f"{spec_name_list[0]}_Bbl_spin0xspin0.npy"))
bbls = {cross: bbl for cross in spec_name_list}

beams = None
if d["include_beam_chromaticity_effect_in_sacc"]:
    log.info(f"include beam array accounting for beam chromaticity \n")

    # Get beam chromaticity
    beams = {}
    alpha_dict, nu_ref_dict = beam_chromaticity.act_dr6_beam_scaling()
    for map_set in map_set_list:
        bl_mono_file_name = d[f"beam_mono_{map_set}"]
        l, bl = pspy_utils.read_beam_file(bl_mono_file_name, lmax=10000)
        l, nu_array, bl_nu = beam_chromaticity.get_multifreq_beam(l,
                                                                  bl,
                                                                  passbands[map_set],
                                                                  nu_ref_dict[map_set],
                                                                  alpha_dict[map_set])
        beams[map_set] = [l, dict(T=bl_nu, E=bl_nu)]

# Define metadata such as dict content or libraries version
metadata = dict(
    author=d.get("author", "SO Collaboration PS Task Force"),
    date=d.get("date", datetime.today().strftime("%Y-%m-%d %H:%M:%S")),
)
modules = [
    "camb",
    "fgspectra",
    "mflike",
    "numpy",
    "pixell",
    "pspipe",
    "pspipe_utils",
    "pspy",
    "sacc",
]
for m in modules:
    metadata[f"{m}_version"] = importlib.import_module(m).__version__
# Store dict file as strings
for k, v in d.items():
    metadata[k] = str(v)

# Common port2sacc arguments
common_kwargs = dict(
    binning_file=d["binning_file"],
    lmax=d["lmax"],
    cov_order=cov_order,
    passbands=passbands,
    beams=beams,
    metadata=metadata,
    log=log,
)

if sacc_fname in ["simu_sacc", "simu_w_syst_sacc"]:

    spec_dir = d["sim_spec_dir"]
    cov_name = "final_cov_sim"

    cov = np.load(f"{cov_dir}/x_ar_{cov_name}_gp.npy")

    if sacc_fname == "simu_w_syst_sacc":
        assert "_syst" in sim_spec_dir
        
        beam_cov = np.load(f"{cov_dir}/x_ar_beam_cov.npy")
        leakage_cov = np.load(f"{cov_dir}/x_ar_leakage_cov.npy")
        cov += beam_cov + leakage_cov
        cov_name = "final_cov_sim_with_syst"

    i_start = d["iStart"]
    i_stop = d["iStop"] + 1
    log.info(f"Storing simulations from {i_start} to {i_stop}...")
    for iii in range(i_start, i_stop):
        log.info(f"Storing simulation n°{iii}...")

        # Reading the flat data vector
        data_vec = covariance.read_x_ar_spectra_vec(
            spec_dir, spec_name_list, f"cross_{iii:05d}", spectra_order=spectra_order, type=d["type"]
        )

        # Let's store covariance and bbl in one extra file
        if iii == i_start:
            io.port2sacc(
                **common_kwargs,
                data_vec=data_vec,
                cov=cov,
                bbls=bbls,
                sacc_file_name=os.path.join(sacc_dir, f"{cov_name}_and_Bbl.fits"),
            )

        sacc_file_name = os.path.join(sacc_dir, f"{'_'.join(d['surveys'])}_{sacc_fname}_{iii:05d}.fits")
        io.port2sacc(
            **common_kwargs, data_vec=data_vec, cov=None, bbls=None, sacc_file_name=sacc_file_name
        )

else:

    if sacc_fname == "data_sacc":
        spec_dir = "spectra_leak_corr_ab_corr"
        cov_name = "final_cov_data"
        
    if sacc_fname == "ng_simu_sacc":
        spec_dir = "sim_spectra_ng"
        cov_name = "final_cov_sim"

    log.info(f"Getting '{cov_name}' covariance...")
    cov = np.load(os.path.join(cov_dir, f"x_ar_{cov_name}.npy"))

    # Reading the flat data vector
    data_vec = covariance.read_x_ar_spectra_vec(
        spec_dir, spec_name_list, "cross", spectra_order=spectra_order, type=d["type"]
    )

    sacc_file_name = os.path.join(sacc_dir, f"{'_'.join(d['surveys'])}_{sacc_fname}.fits")
    io.port2sacc(
        **common_kwargs, data_vec=data_vec, cov=cov, bbls=bbls, sacc_file_name=sacc_file_name
    )

log.info(f"Spectra name list: {spec_name_list} \n")
log.info(f"spectra_order '{spectra_order}' \n")
log.info(f"we have written the sacc file identified as {sacc_fname} \n ")
log.info(f"we used the'{cov_name}' covariance \n")
log.info(f"and the spectra located in '{spec_dir}'  \n")
