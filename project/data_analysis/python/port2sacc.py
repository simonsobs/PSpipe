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
from pspipe_utils import covariance, io, log, misc, pspipe_list
from pspy import pspy_utils, so_dict

matplotlib.use("Agg")

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

# Either use simulation or real data
store_data = True

# Set covariance file name
cov_name = "analytic_cov"

mcm_dir = cvt.get_mcms_dir()
cov_dir = cvt.get_covariances_dir()
spec_dir = cvt.get_spectra_dir() if store_data else cvt.get_sim_spectra_dir()
sacc_dir = cvt.get_sacc_dir(create=True)

spec_name_list, nu_tag_list = pspipe_list.get_spec_name_list(d, delimiter="_", return_nu_tag=True)
log.info(f"Spectra name list: {spec_name_list}")

# Get spectra/cov orders
spectra_order = cvt.get_spectra_order()
cov_order = pspipe_list.x_ar_cov_order(spec_name_list, nu_tag_list, spectra_order=spectra_order)

# Reading covariance
log.info(f"Getting '{cov_name}' covariance...")
cov = np.load(os.path.join(cov_dir, f"x_ar_{cov_name}.npy"))

# Build list of survey x array
svxar = pspipe_list.get_map_set_list(d)

# Reading beams
beams = {
    f"{sv_ar}": misc.read_beams(d[f"beam_T_{sv_ar}"], d[f"beam_pol_{sv_ar}"]) for sv_ar in svxar
}
log.debug(f"Beam information: {beams}")

# Reading passbands : the passband file should be within the dict file
passbands = {}
for sv_ar in svxar:
    freq_info = d[f"freq_info_{sv_ar}"]
    if d["do_bandpass_integration"]:
        nu_ghz, passband = np.loadtxt(freq_info["passband"]).T
    else:
        nu_ghz, passband = np.array([freq_info["freq_tag"]]), np.array([1.0])

    passbands[f"{sv_ar}"] = [nu_ghz, passband]
log.debug(f"Passband information: {passbands}")

# Get Bbl : same bbl for every cross so load one and then assign it to every cross (to make the code
# as versatile as possible)
bbl = np.load(os.path.join(mcm_dir, f"{spec_name_list[0]}_Bbl_spin0xspin0.npy"))
bbls = {cross: bbl for cross in spec_name_list}

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

# Store data with sacc
if store_data:
    # Reading the flat data vector
    data_vec = covariance.read_x_ar_spectra_vec(
        spec_dir, spec_name_list, "cross", spectra_order=spectra_order, type=d["type"]
    )

    sacc_file_name = os.path.join(sacc_dir, f"{'_'.join(d['surveys'])}_data_sacc.fits")
    io.port2sacc(
        **common_kwargs, data_vec=data_vec, cov=cov, bbls=bbls, sacc_file_name=sacc_file_name
    )

else:
    chi2_list = []
    # Get theory vec for chi2 check
    theory_vec = covariance.read_x_ar_theory_vec(
        cvt.get_best_fits_dir(), mcm_dir, spec_name_list, d["lmax"], spectra_order=spectra_order
    )
    # Inverting covariance
    log.info(f"Inverting '{cov_name}' covariance...")
    inv_cov = np.linalg.inv(cov)

    i_start = d["iStart"]
    i_stop = d["iStop"] + 1
    log.info(f"Storing simulations from {i_start} to {i_stop}...")
    for i in range(i_start, i_stop):
        log.info(f"Storing simulation n°{i}...")

        # Reading the flat data vector
        data_vec = covariance.read_x_ar_spectra_vec(
            spec_dir, spec_name_list, f"cross_{i:05d}", spectra_order=spectra_order, type=d["type"]
        )

        # Let's store covariance and bbl in one extra file
        if i == i_start:
            io.port2sacc(
                **common_kwargs,
                data_vec=data_vec,
                cov=cov,
                bbls=bbls,
                sacc_file_name=os.path.join(sacc_dir, f"{cov_name}_and_Bbl.fits"),
            )

        sacc_file_name = os.path.join(sacc_dir, f"{'_'.join(d['surveys'])}_simu_sacc_{i:05d}.fits")
        io.port2sacc(
            **common_kwargs, data_vec=data_vec, cov=None, bbls=None, sacc_file_name=sacc_file_name
        )

        res = data_vec - theory_vec
        chi2_list.append(chi2 := res @ inv_cov @ res)
        log.info(f"Check X²/DoF = {chi2:.2f}/{len(data_vec)}")

    import scipy.stats as ss

    x = np.linspace(len(res) * 0.8, len(res) * 1.2, 300)
    kernel = ss.gaussian_kde(chi2_list)
    y_kde = kernel(x)
    y_pdf = ss.chi2(len(res)).pdf(x)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set(xlabel=r"$\chi^2$", yticks=[])
    ax.hist(chi2_list, density=True, bins=15, color="blue", alpha=0.4)
    ax.plot(x, y_kde, color="darkorange", label="KDE")
    ax.plot(x, y_pdf, color="forestgreen", label=r"$\chi^2$ dist.")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(cvt.get_plots_dir(), "chi2_sacc.png"), dpi=300)
