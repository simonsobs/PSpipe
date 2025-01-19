"""
This script apply the likelihood calibration model to the data power spectra
"""

from pspy import so_dict, pspy_utils, so_spectra
from pspipe_utils import pspipe_list, log
import numpy as np
import sys, os


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

tag = d["best_fit_tag"]


spec_dir = "spectra_leak_corr_ab_corr"
cal_spec_dir = f"spectra_leak_corr_ab_corr_cal{tag}"
type = d["type"]
survey = "dr6"
arrays = d["arrays_dr6"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


pspy_utils.create_directory(cal_spec_dir)

cal_dipole = d[f"cal_like_dipole"]
cal, p_eff = {}, {}
for array in arrays:
    cal[f"{survey}_{array}"] = d[f"cal_like_{survey}_{array}"]
    p_eff[f"{survey}_{array}"] = d[f"pol_eff_like_{survey}_{array}"]


spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")
for spec_name in spec_name_list:
    print(spec_name)
    n1, n2 = spec_name.split("x")
    lb, ps = so_spectra.read_ps(f"{spec_dir}/{type}_{spec_name}_cross.dat", spectra=spectra)
    for spec in spectra:
        A, B = spec
        cal_ps = cal[n1] * cal[n2] * cal_dipole ** 2
        if A in ["E", "B"]: cal_ps *= p_eff[n1]
        if B in ["E", "B"]: cal_ps *= p_eff[n2]
        print(spec_name, spec, cal_ps)
        ps[spec] *= cal_ps
    so_spectra.write_ps(f"{cal_spec_dir}/{type}_{spec_name}_cross.dat", lb, ps, type, spectra=spectra)
