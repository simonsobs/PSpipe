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


spec_dir = "spectra_leak_corr_ab_corr"
cal_spec_dir = "spectra_leak_corr_ab_corr_cal"
type = d["type"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


pspy_utils.create_directory(cal_spec_dir)
cal = {}
cal["dr6_pa4_f220"] = 0.9879
cal["dr6_pa5_f090"] = 1.0011
cal["dr6_pa5_f150"] = 0.9977
cal["dr6_pa6_f090"] = 1.0005
cal["dr6_pa6_f150"] = 1.0006

p_eff = {}
p_eff["dr6_pa4_f220"] = 1.
p_eff["dr6_pa5_f090"] = 0.9818
p_eff["dr6_pa5_f150"] = 0.9954
p_eff["dr6_pa6_f090"] = 0.9924
p_eff["dr6_pa6_f150"] = 0.9926


spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")
for spec_name in spec_name_list:
    print(spec_name)
    n1, n2 = spec_name.split("x")
    lb, ps = so_spectra.read_ps(f"{spec_dir}/{type}_{spec_name}_cross.dat", spectra=spectra)
    for spec in spectra:
        A, B = spec
        cal_ps = cal[n1] * cal[n2]
        if A in ["E", "B"]: cal_ps *= p_eff[n1]
        if B in ["E", "B"]: cal_ps *= p_eff[n2]
        print(spec_name, spec, cal_ps)
        ps[spec] *= cal_ps
    so_spectra.write_ps(f"{cal_spec_dir}/{type}_{spec_name}_cross.dat", lb, ps, type, spectra=spectra)
