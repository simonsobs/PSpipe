"""
This script simply takes existing calib and poleff measurement and 
apply these on spectra to make "cal" spectra
"""

import matplotlib
from pspy import pspy_utils, so_dict, so_spectra, so_cov
from pspipe_utils import consistency, log
import numpy as np
import pylab as plt
import itertools
import pickle
import sys
import yaml
import scipy.stats as ss
import argparse

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]


parser = argparse.ArgumentParser(description="calibrate the spectra using calibs and poleffs yaml files")

parser.add_argument("paramfile", help="Paramfile to use")

parser.add_argument("--calibs", type=str, help="calibs yaml file", default=None)
parser.add_argument("--calib-test", type=str, help="which calib test to use to calib: AxA-AxB, AxA-BxB or AxB-BxB", default='AxA-BxB')
parser.add_argument("--poleffs", type=str, help="poleffs yaml file", default=None)
parser.add_argument("--poleff-mode", type=str, help="which calib test to use to calib: AxA-AxB, AxA-BxB or AxB-BxB", default='EE')
parser.add_argument("--force-calib", type=bool, help="Force the usage of _calib spectra for poleff, even if no calibs is used", default=False)
parser.add_argument("--force-poleff", type=bool, help="Force the usage of _poleff spectra for calib, even if no poleff is used", default=False)

args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

spec_dir = d["spec_dir"]
survey_arrays = [
    f"{sv}_{ar}"
    for sv in d['surveys'] 
    for ar in d[f'arrays_{sv}']
]
survey_arrays_tuple = [
    (sv, ar) 
    for sv in d['surveys'] 
    for ar in d[f'arrays_{sv}']
]

calib_tests = ["AxA-AxB", "AxA-BxB", "BxB-AxB"]
poleff_modes = ["EE", "TE"]

if args.calibs is not None:
    calib_test = args.calib_test
    log.info(f'Calibrate spectra using calibs yaml file and {calib_test} test.')
    with open(args.calibs, "r") as file:
        calibs_dict: dict = yaml.safe_load(file)
    calibs_survey_arrays = list(calibs_dict['bestfits'].keys())
    
    poleff_suffix = '_poleff' if args.force_poleff else ''
    
    # Compare all sv_ar with the ones in calibs yaml file and put non existing ones to 1.
    for sv_ar in survey_arrays:
        if sv_ar not in calibs_survey_arrays:
            log.info(f'{sv_ar} not in calibs yaml, setting it to 1.')
            calibs_dict['bestfits'][sv_ar] = {test: 1. for test in calib_tests}
    
    # Calibrate the spectra and save with _cal suffix
    for (sv1, ar1), (sv2, ar2) in itertools.combinations_with_replacement(survey_arrays_tuple, r=2):
        sv_ar1 = f"{sv1}_{ar1}"
        sv_ar2 = f"{sv2}_{ar2}"
        # Only arrays of a same survey have _auto and _noise spectra
        spec_types = ['cross', 'auto', 'noise'] if sv1 == sv2 else ['cross']
        for spec_type in spec_types:
            # Load & calib
            spec_filename_load = f'{spec_dir}/Dl_{sv_ar1}x{sv_ar2}_{spec_type}{poleff_suffix}.dat'
            ls, Dls = so_spectra.read_ps(spec_filename_load, spectra=spectra)
            Dls_cal = {spec: Dls[spec] / calibs_dict['bestfits'][sv_ar1][calib_test] / calibs_dict['bestfits'][sv_ar2][calib_test] for spec in spectra}

            # Save with _cal suffix
            spec_filename_save = f'{spec_dir}/Dl_{sv_ar1}x{sv_ar2}_{spec_type}_calib{poleff_suffix}.dat'
            so_spectra.write_ps(spec_filename_save, ls, Dls_cal, type='Dl', spectra=spectra)


if args.poleffs is not None:
    poleff_mode = args.poleff_mode
    log.info(f'Calibrate spectra using poleffs yaml file and {poleff_mode} mode.')
    with open(args.poleffs, "r") as file:
        poleffs_dict: dict = yaml.safe_load(file)
    poleffs_survey_arrays = list(poleffs_dict['bestfits'].keys())
    
    cal_suffix = '_calib' if (args.calibs is not None) or args.force_calib else ''

    # Compare all sv_ar with the ones in poleffs yaml file and put non existing ones to 1.
    for sv_ar in survey_arrays:
        if sv_ar not in poleffs_survey_arrays:
            log.info(f'{sv_ar} not in poleffs yaml, setting it to 1.')
            poleffs_dict['bestfits'][sv_ar] = {mode: 1. for mode in poleff_modes}
    
    # Calibrate the spectra and save with _cal suffix
    for (sv1, ar1), (sv2, ar2) in itertools.combinations_with_replacement(survey_arrays_tuple, r=2):
        sv_ar1 = f"{sv1}_{ar1}"
        sv_ar2 = f"{sv2}_{ar2}"
        # Only arrays of a same survey have _auto and _noise spectra
        spec_types = ['cross', 'auto', 'noise'] if sv1 == sv2 else ['cross']
        for spec_type in spec_types:
            # Load & calib
            spec_filename_load = f'{spec_dir}/Dl_{sv_ar1}x{sv_ar2}_{spec_type}{cal_suffix}.dat'
            ls, Dls = so_spectra.read_ps(spec_filename_load, spectra=spectra)
            
            # TODO : make it for E and B and take each split's poleffs
            Dls_cal = {spec: Dls[spec] / poleffs_dict['bestfits'][sv_ar1][poleff_mode] / poleffs_dict['bestfits'][sv_ar2][poleff_mode] for spec in spectra}

            # Save with _cal suffix
            spec_filename_save = f'{spec_dir}/Dl_{sv_ar1}x{sv_ar2}_{spec_type}{cal_suffix}_poleff.dat'
            so_spectra.write_ps(spec_filename_save, ls, Dls_cal, type='Dl', spectra=spectra)