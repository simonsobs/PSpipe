description = """
This script performs array null tests and plot residual power spectra and a summary PTE distribution
"""
import numpy as np
import scipy.stats as ss
import pandas as pd
import json

from pspy import so_dict, pspy_utils, so_mpi
from pspipe_utils import log, pspipe_list, covariance

import sys
import os
import shutil
from os.path import join as opj
import itertools
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--include-parametric', action='store_true',
                    help='Params are absolute not nulled')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

surveys = d["surveys"]
binning_file = d["binning_file"]
lmax = d['lmax']
type = d['type']

bestfit_dir = d["best_fits_dir"]
spec_dir = d['spec_dir']
cov_dir = d['cov_dir']
null_test_dir = d["nulls_dir"]

_, sv_list, m_list = pspipe_list.get_arrays_list(d)
sv_m_list = ['_'.join(sv_m) for sv_m in zip(sv_list, m_list)]

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_", from_spec_nullgroups=d['spectra_list_from_spec_nullgroups'])
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

l_pows = [-1, 0, 1] # and sigma

max_lmin = 1500
min_lmax = 2000
# lmins = np.concatenate([[0], bin_mean[bin_mean <= max_lmin]]) 
lmins = bin_mean[np.logical_and(bin_mean <= max_lmin, bin_mean > 300)]
lmaxs = np.concatenate([bin_mean[bin_mean >= min_lmax], [lmax]])
lmin_lmax_keys = [f"{lmin}_{lmax}" for lmin, lmax in itertools.product(lmins, lmaxs)]

# get the slices into the x_ar
bin_out_dict, _ = covariance.get_indices(bin_low, bin_high, bin_mean, spec_name_list,
                                         spectra_order=spectra)

x_ar_slices_dict = {}
for (spec_name, pspipespec), (idxs, _) in bin_out_dict.items():
    na, nb = spec_name.split("x")
    if pspipespec in ["ET", "BT", "BE"]:
        mpair = 'x'.join([nb, na])
        spec = pspipespec[::-1]
    else:
        mpair = spec_name
        spec = pspipespec
    
    x_ar_slices_dict[spec, mpair] = slice(idxs[0], idxs[-1]+1)

spec2nullgroup2nullflag_mpairs = pspipe_list.get_spec2nullgroup2nullflag_mpairs(d, delimiter='_')

null_tests = []
for spec, nullgroup2nullflag_mpairs in spec2nullgroup2nullflag_mpairs.items():
    for nullgroup, (_, mpairs) in nullgroup2nullflag_mpairs.items():
        for mpair1, mpair2 in itertools.combinations(mpairs, r=2):
            null_tests.append(('simple_spec_analytic_cov', spec, nullgroup, mpair1, mpair2))
n_nulls = len(null_tests)

x_ar_data_vec = np.load(opj(spec_dir, "x_ar_data_vec.npy"))
x_ar_theory_vec = np.load(opj(bestfit_dir, "x_ar_theory_vec.npy"))
x_ar_res_vec = (x_ar_data_vec - x_ar_theory_vec)[:, None] # for slicing and matrix math
x_ar_cov = np.load(opj(cov_dir, "x_ar_analytic_cov.npy"))

if args.include_parametric:
    x_ar_parametricnull_data_vec = np.load(opj(null_test_dir, "x_ar_parametricnull_data_vec.npy"))[:, None] # for slicing and matrix math
    x_ar_parametricnull_cov = np.load(opj(null_test_dir, "x_ar_parametricnull_cov.npy"))

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_nulls - 1)
log.info(f"[Rank {so_mpi.rank}] number of cross-map pairs to compute: {len(subtasks)} / {n_nulls}")

null_database = []
null_ptes = {key: {'chi': [], 'chi2': []} for key in lmin_lmax_keys}
for task in subtasks:
    database_entry = {}

    scenario, spec, nullgroup, mpair1, mpair2 = null_tests[task]
    
    tested_mpairs = [mpair1, mpair2]
    
    tested_surveys = []
    tested_maps = []
    for mpair in tested_mpairs:
        na, nb = mpair.split('x')
        sva, _ = pspipe_list.get_sv_and_m_from_sv_m(na, sv_m_list, sv_list, m_list)
        svb, _ = pspipe_list.get_sv_and_m_from_sv_m(nb, sv_m_list, sv_list, m_list)
        if sva not in tested_surveys:
            tested_surveys.append(sva)
        if svb not in tested_surveys:
            tested_surveys.append(svb)
        if na not in tested_maps:
            tested_maps.append(na)
        if nb not in tested_maps:
            tested_maps.append(nb)

    database_entry = {
        'scenario': scenario,
        'spec': spec,
        'nullgroup': nullgroup,
        'leg1': mpair1,
        'leg2': mpair2,
        'tested_mpairs': tested_mpairs,
        'tested_surveys': tested_surveys,
        'tested_maps': tested_maps
    }
    
    r = 0
    cov = 0
    if args.include_parametric:
        r_param = 0
        cov_param = 0
    mpairs = (mpair1, mpair2)
    signs = (1, -1)
    for mpairij, signij in zip(mpairs, signs):
        sliceij = x_ar_slices_dict[spec, mpairij]
        r += signij * x_ar_res_vec[sliceij]
        if args.include_parametric:
            r_param += signij * x_ar_parametricnull_data_vec[sliceij]
        for mpairpq, signpq in zip(mpairs, signs):
            slicepq = x_ar_slices_dict[spec, mpairpq]
            cov += signij * signpq * x_ar_cov[sliceij, slicepq]
            if args.include_parametric:
                cov_param += signij * signpq * x_ar_parametricnull_cov[sliceij, slicepq]

    for _lmin, _lmax in itertools.product(lmins, lmaxs):
        mask = np.nonzero(np.logical_and(_lmin < bin_mean, bin_mean < _lmax))[0]
        _lmin_lmax_slice = slice(mask[0], mask[-1]+1)
        _r = r[_lmin_lmax_slice]
        _cov = cov[_lmin_lmax_slice, _lmin_lmax_slice]
        _l, _O = np.linalg.eigh(_cov)
        if not np.all(_l > 0):
            log.info(f"[Rank {so_mpi.rank}] {scenario} {spec} {nullgroup} {mpair1}-{mpair2} {_lmin=} {_lmax=} has non-positive eigenvalues")
        _reig = _O.T @ _r
        _chi = (_reig.T / _l**0.5).sum()
        _chi2 = ((_reig.T / _l) @ _reig).squeeze() # because the product is shape (1, 1) (2d scalar)
        _chi_pte = 2 * ss.norm.sf(abs(_chi), scale=len(_l)**0.5)
        _chi2_pte = ss.chi2.sf(_chi2, df=len(_l))

        key = f"{_lmin}_{_lmax}"
        null_ptes[key]['chi'].append(_chi_pte)
        null_ptes[key]['chi2'].append(_chi2_pte)

    null_database.append(database_entry)

    # now we need to save the spectra and err in the right place
    if args.include_parametric:
        null_data = np.array([bin_mean, r.reshape(-1), np.diag(cov)**0.5, r_param.reshape(-1), np.diag(cov_param)**0.5]).T # flatten r
        df = pd.DataFrame(null_data, columns=["lb", "null_value", "null_error", "parametric_null_value", "parametric_null_error"]).round(5)
    else:
        null_data = np.array([bin_mean, r.reshape(-1), np.diag(cov)**0.5]).T # flatten r
        df = pd.DataFrame(null_data, columns=["lb", "null_value", "null_error"]).round(5)

    subdir = opj(null_test_dir, f'interactive/{scenario}_{spec}_{nullgroup}/{mpair1}')
    pspy_utils.create_directory(subdir)
    df.to_json(opj(subdir, f'{mpair2}.json'), orient='values')

# save the database
list_of_null_database = so_mpi.comm.gather(null_database, root=0)
list_of_null_ptes = so_mpi.comm.gather(null_ptes, root=0)
if so_mpi.rank == 0:
    # flatten metadata over rank in rank order
    null_database = list(itertools.chain(*list_of_null_database))

    for i, row in enumerate(null_database):
        row['_orig_index'] = i

    # dump the database
    subdir = opj(null_test_dir, 'interactive')
    pspy_utils.create_directory(subdir)
    with open(opj(subdir, "index.json"), "w") as f:
        json.dump({"rows": null_database, "lmin_lmax_keys": lmin_lmax_keys}, f)

    # copy the index.html next to the database
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source = opj(base_dir, 'index.html')
    destination = opj(subdir, 'index.html')
    shutil.copy2(source, destination)

    # flatten ptes over rank in rank order
    subdir = opj(subdir, 'null_ptes')
    pspy_utils.create_directory(subdir)
    for key in lmin_lmax_keys:
        global_chi = []
        global_chi2 = []
        for rank_ptes in list_of_null_ptes:
            global_chi += rank_ptes[key]['chi']
            global_chi2 += rank_ptes[key]['chi2']

        with open(opj(subdir, f"null_ptes_{key}.json"), "w") as f:
            json.dump({"chi": global_chi, "chi2": global_chi2}, f)

    log.info(f"Exported metadata to index.json and {len(lmin_lmax_keys)} lazy-loadable PTE files.")