"""
Run me with but supply a paramfile that matches the PSpipe params. However, the paramfile
must also have `pspipe_products_dir` (base directory with mcms) as well as 
`pspipe_operator_dir`, the location to produce outputs.

```
python python/get_pspipe_operator.py paramfiles/global_dr6_v4.dict
```
"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from pspy import so_map, so_dict, pspy_utils, so_mcm, so_spectra
from pspipe_utils import pspipe_list, kspace
import scipy.linalg

def get_binning_matrix(bin_lo, bin_hi, lmax, cltype = "Dl"):
    """Returns P_bl, the binning matrix that turns C_ell into C_b."""
    l = np.arange(2, lmax)
    if cltype == "Dl": fac = (l * (l + 1) / (2 * np.pi))
    elif cltype == "Cl": fac = l * 0 + 1
    n_bins = len(bin_lo)  # number of bins is same for all spectra in block
    Pbl = np.zeros( (n_bins, lmax-2) )
    for ibin in range(n_bins):
        loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))[0]
        Pbl[ibin,loc] = fac[loc] / len(loc)
    return Pbl

def read_all_Mbb(mcm_dir, na, nb, 
                 spin_pairs=("spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2")):
    """Convenience function to read inverse mode-coupling matrices computed by PSpipe."""
    mbb_inv, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{na}x{nb}", spin_pairs=spin_pairs)
    return mbb_inv

def get_PblMinv_matrix(binning_matrix, mbb_inv):
    """Computes P_{bl} Minv_{ll'}, the binning operator applied to the inverse MCM. Better
    to do this block-wise than materialize the full unbinned MCM across all polarizations.
    """
    Pbl = binning_matrix
    M00 = Pbl @ mbb_inv['spin0xspin0']
    M02 = Pbl @ mbb_inv['spin0xspin2']
    M20 = Pbl @ mbb_inv['spin2xspin0']
    Pbl_pol =  scipy.linalg.block_diag(Pbl, Pbl, Pbl, Pbl)
    M22 = Pbl_pol @ mbb_inv['spin2xspin2']
    PblMinv_bl = scipy.linalg.block_diag(M00, M02, M02, M20, M20, M22)
    return PblMinv_bl

def get_inverse_kspace_transfer_matrix(na, nb, param_dict, data_dir):
    d = param_dict  # convenience
    kspace_tf_path = d["kspace_tf_path"]
    surveys = d["surveys"]
    binning_file = d["binning_file"]
    spec_name = f"{na}x{nb}"
    lmax = d["lmax"]

    if kspace_tf_path == "analytical":
        arrays, templates, filter_dicts =  {}, {}, {}
        for sv in surveys:
            arrays[sv] = d[f"arrays_{sv}"]
            filter_dicts[sv] = d[f"k_filter_{sv}"]
            templates[sv] = so_map.read_map(str(data_dir / d[f"window_T_{sv}_{arrays[sv][0]}"]))
        kspace_transfer_matrix = kspace.build_analytic_kspace_filter_matrices(
            surveys, arrays, templates, filter_dicts, binning_file, lmax)[spec_name]
    else:
        kspace_transfer_matrix = np.load(f"{kspace_tf_path}/kspace_matrix_{spec_name}.npy", allow_pickle=True)

    inv_kspace_mat = np.linalg.inv(kspace_transfer_matrix)
    return inv_kspace_mat


def get_TE_corr_spectra(na, nb, param_dict):
    """Get the sim-based extra correction as the full stacked binned data vector."""
    spec_name = f"{na}x{nb}"
    kspace_tf_path = param_dict["kspace_tf_path"]
    spectra = ("TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB")
    _, TE_corr = so_spectra.read_ps(f"{kspace_tf_path}/TE_correction_{spec_name}.dat", spectra=spectra)
    Cb_TE_corr = np.hstack([TE_corr[s] for s in spectra])
    return Cb_TE_corr


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
spec_list = pspipe_list.get_spec_name_list(d, delimiter="_")  # unrolled fields

lmax = d["lmax"]
binning_file = d["binning_file"]
pspipe_products_dir = d["pspipe_products_dir"]
pspipe_operator_dir = d['pspipe_operator_dir']
bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)

Path(pspipe_operator_dir).mkdir(parents=True, exist_ok=True)


for spec1 in spec_list:
    na, nb = spec1.split("x")
    Pbl = get_binning_matrix(bin_lo, bin_hi, lmax)
    Mbb_inv = read_all_Mbb(pspipe_products_dir + "/mcms/", na, nb)
    Pbl_Minv = get_PblMinv_matrix(Pbl, Mbb_inv)
    inv_kspace_mat = get_inverse_kspace_transfer_matrix(na, nb, d, pspipe_products_dir)
    Finv_Pbl_Minv = inv_kspace_mat @ Pbl_Minv
    np.save(f'{pspipe_operator_dir}/Finv_Pbl_Minv_{na}x{nb}.npy', Finv_Pbl_Minv)
