"""
This script compute the analytical covariance matrix elements
between split power spectra
"""
import sys
import numpy as np
from pspipe_utils import best_fits, log
from pspy import pspy_utils, so_cov, so_dict, so_map, so_mcm, so_mpi, so_spectra
from itertools import combinations_with_replacement as cwr
from itertools import combinations
from pspipe_utils import best_fits,  pspipe_list

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

windows_dir = "windows"
mcms_dir = "mcms"
coupling_dir = "coupling"
spectra_dir = "spectra"
noise_dir = "split_noise"
cov_dir = "split_covariances"
bestfit_dir = "best_fits"
sq_win_alms_dir = "sq_win_alms"

pspy_utils.create_directory(cov_dir)
surveys = d["surveys"]
binning_file = d["binning_file"]
lmax = d["lmax"]
niter = d["niter"]
binned_mcm = d["binned_mcm"]
apply_kspace_filter = d["apply_kspace_filter"]

arrays = {sv: d[f"arrays_{sv}"] for sv in surveys}
n_splits = {sv: d[f"n_splits_{sv}"] for sv in surveys}
array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]

spec_name_list = pspipe_list.get_spec_name_list(d)
sv_array_list = [f"{sv}_{ar}" for sv in surveys for ar in arrays[sv]]

spintypes = ("00", "02", "20", "++", "--")

# format:
# - unroll all "fields" i.e. (survey x array) is a "field"
# - any given combination is then ("field" x "field" x "spintype")
# - canonical spintypes are ("00", "02", "++", "--")

fields = []
for sv1 in surveys:
    for ar1 in arrays[sv1]:
        fields.append((sv1, ar1))

# ensure no "20" terms
def canonize_mcm(f1, f2, spintype):
    if spintype == "20":
        return f2, f1, "02"
    else:
        return f1, f2, spintype
    
def get_window(sv1, ar1, spin):
    polstr = "T" if spin == "0" else "pol"
    return so_map.read_map(d[f"window_{polstr}_{sv1}_{ar1}"])

for _f1, _f2 in cwr(fields, 2):  # find combinations with replacement of unrolled sv x arr
    for _spintype in spintypes:
        f1, f2, spintype = canonize_mcm(_f1, _f2, _spintype)  # flip "20" to "02"
        sv1, ar1 = f1
        sv2, ar2 = f2
        log.info(f"({sv1}_{ar1}) × ({sv2}_{ar2}) × ({spintype})")
        w1 = get_window(sv1, ar1, spintype[0])
        w2 = get_window(sv2, ar2, spintype[1])
        coupling = so_mcm.coupling_block(spintype, win1=w1, win2=w2, lmax=lmax, niter=niter)
        np.save(f"{coupling_dir}/{sv1}_{ar1}x{sv2}_{ar2}_{spintype}", coupling)

