"""
This script reads the recipe from get_4pt_coupling_matrices_recipe.py and then cooks!

# For production on all, do something like
alias shortjob="sbatch paramfiles/1perlmutternode.slurm $1"  # short QOS
for i in {0..57}; do \
    shortjob "srun --ntasks 1 --cpus-per-task 128 --cpu-bind=cores python -u \
        python/get_4pt_coupling_result.py paramfiles/cov_dr6_v4_20231128.dict $((50*i)) $((50*i+50))"
done

# For testing only:
alias testjob="sbatch paramfiles/1perlmutterdebug.slurm $1"  # test QOS
testjob "srun --ntasks 1 --cpus-per-task 128 --cpu-bind=cores python -u \
    python/get_4pt_coupling_result.py paramfiles/cov_dr6_v4_20231128.dict 1 2"
"""

import sys
import numpy as np
from pspipe_utils import log, covariance as psc
from pspy import so_dict, so_map, so_mcm, pspy_utils
from itertools import product, combinations_with_replacement as cwr
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

ewin_alms_dir = d['ewin_alms_dir']
couplings_dir = d['couplings_dir']
pspy_utils.create_directory(couplings_dir)

surveys = d['surveys']
arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}

if d['use_toeplitz_cov'] == True:
    log.info('we will use the toeplitz approximation')
    l_exact, l_band, l_toep = 800, 2000, 2750
else:
    l_exact, l_band, l_toep = None, None, None

lmax = d['lmax']


recipe = np.load(f'{couplings_dir}/4pt_recipe.npy', allow_pickle=True)[()]

unrolled = recipe['S_only'] + recipe['SxN'] + recipe['N_only']
total_couplings = len(unrolled)

if len(sys.argv) == 4:
    log.info(f"computing only the covariance matrices : " + 
             f"{int(sys.argv[2])}:{int(sys.argv[3])} of {total_couplings}")
    unrolled = unrolled[int(sys.argv[2]):int(sys.argv[3])]

for k in unrolled:
    spintype, filename1, filename2, coupling_fn = k['spintype'], k['w1'], k['w2'], k['coupling_fn']
    w1 = np.load(filename1)
    w2 = np.load(filename2)
    if os.path.isfile(coupling_fn):
        log.info(f'{coupling_fn} exists, skipping')
    else:
        log.info(f"{spintype} {coupling_fn} from {filename1} {filename2}")
        coupling = so_mcm.coupling_block(spintype, win1=w1, win2=w2,
                                        lmax=lmax, input_alm=True,
                                        l_exact=l_exact, l_toep=l_toep,
                                        l_band=l_band)
        np.save(coupling_fn, coupling)  
