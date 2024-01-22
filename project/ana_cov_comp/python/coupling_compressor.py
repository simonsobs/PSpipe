"""
compresses the coupling files, for easy transport!

example: 
srun -n 64 -c 4 --cpu-bind=cores python python/coupling_compressor.py
"""

import sys
import numpy as np
from os import listdir
from os.path import isfile, join

from pspipe_utils import log, pspipe_list, misc
from pspy import pspy_utils, so_dict, so_map, so_mcm, so_mpi

coupling_dir = "/pscratch/sd/x/xzackli/so/data/dr6v4/couplings"
out_dir = "/pscratch/sd/x/xzackli/so/data/dr6v4/compressed_couplings"

coupling_files = [f for f in listdir(coupling_dir) if isfile(join(coupling_dir, f))]
n_coup = len(coupling_files)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_coup - 1)
for task in subtasks:
    fname = coupling_files[int(task)]
    if 'coupling' in fname:
        np.savez_compressed(join(out_dir, fname), 
            data=np.load(join(coupling_dir, fname)).astype(np.float32))
