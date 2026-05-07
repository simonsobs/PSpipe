from pathlib import Path
import sys
from pspy import so_dict, pspy_utils, so_cov, so_spectra, so_mpi
from pspipe_utils import pspipe_list
from tqdm import tqdm
import os
from pathlib import Path

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

plot_dir = d["plots_dir"] + '/nulls/'

for mode in spectra:
    pspy_utils.create_directory(plot_dir + f"/{mode}/")
    pspy_utils.create_directory(plot_dir + f"/{mode}/ALL")
    pspy_utils.create_directory(plot_dir + f"/{mode}/per_spec")

null_list = pspipe_list.get_null_list_from_cov_list(d, spectra=spectra, remove_TT_diff_freq=True)

spec_list = pspipe_list.get_spectra_list(d)


for mode, ms1, ms2, ms3, ms4 in tqdm(null_list):
    os.makedirs(f"{plot_dir}/{mode}/per_spec/{ms1}x{ms2}", exist_ok=True)
    os.makedirs(f"{plot_dir}/{mode}/per_spec/{ms3}x{ms4}", exist_ok=True)
    plot_fn = f"{plot_dir}/{mode}/ALL/diff_{mode}_{ms1}x{ms2}_{ms3}x{ms4}.png"

    os.symlink(
        plot_fn,
        f"{plot_dir}/{mode}/per_spec/{ms1}x{ms2}/diff_{mode}_{ms1}x{ms2}_{ms3}x{ms4}.png"
    )
    
    os.symlink(
        plot_fn,
        f"{plot_dir}/{mode}/per_spec/{ms3}x{ms4}/diff_{mode}_{ms1}x{ms2}_{ms3}x{ms4}.png"
    )

