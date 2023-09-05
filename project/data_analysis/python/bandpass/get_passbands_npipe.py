#!/usr/bin/env python

import numpy as np
from pspipe_utils import external_data as ext
from pspy import pspy_utils

data_dir = "/global/cfs/cdirs/act/data/tlouis/dr6v4"

output_dir = f"{data_dir}/passbands"
pspy_utils.create_directory(output_dir)

npipe_wafers = [f"npipe_f{freq}" for freq in [100, 143, 217, 353, 545, 857]]
npipe_freq_range = [(50, 1100) for wafer in npipe_wafers]

npipe_passbands = ext.get_passband_dict_npipe(npipe_wafers, freq_range_list=npipe_freq_range)

for wafer, [nu_ghz, passband] in npipe_passbands.items():
    np.savetxt(
        f"{output_dir}/passband_{wafer}.dat",
        np.array([nu_ghz, passband]).T,
        header="nu_ghz                   passband",
    )
