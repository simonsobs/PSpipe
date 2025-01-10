#!/usr/bin/env python

import numpy as np
from pspipe_utils import external_data as ext
from pspy import pspy_utils

data_dir = "/global/cfs/cdirs/data/tlouis/dr6v4"

output_dir = f"{data_dir}/passbands"
pspy_utils.create_directory(output_dir)

arrays = ["pa4_f150", "pa4_f220", "pa5_f090", "pa5_f150", "pa6_f090", "pa6_f150"]

dr6_passbands = ext.get_passband_dict_dr6(arrays)

for array, [nu_ghz, passband] in dr6_passbands.items():
    np.savetxt(
        f"{output_dir}/passband_dr6_{array}.dat",
        np.array([nu_ghz, passband]).T,
        header="nu_ghz                   passband",
    )
