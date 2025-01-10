"""
Read the Planck NPIPE beam files (per split beams)
and save the per-split beam as tables in .dat files.
Main usage: mean beams are required to run the power
spectrum pipeline with Planck maps. Per-split beams
are required to run dory source subtraction
"""
import astropy.io.fits as pyfits
import numpy as np
from pspy import pspy_utils

npipe_beam_dir = "/global/cfs/cdirs/cmb/data/planck2020/npipe/npipe6v20/quickpol"

freqs = [100, 143, 217, 353]
splits = ["A", "B"]

output_dir = "npipe6v20_beams"
pspy_utils.create_directory(output_dir)
for freq in freqs:
    bl_mean = 0.
    for split in splits:

        hdul = pyfits.open(f"{npipe_beam_dir}/Bl_npipe6v20_{freq}{split}x{freq}{split}.fits")

        N = len(hdul[1].data)

        bl = np.array([hdul[1].data[i][0] for i in range(N)])
        ell = np.arange(N)

        bl_mean += bl
        np.savetxt(f"{output_dir}/npipe6v20_beam_{freq}{split}.dat", np.transpose([ell,bl]))

        hdul.close()

    bl_mean /= len(splits)
    np.savetxt(f"{output_dir}/npipe6v20_beam_{freq}_mean.dat", np.transpose([ell, bl_mean]))
