"""
This script is used to reformat the ACT multifrequency
point source catalog `cat_skn_multifreq_20220526_nightonly`
in three different monofrequency catalogs (at 90,150 and 220 GHz)
"""
from pixell import utils
import pandas as pd
import numpy as np
from pspy import pspy_utils, so_dict
from pspipe_utils import log
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

cat_file = d["source_catalog"]
out_dir = "catalogs"

pspy_utils.create_directory(out_dir)

input_catalog = pd.read_table(cat_file, escapechar="#", sep="\s+")
input_catalog = input_catalog.shift(1, axis=1)

flux_id = {90: 1, 150: 2, 220: 3}

# https://phy-wiki.princeton.edu/polwiki/pmwiki.php?n=BeamAnalysis.BeamDistributionCenter
beam_areas = {
    90: [484.17, 487.03],
    150: [228.44, 215.24, 221.88], #nsr
    220: [107.34]
}

for freq, freq_id in flux_id.items():

    flux = {
        m: getattr(input_catalog, f"flux_{m}{freq_id}") for m in "TQU"
    }
    dflux = {
        m: getattr(input_catalog, f"dflux_{m}{freq_id}") for m in "TQU"
    }

    # nsr to sr
    beam_area = np.mean(beam_areas[freq])/1e9

    amp = {}
    damp = {}
    for m in ["T", "Q", "U"]:
        # utils.dplanck converts Jy to Kelvins
        # flux is given in mJy and we want amps in mK
        amp[m] = flux[m] / (utils.dplanck(freq*1e9) * beam_area)
        damp[m] = dflux[m] / (utils.dplanck(freq*1e9) * beam_area)

    ca = input_catalog.ca
    status = np.where(ca < 3, 1, 2)

    snr = amp["T"] / damp["T"]
    # npix is not used anywhere
    npix = np.zeros_like(snr)

    output_catalog = [input_catalog.ra, input_catalog.dec, snr, amp["T"], damp["T"], amp["Q"], damp["Q"], amp["U"], damp["U"],
                      flux["T"], dflux["T"], flux["Q"], dflux["Q"], flux["U"], dflux["U"],
                      npix, status]

    output_catalog = np.transpose(output_catalog)

    # ordering by snr
    output_catalog = output_catalog[output_catalog[:,2].argsort()[::-1]]

    header = "ra dec SNR Tamp dTamp Qamp dQamp Uamp dUamp Tflux dTflux Qflux dQflux Uflux dUflux npix status"
    fmt = "%11.6f %11.6f %8.3f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %5.0f %2.0f"
    np.savetxt(f"{out_dir}/cat_skn_{freq:03d}_20220526_nightonly_ordered.txt", output_catalog, header=header, fmt=fmt)
