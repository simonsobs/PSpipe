#PRELIM

import os
import sys
from itertools import combinations_with_replacement as cwr
from itertools import product

import numpy as np
import sacc
from pspy import pspy_utils, so_dict, so_mcm

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

mcm_dir = "mcms"
like_product_dir = "sim_like_product"

lmax = d["lmax"]
iStart = d["iStart"]
iStop = d["iStop"]
binning_file = d["binning_file"]

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(bin_hi)

# let's get a list of all frequencies we plan to study
surveys = d["surveys"]
frequencies = []
for sv in surveys:
    arrays = d["arrays_%s" % sv]
    for ar in arrays:
        frequencies += [int(d["nu_eff_%s_%s" % (sv, ar)])]
frequencies = np.sort(list(dict.fromkeys(frequencies)))

for iii in range(iStart,iStop):
    # Saving into sacc format
    act_sacc = sacc.Sacc()
    for freq in frequencies:
        for spin, quantity in zip([0, 2], ["temperature", "polarization"]):
            # dummies file: not in used
            data_beams = {"l": np.arange(10000), "bl": np.ones(10000)}

            act_sacc.add_tracer(
                "NuMap",
                f"ACT_{freq}_s{spin}",
                quantity=f"cmb_{quantity}",
                spin=spin,
                nu=[freq],
                bandpass=[1.0],
                ell=data_beams.get("l"),
                beam=data_beams.get("bl"),
            )

    proj_data_vec = np.loadtxt("%s/data_vec_%05d.dat" % (like_product_dir, iii))

    count = 0
    for spec in ["TT", "TE", "EE"]:
        spec_frequencies = cwr(frequencies, 2) if spec != "TE" else product(frequencies, frequencies)
        for f1, f2 in spec_frequencies:
            print(f"Adding {f1}x{f2} GHz - {spec} spectra")
            # Set sacc tracer type and names
            pa, pb = spec
            ta_name = f"ACT_{f1}_s0" if pa == "T" else f"ACT_{f1}_s2"
            tb_name = f"ACT_{f2}_s0" if pb == "T" else f"ACT_{f2}_s2"

            map_types = {"T": "0", "E": "e", "B": "b"}
            if pb == "T":
                cl_type = "cl_" + map_types[pb] + map_types[pa]
            else:
                cl_type = "cl_" + map_types[pa] + map_types[pb]

            # Get Bbl
            mbb_inv, Bbl = so_mcm.read_coupling(
                os.path.join(mcm_dir, f"dr6_pa4_f150xdr6_pa4_f150"),
                spin_pairs=["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"],
            )
            Bbl_TT = Bbl["spin0xspin0"]
            Bbl_TE = Bbl["spin0xspin2"]
            Bbl_EE = Bbl["spin2xspin2"][: Bbl_TE.shape[0], : Bbl_TE.shape[1]]

            if spec in ["EE", "EB", "BE", "BB"]:
                Bbl = Bbl_EE
            elif spec in ["TE", "TB", "ET", "BT"]:
                Bbl = Bbl_TE
            else:
                Bbl = Bbl_TT
            ls_w = np.arange(2, Bbl.shape[-1] + 2)
            bp_window = sacc.BandpowerWindow(ls_w, Bbl.T)

            # Add ell/cl to sacc
            Db = proj_data_vec[count * n_bins : (count + 1) * n_bins]
            act_sacc.add_ell_cl(cl_type, ta_name, tb_name, lb, Db, window=bp_window)

            count += 1

    print("Adding covariance")
    proj_cov_mat = np.load("%s/combined_analytic_cov.npy" % like_product_dir)
    act_sacc.add_covariance(proj_cov_mat)
    print("Writing sacc file")
    act_sacc.save_fits(
        os.path.join(like_product_dir, f"act_simu_sacc_%05d.fits" % iii),
        overwrite=True,
    )
