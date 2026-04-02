from pspy import so_map, so_spectra, pspy_utils, so_mcm, so_dict
import numpy as np
import sys
import candl
import spt_candl_data
from matplotlib import pyplot as plt

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
l, Dl = so_spectra.read_ps("/pscratch/sd/m/merrydup/LAT_ISO/spectra/cmb.dat", spectra=spectra)
LMAX = d["lmax"]
_, _, ell, _ =  pspy_utils.read_binning_file(d["binning_file"], lmax=LMAX)


mcms_dir = "mcms/"

# Load SPT window functions from candl data
candl_like = candl.Like(spt_candl_data.SPT3G_D1_TnE)
window_function_dict = {spec_to_load: candl_like.window_functions[spec_id] for spec_id, spec_to_load in enumerate(candl_like.spec_order)}

# SPT 90GHz is named 95 in Thibaut's paramfile :(
freq_mapping = {
    "90": "95",
    "150": "150",
    "220": "220",
}

# Load Bbl from pipeline outputs (mcms dir) and compute binned theory spectra
Bbl_dict = {}
Dl_wf = {}     # spectra binned with SPT window function
Dl_bbl = {}    # spectra binned with Bbl
for freqs in [["90", "90"], ["90", "150"], ["90", "220"], ["150", "150"], ["150", "220"], ["220", "220"]]:
    freqs_str = 'x'.join(freqs)
    
    mbb_inv, Bbl = so_mcm.read_coupling(
        prefix=f"{mcms_dir}/spt_{int(freq_mapping[freqs[0]]):03d}xspt_{int(freq_mapping[freqs[1]]):03d}",
        spin_pairs=spin_pairs
    )
    
    Bbl_dict[f"TT {freqs_str}"] = Bbl["spin0xspin0"]
    Bbl_dict[f"TE {freqs_str}"] = Bbl["spin0xspin2"]

    # Load EE spin2xspin2
    nbins, my_lmax = int(Bbl["spin2xspin2"].shape[0]/4), int(Bbl["spin2xspin2"].shape[1]/4)
    Bbl_dict[f"EE {freqs_str}"] = Bbl["spin2xspin2"][:nbins, :my_lmax]

    for spec in ["TT", "TE", "EE"]:
        Dl_wf[f"{spec} {freqs_str}"] = np.dot(window_function_dict[f"{spec} {freqs_str}"].T, Dl[spec][:window_function_dict[f"{spec} {freqs_str}"].shape[0]])
        Dl_bbl[f"{spec} {freqs_str}"] = np.dot(Bbl_dict[f"{spec} {freqs_str}"], Dl[spec][:LMAX])

# SPT has different lmin and lmax depending on spectra and idk how to get these from candl_data so I had to figure out these
lmin_id = {
    "TT":8,
    "TE":8,
    "EE":8,
}
lmax_id = {
    "TT":60,
    "TE":71,
    "EE":71,
}

for freqs in [["90", "90"], ["90", "150"], ["90", "220"], ["150", "150"], ["150", "220"], ["220", "220"]]:
    freqs_str = 'x'.join(freqs)
    for spec in ["TT", "TE", "EE"]:
        fig, ax = plt.subplots(2, figsize=(8, 6))
        ax[0].plot(ell[lmin_id[spec]:lmax_id[spec]], Dl_wf[f"{spec} {freqs_str}"][:61], label="win func SPT * theory")
        ax[0].plot(ell[lmin_id[spec]:lmax_id[spec]], Dl_bbl[f"{spec} {freqs_str}"][lmin_id[spec]:lmax_id[spec]], label="bbl * theory")
        ax[0].legend()
        ax[0].set_ylabel(fr"$D_\ell^{{{spec}}}$", fontsize=15)

        Dls_res = Dl_wf[f"{spec} {freqs_str}"][:61] - Dl_bbl[f"{spec} {freqs_str}"][lmin_id[spec]:lmax_id[spec]]
        
        ax[1].plot(ell[lmin_id[spec]:lmax_id[spec]], Dls_res, label="win func SPT * theory - bbl * theory")
        ax[1].legend()
        ax[1].set_ylabel(fr"$\Delta D_\ell^{{{spec}}}$", fontsize=15)
        
        np.savetxt(f"{mcms_dir}/res_bbl_{spec}_{freqs_str}.txt", np.array([ell[lmin_id[spec]:lmax_id[spec]], Dls_res]).T)
        plt.savefig(f"plots/{spec}_{freqs_str}_compare_binning")