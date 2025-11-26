import numpy as np
import matplotlib.pyplot as plt
from pspipe_utils import consistency, covariance
from pspy import so_spectra, so_cov

def get_lmin_lmax(null, multipole_range):
    """
    compute the lmin and lmax associated to a given null test
    """
    m, ar1, ar2, ar3, ar4 = null
    m0, m1 = m[0], m[1]

    lmin0, lmax0 = multipole_range[ar1][m0]
    lmin1, lmax1 = multipole_range[ar2][m1]
    ps12_lmin = max(lmin0, lmin1)
    ps12_lmax = min(lmax0, lmax1)

    lmin2, lmax2 = multipole_range[ar3][m0]
    lmin3, lmax3 = multipole_range[ar4][m1]
    ps34_lmin = max(lmin2, lmin3)
    ps34_lmax = min(lmax2, lmax3)

    lmin = max(ps12_lmin, ps34_lmin)
    lmax = min(ps12_lmax, ps34_lmax)
    
    return lmin, lmax

def pte_histo(pte_list, file_name, n_bins):
    n_samples = len(pte_list)
    bins = np.linspace(0, 1, n_bins + 1)

    plt.figure(figsize=(8,6))
    plt.xlabel(r"Probability to exceed (PTE)")
    plt.hist(pte_list, bins=bins)
    plt.axhline(n_samples/n_bins, color="k", ls="--")
    plt.tight_layout()
    plt.savefig(f"{file_name}", dpi=300)
    plt.clf()
    plt.close()
    
    
def read_data(map_set_list, spec_dir, cov_dir, cov_type_list, spectra, iii=None):
    all_cov = {}
    if iii is None:
        ps_template = spec_dir + "/Dl_{}x{}_cross.dat"
    else:
        ps_template = spec_dir + "/Dl_{}x{}_cross_%05d.dat" % iii

    for cov in cov_type_list:
        cov_template = f"{cov_dir}/{cov}" + "_{}x{}_{}x{}.npy"
        all_ps, all_cov[cov] =  consistency.get_ps_and_cov_dict(map_set_list, ps_template, cov_template, spectra_order=spectra)
    lb = all_ps["ell"]
    return lb, all_ps, all_cov


def null_list_from_pairs(spectrum_list, map_set_pairs):
    null_list = []
    for spec in spectrum_list:
        for pa in map_set_pairs:
            a,b = pa
            null_list += [[spec, a, a, a, b]]
            null_list += [[spec, a, a, b, b]]
            null_list += [[spec, a, b, b, b]]
            
    return null_list

def get_plot_params():
    # Define the multipole range
    multipole_range = {
        "dr6_pa4_f150": {
            "T": [1250, 8500],
            "E": [500, 8500],
            "B": [500, 8500]
        },
        "dr6_pa4_f220": {
            "T": [1000, 8500],
            "E": [500, 8500],
            "B": [500, 8500]
        },
        "dr6_pa5_f090": {
            "T": [990, 8500],
            "E": [500, 8500],
            "B": [500, 8500]
        },
        "dr6_pa5_f150": {
            "T": [800, 8500],
            "E": [500, 8500],
            "B": [500, 8500]
        },
        "dr6_pa6_f090": {
            "T": [990, 8500],
            "E": [500, 8500],
            "B": [500, 8500]
        },
        "dr6_pa6_f150": {
            "T": [600, 8500],
            "E": [500, 8500],
            "B": [500, 8500],
        },
        "Planck_f100": {
            "T": [350, 1500],
            "E": [350, 1500],
            "B": [350, 1500]
        },
        "Planck_f143": {
            "T": [350, 2200],
            "E": [350, 2100],
            "B": [350, 2000],
        },
        "Planck_f217": {
            "T": [350, 2200],
            "E": [350, 2100],
            "B": [350, 2000],

        }
    }

    l_pows = {
        "TT": 1,
        "TE": 0,
        "TB": 0,
        "ET": 0,
        "BT": 0,
        "EE": -1,
        "EB": -1,
        "BE": -1,
        "BB": -1
    }
    
    y_lims = {
        "TT": (-100000, 75000),
        "TE": (-40, 40),
        "TB": (-40, 40),
        "ET": (-40, 40),
        "BT": (-40, 40),
        "EE": (-0.02, 0.02),
        "EB": (-0.02, 0.02),
        "BE": (-0.02, 0.02),
        "BB": (-0.02, 0.02)
    }
    return multipole_range, l_pows, y_lims


def read_ps_and_sigma(spec_dir, cov_dir, msa, msb, spectrum, cov_type_list):

    spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
    l, Db = so_spectra.read_ps(f"{spec_dir}/Dl_{msa}x{msb}_cross.dat", spectra=spectra)
    n_bins = len(l)
    cov = np.load(f"{cov_dir}/analytic_cov_{msa}x{msb}_{msa}x{msb}.npy")
    if "mc_cov" in cov_type_list:
        mc_cov = np.load(f"{cov_dir}/mc_cov_{msa}x{msb}_{msa}x{msb}.npy")
        cov = covariance.correct_analytical_cov(cov,
                                                mc_cov,
                                                only_diag_corrections=True)
    
    if "leakage_cov" in cov_type_list:
        leak_cov = np.load(f"{cov_dir}/leakage_cov_{msa}x{msb}_{msa}x{msb}.npy")
        cov += leak_cov
        
        
    sigma = so_cov.get_sigma(cov, spectra, n_bins, spectrum)
    return l, Db[spectrum], sigma
