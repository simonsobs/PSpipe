"""
Plot the TT power spectra with their respective foreground components
"""
import matplotlib
import sys

import numpy as np
import matplotlib.pyplot as plt
from pspipe_utils import best_fits, log, pspipe_list, covariance
from pspy import pspy_utils, so_dict, so_spectra
from matplotlib import rcParams


labelsize = 14
fontsize = 20


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)


tag = d["best_fit_tag"]
fg_components = d["fg_components"]
binning_file = d["binning_file"]
lmax = d["lmax"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

bestfit_dir = f"best_fits{tag}"
components_dir = f"{bestfit_dir}/components"
cov_dir = "covariances"
spec_dir = f"spectra_leak_corr_ab_corr_cal{tag}"

paper_plot_dir = f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter = "_")

## Read the data and extract TT
cov_xar = np.load(f"{cov_dir}/x_ar_final_cov_data.npy")
vec_xar = covariance.read_x_ar_spectra_vec(spec_dir,
                                           spec_name_list,
                                           "cross",
                                           spectra_order = spectra,
                                           type="Dl")

spectra_cuts = {"dr6_pa4_f220": dict(T=[975, lmax], P=[lmax, lmax]),
                "dr6_pa5_f150": dict(T=[775, lmax], P=[775, lmax]),
                "dr6_pa6_f150": dict(T=[575, lmax], P=[575, lmax]),
                "dr6_pa5_f090": dict(T=[975, lmax], P=[975, lmax]),
                "dr6_pa6_f090": dict(T=[975, lmax], P=[975, lmax])}

bin_out_dict, indices = covariance.get_indices(bin_lo,
                                               bin_hi,
                                               lb,
                                               spec_name_list,
                                               spectra_cuts=spectra_cuts,
                                               spectra_order=spectra,
                                               selected_spectra=["TT"])

cov_TT = cov_xar[np.ix_(indices, indices)]
vec_TT = vec_xar[indices]

lb_dict, ps_TT, sigma_TT = {}, {}, {}
for spec_select in bin_out_dict.keys():
    my_spec = spec_select[0]
    my_id, lb_dict[my_spec] = bin_out_dict[spec_select]
    ps_TT[my_spec] = vec_TT[my_id]
    sigma_TT[my_spec] = np.sqrt(cov_TT[np.ix_(my_id, my_id)].diagonal())


#### remove pa6 for clarity
d["arrays_dr6"] = ["pa4_f220", "pa5_f090", "pa5_f150"]
#spectra_list = pspipe_list.get_spec_name_list(d, delimiter = "_")

narrays, _, _ = pspipe_list.get_arrays_list(d)

l_th, ps_dict = so_spectra.read_ps(f"{bestfit_dir}/cmb.dat", spectra=spectra)

# explicitly list components in the order we want them in
fg_components = ['cibp', 'cibc', 'tSZxCIB', 'tSZ', 'kSZ', 'radio', 'dust']
comp_color = ["darkorange", "darkmagenta", "magenta", "brown", "green", "red", "blue"]

spectra_list =  ["dr6_pa5_f090xdr6_pa5_f090", "dr6_pa5_f090xdr6_pa5_f150",  "dr6_pa4_f220xdr6_pa5_f090", "dr6_pa5_f150xdr6_pa5_f150",  "dr6_pa4_f220xdr6_pa5_f150", "dr6_pa4_f220xdr6_pa4_f220"]
fig, axes = plt.subplots(narrays, narrays, sharex=True, sharey=True, figsize=(12, 9), dpi=100)
axes = np.atleast_2d(axes)
indices = np.triu_indices(narrays)[::-1]
for i, cross in enumerate(spectra_list):
    name1, name2 = cross.split("x")
    idx = (indices[0][i], indices[1][i])
    ax = axes[idx]

    if i == 0:
        ax.errorbar(lb_dict[cross], ps_TT[cross], sigma_TT[cross], fmt=".", label="data", markersize=6)
    else:
        ax.errorbar(lb_dict[cross], ps_TT[cross], sigma_TT[cross], fmt=".", markersize=6)
        
    l_th, fg_all = so_spectra.read_ps(f"{bestfit_dir}/fg_{cross}.dat", spectra=spectra)
        
    if i==0:
        ax.plot(l_th, ps_dict["TT"], color="gray", linestyle="--", label="CMB", linewidth=1)
        ax.plot(l_th, ps_dict["TT"] + fg_all["TT"], color="gray", label="CMB + fg", linewidth=1)
    else:
        ax.plot(l_th, ps_dict["TT"], color="gray", linestyle="--", linewidth=1)
        ax.plot(l_th, ps_dict["TT"] + fg_all["TT"], color="gray", linewidth=1)

    for comp, col in zip(fg_components, comp_color):
        l_th, fg_comp = np.loadtxt(f"{components_dir}/tt_{comp}_{cross}.dat", unpack=True)
        if comp == "tSZxCIB":
            fg_comp = np.abs(fg_comp)

        if i==0:
            if comp == "cibp":
                label = "CIB-Poisson"
            elif comp == "cibc":
                label = "CIB-Clustered"
            elif comp == "tSZxCIB":
                label = "|tSZxCIB|"
            elif comp == "dust":
                label = "Galactic dust"
            else:
                label=comp
            
            ax.plot(l_th, fg_comp, label=label, linewidth=1, color=col)
        else:
            ax.plot(l_th, fg_comp, linewidth=1, color=col)

    title_ax = cross.replace("dr6_", "")
    title_ax = title_ax.replace("_", " ")
    title_ax = title_ax.replace("pa", "PA")

    if "PA4 f220" in title_ax:
        a, b = title_ax.split("x")
        title_ax = f"{b}x{a}"

    ax.set_title(title_ax, fontsize=labelsize)
    ax.set_yscale("log")
    ax.set_ylim(1, 1e4)
    ax.set_xlim(200, 7800)
    ax.tick_params(labelsize=labelsize)

for idx in zip(*np.triu_indices(narrays, k=1)):
    ax = axes[idx]
    fig.delaxes(ax)

for i in range(narrays):
    axes[-1, i].set_xlabel(r"$\ell$", fontsize=fontsize)
    axes[i, 0].set_ylabel(r"$D_\ell \ [\mu K^{2}]$", fontsize=fontsize)
    
fig.legend(bbox_to_anchor=(0.94,1), fontsize=labelsize)
plt.tight_layout()
plt.savefig(f"{paper_plot_dir}/TT_per_components{tag}.pdf")
plt.clf()
plt.close()

