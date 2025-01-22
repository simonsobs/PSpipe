"""
plot the Planck and DR6 EE power and TE power spectra
"""
from pspy import so_dict, pspy_utils, so_spectra
from pspipe_utils import best_fits
import matplotlib.pyplot as plt
import numpy as np
import sys
import AxP_utils
import matplotlib


matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "40"

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

legacy_dir = "dr6xlegacy/"
npipe_dir = "dr6xnpipe/"
bestfit_dir = "best_fits"
spec_dir = "spectra_leak_corr_planck_bias_corr"
cov_dir = "covariances"

plot_dir = "paper_plot"
pspy_utils.create_directory(plot_dir)

cov_dir_legacy = f"{legacy_dir}/{cov_dir}"
cov_dir_npipe = f"{npipe_dir}/{cov_dir}"

spec_dir_legacy = f"{legacy_dir}/{spec_dir}"
spec_dir_npipe = f"{npipe_dir}/{spec_dir}"

bf_dir_legacy = f"{legacy_dir}/{bestfit_dir}"


cov_type_list = ["analytic_cov", "mc_cov", "leakage_cov"]

lth, psth = so_spectra.read_ps(f"{bf_dir_legacy}/cmb.dat", spectra=spectra)


my_ylim = {}
my_ylim["EE"] = (0,45)
my_ylim["TE"] = (-150,130)



fig = plt.figure(figsize=(40,40))
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
ax_list = [ax1, ax2]
count = 0
for ax, mode in zip(ax_list, ["EE", "TE"]):

    l, Db_legacy, sigma_legacy =  AxP_utils.read_ps_and_sigma(spec_dir_legacy, cov_dir_legacy, "Planck_f143", "Planck_f143", mode, cov_type_list)
    l, Db_npipe, sigma_npipe =  AxP_utils.read_ps_and_sigma(spec_dir_npipe, cov_dir_npipe, "Planck_f143", "Planck_f143", mode, cov_type_list)

    l, Db_pa6_f150, sigma_pa6_f150 =  AxP_utils.read_ps_and_sigma(spec_dir_legacy, cov_dir_legacy, "dr6_pa6_f150", "dr6_pa6_f150", mode, cov_type_list)
    l, Db_pa5_f150, sigma_pa5_f150 =  AxP_utils.read_ps_and_sigma(spec_dir_npipe, cov_dir_npipe, "dr6_pa5_f150", "dr6_pa5_f150", mode, cov_type_list)
    l, Db_pa6_f090, sigma_pa6_f090 =  AxP_utils.read_ps_and_sigma(spec_dir_legacy, cov_dir_legacy, "dr6_pa6_f090", "dr6_pa6_f090", mode, cov_type_list)
    l, Db_pa5_f090, sigma_pa5_f090 =  AxP_utils.read_ps_and_sigma(spec_dir_npipe, cov_dir_npipe, "dr6_pa5_f090", "dr6_pa5_f090", mode, cov_type_list)
    #plt.subplot(2,1,1 +count)
    ax.errorbar(l-15, Db_legacy, sigma_legacy, fmt="o", color="black", alpha=0.7, label="Planck* legacy 143 GHz")
    ax.errorbar(l-10, Db_pa6_f150, sigma_pa6_f150, fmt="o", color="purple", label="dr6 pa6-f150")
    ax.errorbar(l-5, Db_pa5_f150, sigma_pa5_f150, fmt="o", color="blue", label="dr6 pa5-f150")
    ax.errorbar(l, Db_pa6_f090, sigma_pa6_f090, fmt="o", color="darkorange", label="dr6 pa6-f090")
    ax.errorbar(l+5, Db_pa5_f090, sigma_pa5_f090, fmt="o", color="red", label="dr6 pa5-f090")
    ax.errorbar(l+10, Db_npipe, sigma_npipe, fmt="o", color="dimgrey", alpha=0.7, label="Planck* NPIPE 143 GHz")

    ax.plot(lth, psth[mode], color="gray")
    ax.set_ylim(my_ylim[mode])
    ax.set_xlim(0, 2000)
    if count == 0:
        ax.legend(fontsize=40)
        ax.set_xticks([])
        yticks = ax.yaxis.get_major_ticks()
        yticks[0].label1.set_visible(False)

    ax.set_ylabel(r"$D^{%s}_\ell$" % mode, fontsize=50)
    ax.axvline(500, -150,150, color="gray", linestyle=(0, (5,10)))
    
    if count == 1:
        ax.set_xlabel(r"$\ell$", fontsize=50)
        yticks = ax.yaxis.get_major_ticks()
        yticks[-1].label1.set_visible(False)

    count+=1
plt.subplots_adjust(wspace=0, hspace=0)

plt.savefig(f"{plot_dir}/ACT_and_planck.pdf",bbox_inches='tight')
plt.clf()
plt.close()







