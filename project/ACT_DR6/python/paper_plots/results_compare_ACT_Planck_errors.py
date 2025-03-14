"""
This script plot the multifrequency error along with the cmb-only errors
"""

from pspy import so_dict, so_spectra, pspy_utils, so_cov
from pspipe_utils import  log, best_fits, external_data
import numpy as np
import pylab as plt
import sys, os
import pspipe_utils
import sacc


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

tag = d["best_fit_tag"]
binning_file = d["binning_file"]
lmax = d["lmax"]
labelsize = 24



combined_spec_dir = f"combined_spectra{tag}"

paper_plot_dir =  f"plots/paper_plot/"
pspy_utils.create_directory(paper_plot_dir)

type = d["type"]

planck_data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/planck")

########################################################################################
selected_spectra_list = [["TT"], ["EE"], ["TE", "ET"]]
########################################################################################

ylim = {}
ylim["TT"] = [4, 1000]
ylim["TE"] = [2, 40]
ylim["EE"] = [0.4, 30]

freq_pairs_ACT = ["90x90",  "150x150", "220x220", "90x220", "90x150", "150x220"]
freq_pairs_planck = ["100x100", "143x143", "217x217", "not_published", "not_published", "143x217"]
freq_pairs_ACT_name = ["90 GHz x 90 GHz",  "150 GHz x 150 GHz", "220 GHz x 220 GHz", "90 GHz x 220 GHz", "90 GHz x 150 GHz", "150 GHz x 220 GHz"]
freq_pairs_planck_name = ["f100xf100", "f143xf143", "f217xf217", "not_published", "not_published", "f143xf217"]


bin_low_act, bin_high_act, bin_mean_act, bin_size_act = pspy_utils.read_binning_file(binning_file, lmax)
bin_low_p, bin_high_p, bin_mean_p, bin_size_p = pspy_utils.read_binning_file(f"{planck_data_path}/../../binning_files/bin_planck.dat", lmax)

colors=["blue", "green", "brown", "darkcyan", "purple", "darkorange", "gray"]
show_planck_xfreq = True



plt.figure(figsize=(12,16))

for count, spec_select in enumerate(selected_spectra_list):
    plt.subplot(3, 1, count+1)
    plt.semilogy()
    for col, fp_A, name_A in zip(colors, freq_pairs_ACT, freq_pairs_ACT_name):

        s_name = spec_select[0]

        if (s_name != "TT") & ("220" in fp_A): continue
    
        lb_act, vec_act, sigma_act = np.loadtxt(f"{combined_spec_dir}/{type}_{fp_A}_{s_name}_cmb_only.dat", unpack=True)
        id_act = np.where((bin_mean_act >= lb_act[0]) & (bin_mean_act <= lb_act[-1]))
        assert (lb_act == bin_mean_act[id_act]).all()

        plt.plot(lb_act, sigma_act * np.sqrt(bin_size_act[id_act]), color=col, label = f"ACT {name_A}", linewidth=2)

        
    for col, fp_P, name_P in zip(colors, freq_pairs_planck, freq_pairs_planck_name):
        if fp_P == "not_published": continue

        lb_p, vec_p, sigma_p = np.loadtxt(f"{planck_data_path}/planck_spectrum_{s_name}_{fp_P}.dat", unpack=True)
        id_p = np.where((bin_mean_p >= lb_p[0]) & (bin_mean_p <= lb_p[-1]))
        assert (lb_p == bin_mean_p[id_p]).all()

        fac_p = lb_p * (lb_p + 1) / (2 * np.pi)
        vec_p, sigma_p = vec_p * fac_p, sigma_p * fac_p
        if show_planck_xfreq == True:
            plt.plot(lb_p, sigma_p * np.sqrt(bin_size_p[id_p]), color=col, linestyle=(0, (5,4)), alpha=1, label = f"Planck {name_P}", linewidth=2)
            
        
    if count < 2:
        plt.xticks([])
    if count == 0:
        plt.plot(np.zeros(1), np.zeros([1,2]), color='w', alpha=0, label=' ')

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.8), ncol=2, fontsize=22, frameon=False)

    plt.xlim(0,5000)
    if count == 2:
        plt.xlabel(r"$\ell$", fontsize=30)
    plt.ylabel(r"$\sigma^{%s}_{\ell_{b}} \sqrt{\Delta \ell_{b}}$" % s_name, fontsize=30)
    plt.ylim(ylim[s_name])

    plt.tick_params(labelsize=labelsize)

plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(f"{paper_plot_dir}/multifrequency_error_comparison{tag}.pdf", bbox_inches='tight')
plt.clf()
plt.close()



lb_act_co, sigma_act_co = {}, {}


sacc_cmb_only = d["sacc_cmb_only"]
# Read the cmb only sacc file
s = sacc.Sacc.load_fits(sacc_cmb_only)
lb_act_co["TT"], _, act_cov_co, _ = s.get_ell_cl("cl_00", "dr6_cmb_s0", "dr6_cmb_s0", return_cov = True, return_ind = True)
sigma_act_co["TT"] = np.sqrt(act_cov_co.diagonal())

lb_act_co["TE"], _, act_cov_co, _ = s.get_ell_cl("cl_0e", "dr6_cmb_s0", "dr6_cmb_s2", return_cov = True, return_ind = True)
sigma_act_co["TE"] = np.sqrt(act_cov_co.diagonal())

lb_act_co["EE"], _, act_cov_co, _ = s.get_ell_cl("cl_ee", "dr6_cmb_s2", "dr6_cmb_s2", return_cov = True, return_ind = True)
sigma_act_co["EE"] = np.sqrt(act_cov_co.diagonal())

l_planck, ps_planck_b, sigma_planck, cov_planck = external_data.get_planck_cmb_only_data()


spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
l_th, ps_th = so_spectra.read_ps("best_fits/cmb.dat", spectra=spectra)
ps_b_ACT = {}
ps_b_Planck = {}

for spec in spectra:
    lb, ps_b_ACT[spec] = pspy_utils.naive_binning(l_th, ps_th[spec], binning_file, lmax)
    lb, ps_b_Planck[spec] = pspy_utils.naive_binning(l_th, ps_th[spec], f"{planck_data_path}/../../binning_files/bin_planck.dat", lmax)


plt.figure(figsize=(12,8))
#plt.semilogy()
plt.title("CMB-only uncertainties")
colors = ["red", "teal", "blueviolet"]

for count, spec_select in enumerate(selected_spectra_list):
    s_name = spec_select[0]
    if s_name in ["TT", "TE"]: continue

    id_act = np.where((bin_mean_act >= lb_act_co[s_name][0]) & (bin_mean_act <= lb_act_co[s_name][-1]))
    
    np.testing.assert_allclose(lb_act_co[s_name], bin_mean_act[id_act])
    
    
    plt.plot(lb_act_co[s_name], np.abs(ps_b_ACT[s_name][id_act])  / (sigma_act_co[s_name] * np.sqrt(bin_size_act[id_act])), label = f"ACT {s_name}", linewidth=2, color=colors[count])
    

    my_l_planck, my_sigma_planck = l_planck[s_name].copy(), sigma_planck[s_name].copy()
    id_planck = np.where((bin_mean_p >= my_l_planck[0]) & (bin_mean_p <= my_l_planck[-1]))
    my_sigma_planck *= my_l_planck * (my_l_planck + 1) / (2 * np.pi)
    
    
    np.testing.assert_allclose(my_l_planck, bin_mean_p[id_planck])


    plt.plot(my_l_planck, np.abs(ps_b_Planck[s_name][id_planck]) / (my_sigma_planck * np.sqrt(bin_size_p[id_planck])), linestyle="--", label = f"Planck {s_name}", linewidth=2, color=colors[count])

plt.tick_params(labelsize=labelsize)

plt.ylabel(r"$\sigma^{\rm CMB-only}_{\ell_{b}} \sqrt{\Delta \ell_{b}}$" , fontsize=30)
plt.xlabel(r"$\ell$", fontsize=30)
plt.legend(fontsize=20)
plt.xlim(0,5000)
plt.savefig(f"{paper_plot_dir}/signal_to_noise{tag}.pdf", bbox_inches='tight')
plt.clf()
plt.close()

plt.figure(figsize=(12,8))
plt.semilogy()
plt.title("CMB-only uncertainties", fontsize=28)
#colors = ["red", "gray", "red"]
linestyle_list = [":", "-", "--"]

for count, spec_select in enumerate(selected_spectra_list):
    s_name = spec_select[0]
    if s_name in "TT": continue


    my_l_planck, my_sigma_planck = l_planck[s_name].copy(), sigma_planck[s_name].copy()
    id_planck = np.where((bin_mean_p >= my_l_planck[0]) & (bin_mean_p <= my_l_planck[-1]))
    my_sigma_planck *= my_l_planck * (my_l_planck + 1) / (2 * np.pi)
    
    np.testing.assert_allclose(my_l_planck, bin_mean_p[id_planck])

    plt.plot(my_l_planck, my_sigma_planck * np.sqrt(bin_size_p[id_planck]), linestyle=linestyle_list[count], label = f"{s_name}  Planck ", linewidth=2, color="darkblue")

    id_act = np.where((bin_mean_act >= lb_act_co[s_name][0]) & (bin_mean_act <= lb_act_co[s_name][-1]))
    
    np.testing.assert_allclose(lb_act_co[s_name], bin_mean_act[id_act])

    plt.plot(lb_act_co[s_name], sigma_act_co[s_name] * np.sqrt(bin_size_act[id_act]), linestyle=linestyle_list[count], label = f"{s_name}   ACT ", linewidth=2, color="red")
    

plt.tick_params(labelsize=labelsize)
plt.ylabel(r"$\sigma^{\rm CMB-only}_{\ell_{b}} \sqrt{\Delta \ell_{b}}$" , fontsize=30)
plt.xlabel(r"$\ell$", fontsize=30)
plt.legend(fontsize=26)
plt.xlim(0,5000)
plt.ylim(0.2, 80)
#plt.show()
plt.savefig(f"{paper_plot_dir}/cmb-only_error_comparison{tag}.pdf", bbox_inches='tight')
plt.clf()
plt.close()
