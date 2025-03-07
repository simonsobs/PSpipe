"""
Just a script to check that the leakage correction we apply make sense and to see the impact on the spectra
Additionaly look at the effect of the aberration correction.
"""
from pspy import so_spectra, so_dict, so_cov, pspy_utils
from pspipe_utils import pspipe_list
import pylab as plt
import numpy as np
import sys, os


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

plot_dir = f"plots/TE_TB_correction/"
pspy_utils.create_directory(plot_dir)

for my_spec in ["TE", "TB"]:

    my_spec_r = my_spec[::-1]
    for spec_name in spec_name_list:
        if "pa4" in spec_name: continue
        lth, ps_th = so_spectra.read_ps(f"best_fits/cmb_and_fg_{spec_name}.dat", spectra=spectra)


        na, nb  = spec_name.split("x")
        na, nb =  na.replace("dr6_", ""),  nb.replace("dr6_", "")
        if my_spec == "TE":
            gamma_TX = np.loadtxt(f"nominal/{nb}_gamma_t2e.txt")
            gamma_XT = np.loadtxt(f"nominal/{na}_gamma_t2e.txt")
        if my_spec == "TB":
            gamma_TX = np.loadtxt(f"nominal/{nb}_gamma_t2b.txt")
            gamma_XT = np.loadtxt(f"nominal/{na}_gamma_t2b.txt")

        l_, g_TX = gamma_TX[:,0], gamma_TX[:,1]
        l_, g_XT = gamma_XT[:,0], gamma_XT[:,1]
        
        leak_TX = ps_th["TT"] * g_TX[:len(ps_th["TT"])]
        leak_XT = ps_th["TT"] * g_XT[:len(ps_th["TT"])]

        ab_corr =  np.loadtxt(f"ab_corr/aberration_correction_{spec_name}.dat")
        if my_spec == "TE":
            ab_corr_TX = ab_corr[:,2]
            ab_corr_XT = ab_corr[:,4]
        if my_spec == "TB":
            ab_corr_TX = ab_corr[:,3]
            ab_corr_XT = ab_corr[:,5]

        l, ps = so_spectra.read_ps(f"spectra/Dl_{spec_name}_cross.dat", spectra=spectra)
        l, ps_leak_corr = so_spectra.read_ps(f"spectra_leak_corr/Dl_{spec_name}_cross.dat", spectra=spectra)
        l, ps_leak_ab_corr = so_spectra.read_ps(f"spectra_leak_corr_ab_corr/Dl_{spec_name}_cross.dat", spectra=spectra)
    
        ps_TX_list = []
        ps_XT_list = []

        for iii in range(0, 999):
            l, ps_sim_leak = so_spectra.read_ps(f"montecarlo_beam_leakage/Dl_{spec_name}_cross_{iii:05d}.dat", spectra=spectra)
            ps_TX_list += [ps_sim_leak[my_spec]]
            ps_XT_list += [ps_sim_leak[my_spec_r]]

        sim_TX_mean = np.mean(ps_TX_list, axis=0)
        sim_TX_std = np.std(ps_TX_list, axis=0)
        sim_XT_mean = np.mean(ps_XT_list, axis=0)
        sim_XT_std = np.std(ps_XT_list, axis=0)

    
        cov_mc = np.load(f"covariances/mc_cov_{spec_name}_{spec_name}.npy")
        cov_leakage = np.load(f"covariances/leakage_cov_{spec_name}_{spec_name}.npy")
        cov = cov_mc + cov_leakage
        nbins = len(l)
    
        sub_cov_TX = so_cov.selectblock(cov, spectra, nbins, block=f"{my_spec}{my_spec}")
        sub_cov_XT = so_cov.selectblock(cov, spectra, nbins, block=f"{my_spec_r}{my_spec_r}")
    
        sigma_TX = np.sqrt(sub_cov_TX.diagonal())
        sigma_XT = np.sqrt(sub_cov_XT.diagonal())

        f, (a0, a1) = plt.subplots(2, 1, height_ratios=[2, 1], figsize=(15, 10))
        a0.plot(lth, ps_th[my_spec]*0.1, color="gray", linestyle="--", label = r"$10\%  D_{\ell}$" )
        a0.plot(lth, leak_TX, label="best fit leakage", color="green")
        a0.errorbar(l, ps[my_spec] - ps_leak_corr[my_spec], sim_TX_std, fmt= "o", label="leakage correction", color="green")
        a0.plot(l, ps_leak_corr[my_spec] - ps_leak_ab_corr[my_spec], "o", label="aberration correction", color="orange")
        a0.plot(l, ab_corr_TX, "--", label="aberration model", color="orange")
        a0.legend(fontsize=18)
        a0.tick_params(labelsize=18)
        a0.set_title(f"{na} x {nb}", fontsize=18)
        a0.set_xlim(0, 4000)
        a0.set_ylabel(r"$D^{%s}_{\ell}$" % my_spec, fontsize=30)
        a0.set_xlabel(r"$\ell$", fontsize=30)
        a1.set_ylabel(r"$(\Delta D_{\ell})/ \sigma^{\rm tot}_{\ell}$", fontsize=30)
        a1.set_xlabel(r"$\ell$", fontsize=30)
        a1.set_xlim(0, 4000)
        a1.set_ylim(-2,2)
        a1.hlines(1, 0, 10000, color="gray", alpha=0.4, linestyle=":")
        a1.hlines(-1, 0, 10000, color="gray", alpha=0.4, linestyle=":")
        a1.hlines(0, 0, 10000, color="gray", alpha=0.4, linestyle=":")
        a1.tick_params(labelsize=18)
        a1.plot(l, (ps[my_spec] - ps_leak_ab_corr[my_spec])/sigma_TX)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/corr_{my_spec}_{na}x{nb}.png")
        plt.clf()
        plt.close()

        if na != nb:
            f, (a0, a1) = plt.subplots(2, 1, height_ratios=[2, 1], figsize=(15, 10))
            a0.plot(lth, ps_th[my_spec_r]*0.1, color="gray", linestyle="--", label = r"$10\%  D_{\ell}$" )
            a0.plot(lth, leak_XT, label="best fit leakage", color="green")
            a0.errorbar(l, ps[my_spec_r] - ps_leak_corr[my_spec_r], sim_XT_std, fmt= "o", label="leakage correction", color="green")
            a0.plot(l, ps_leak_corr[my_spec_r] - ps_leak_ab_corr[my_spec_r], "o", label="aberration correction", color="orange")
            a0.plot(l, ab_corr_XT, "--", label="aberration model", color="orange")
            a0.legend(fontsize=18)
            a0.tick_params(labelsize=18)
            a0.set_title(f"{na} x {nb}", fontsize=18)
            a0.set_xlim(0, 4000)
            a0.set_ylabel(r"$D^{%s}_{\ell}$" % my_spec_r, fontsize=30)
            a0.set_xlabel(r"$\ell$", fontsize=30)
            a1.set_ylabel(r"$(\Delta D_{\ell})/ \sigma^{\rm tot}_{\ell}$", fontsize=30)
            a1.set_xlabel(r"$\ell$", fontsize=30)
            a1.set_xlim(0, 4000)
            a1.set_ylim(-2,2)
            a1.hlines(1, 0, 10000, color="gray", alpha=0.4, linestyle=":")
            a1.hlines(-1, 0, 10000, color="gray", alpha=0.4, linestyle=":")
            a1.hlines(0, 0, 10000, color="gray", alpha=0.4, linestyle=":")
            a1.tick_params(labelsize=18)
            a1.plot(l, (ps[my_spec_r] - ps_leak_ab_corr[my_spec_r])/sigma_XT)
            plt.tight_layout()
            plt.savefig(f"{plot_dir}/corr_{my_spec}_{nb}x{na}.png")
            plt.clf()
            plt.close()
