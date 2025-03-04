"""
This script plot the spectra of the release
"""

from pspy import so_dict, pspy_utils, so_cov
import numpy as np
import pylab as plt
import sys, os
import pickle
import matplotlib.gridspec as gridspec


labelsize = 14
fontsize = 20

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

tag = d["best_fit_tag"]
spec_dir = f"spectra_leak_corr_ab_corr_cal{tag}"
release_spec_dir = f"release_spectra{tag}"
plot_dir = f"plots/release_spectra{tag}/"

pspy_utils.create_directory(f"{plot_dir}/x_array_bands/")
pspy_utils.create_directory(f"{plot_dir}/x_freqs/")
pspy_utils.create_directory(f"{plot_dir}/combined/")

file = open(f"{release_spec_dir}/dataset_trace.pkl", 'rb')
dataset_trace = pickle.load(file)

ylim= {}
ylim["TT"] = [10, 4 * 10 ** 3]
ylim["TE"] = [-140, 70]
ylim["TB"] = [-10,10]
ylim["ET"] = [-140, 70]
ylim["BT"] = [-10,10]
ylim["EE"] = [-10,50]
ylim["EB"] = [-3,3]
ylim["BE"] = [-3,3]
ylim["BB"] = [-3,3]

ylim_res = {}
ylim_res["TT"] = None
ylim_res["TE"] = [-6,6]
ylim_res["TB"] = [-6,6]
ylim_res["ET"] = [-6,6]
ylim_res["BT"] = [-6,6]
ylim_res["EE"] = [-4,4]
ylim_res["EB"] = [-3,3]
ylim_res["BE"] = [-3,3]
ylim_res["BB"] = [-3,3]

x_freq = ["90x90", "90x150", "150x150", "90x220", "150x220", "220x220"]


release = "all_of_them"
if release == "likelihood":
    my_spectra = ["TT", "TE", "EE"]
if release == "all_of_them":
    my_spectra = ["TT", "TE", "TB" ,"EE", "EB", "BB"]


loc_shift = [0, -5, 5, -10, 10, -15, 15, -20, 20]
xmax = 3000

for my_spectrum in my_spectra:

    lb, Db_fg_sub, std = np.loadtxt(f"{release_spec_dir}/combined/fg_subtracted_{my_spectrum}.dat", unpack=True)
    cov =  np.load(f"{release_spec_dir}/combined/cov_{my_spectrum}.npy")
    if my_spectrum == "TT":
        nspec = 7
    else:
        nspec = 4
    
    
    fig = plt.figure(constrained_layout=True, figsize=(18,8))
    gs =  gridspec.GridSpec(ncols=nspec, nrows=3, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, :])

    if my_spectrum == "TT":
        plt.semilogy()

    count = 0
    ax1.errorbar(lb + loc_shift[count], Db_fg_sub, std, fmt = ".", color="black", label = f"{my_spectrum}")
    if my_spectrum in ["TB", "EB", "BB"]:
        ax1.plot(lb, lb*0, linestyle = "--", color="black")

    ax = fig.add_subplot(gs[2, count])
    ax.matshow(so_cov.cov2corr(cov, remove_diag=True), vmin=-0.1, vmax=0.3)
    
    ax.set_xticks(np.arange(len(lb))[::5], lb[::5], rotation=90)
    ax.set_yticks(np.arange(len(lb))[::5], lb[::5])
    ax.set_title(f"{my_spectrum}")

    count += 1

    for  xf in x_freq:
        if ("220" in xf) & (my_spectrum != "TT"): continue
        lb, Db, std, Db_fg_sub = np.loadtxt(f"{release_spec_dir}/x_freqs/{xf}_{my_spectrum}.dat", unpack=True)
        cov =  np.load(f"{release_spec_dir}/x_freqs/cov_{xf}_{my_spectrum}.npy")

        ax1.errorbar(lb + loc_shift[count], Db_fg_sub, std, fmt=".", label = f"{xf} {my_spectrum}")
        ax = fig.add_subplot(gs[2, count])
        im = ax.matshow(so_cov.cov2corr(cov, remove_diag=True), vmin=-0.1, vmax=0.3)
        ax.set_xticks(np.arange(len(lb))[::5], lb[::5], rotation=90)
        ax.set_yticks(np.arange(len(lb))[::5], lb[::5])
        ax.set_title(f"{xf} {my_spectrum}")

        count += 1
        if count == nspec:
            plt.colorbar(im)

    ax1.set_xlim(400, xmax)
    ax1.set_ylim(ylim[my_spectrum])
    ax1.set_xlabel(r"$\ell$", fontsize=fontsize)
    ax1.set_ylabel(r"$ D^{%s}_{\ell} \ [\mu \rm K^{2}]$" % (my_spectrum), fontsize=fontsize)
    ax1.legend(fontsize=labelsize)
    ax1.tick_params(labelsize=labelsize)
    plt.savefig(f"{plot_dir}/combined/fg_subtracted_{my_spectrum}.pdf", bbox_inches="tight")
    plt.clf()
    plt.close()
    

    for xf in x_freq:
        if ("220" in xf) & (my_spectrum != "TT"): continue

        lb, Db, std, Db_fg_sub = np.loadtxt(f"{release_spec_dir}/x_freqs/{xf}_{my_spectrum}.dat", unpack=True)
        cov =  np.load(f"{release_spec_dir}/x_freqs/cov_{xf}_{my_spectrum}.npy")

        all_spectra = dataset_trace[my_spectrum, xf]
        
        nspec = len(all_spectra)
        fig = plt.figure(constrained_layout=True, figsize=(18,8))
        gs =  gridspec.GridSpec(ncols=nspec + 1, nrows=3, figure=fig)
        ax1 = fig.add_subplot(gs[0:2, :])

       # plt.figure(figsize=(18,8))
        if my_spectrum == "TT":
            plt.semilogy()
            
        count = 0
        ax1.errorbar(lb + loc_shift[count], Db, std, fmt = ".", color="black", label = f"{xf} {my_spectrum}")
        if my_spectrum in ["TB", "EB", "BB"]:
            ax1.plot(lb, lb*0, linestyle = "--", color="black")

        ax = fig.add_subplot(gs[2, count])
        ax.matshow(so_cov.cov2corr(cov, remove_diag=True), vmin=-0.1, vmax=0.3)
    
        ax.set_xticks(np.arange(len(lb))[::5], lb[::5], rotation=90)
        ax.set_yticks(np.arange(len(lb))[::5], lb[::5])
        ax.set_title(f"{xf} {my_spectrum}")
        count += 1

        for my_spec in all_spectra:
            s_name, spectrum = my_spec
            
            name = s_name.replace("dr6_", "")

            lb, Db, std, Db_th, Db_fg_th = np.loadtxt(f"{release_spec_dir}/x_array_bands/{s_name}_{spectrum}.dat", unpack=True)
            cov =  np.load(f"{release_spec_dir}/x_array_bands/cov_{s_name}_{spectrum}.npy")

            ax1.errorbar(lb + loc_shift[count], Db, std, fmt = ".", label= f"{name}")
            ax = fig.add_subplot(gs[2, count])
            ax.matshow(so_cov.cov2corr(cov, remove_diag=True), vmin=-0.1, vmax=0.3)
            ax.set_xticks(np.arange(len(lb))[::5], lb[::5], rotation=90)
            ax.set_yticks(np.arange(len(lb))[::5], lb[::5])
            ax.set_title(f"{name}")

            count += 1

        ax1.legend(fontsize=labelsize)
        ax1.set_ylim(ylim[my_spectrum])
        ax1.set_xlim(400, xmax)
        ax1.set_xlabel(r"$\ell$", fontsize=fontsize)
        ax1.set_ylabel(r"$ D^{%s}_{\ell} \ [\mu \rm K^{2}]$" % (my_spectrum), fontsize=fontsize)
        ax1.tick_params(labelsize=labelsize)

        plt.savefig(f"{plot_dir}/x_freqs/{xf}_{my_spectrum}.pdf", bbox_inches="tight")
        plt.clf()
        plt.close()

    all_spectra = dataset_trace[my_spectrum, "combined"]
    for my_spec in all_spectra:
        s_name, spectrum = my_spec
        lb, Db, std, Db_th, Db_fg_th = np.loadtxt(f"{release_spec_dir}/x_array_bands/{s_name}_{spectrum}.dat", unpack=True)

        plt.figure(figsize=(18,8))
        plt.suptitle(f"{s_name} {spectrum}", fontsize=fontsize)
        
        plt.subplot(2,1,1)
        if my_spectrum == "TT":
            plt.semilogy()
        plt.ylim(ylim[my_spectrum])
        plt.xlim(400, xmax)
        plt.errorbar(lb, Db, std, fmt = ".", color="red")
        plt.errorbar(lb, Db_th, color="gray")
        plt.errorbar(lb, Db_fg_th, color="lightblue")
        plt.xlabel(r"$\ell$", fontsize=fontsize)
        plt.ylabel(r"$ D^{%s}_{\ell} \ [\rm K^{2}]$" % (my_spectrum), fontsize=fontsize)
        plt.tick_params(labelsize=labelsize)

        plt.subplot(2,1,2)
        plt.errorbar(lb, Db - Db_th, std, fmt = ".")
        plt.plot(lb, lb * 0, color="black")
        plt.ylim(ylim_res[spectrum])
        plt.xlim(400, xmax)
        plt.xlabel(r"$\ell$", fontsize=fontsize)
        plt.ylabel(r"$ \Delta D^{%s}_{\ell} \ [\rm K^{2}]$" % (my_spectrum), fontsize=fontsize)
        plt.tick_params(labelsize=labelsize)

        plt.savefig(f"{plot_dir}/x_array_bands/{s_name}_{my_spectrum}.pdf", bbox_inches="tight")
        plt.clf()
        plt.close()


