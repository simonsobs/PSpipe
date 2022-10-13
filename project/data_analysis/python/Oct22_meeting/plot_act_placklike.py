"""
This script compare ACT spectra computed using planck window function and binning file to Planck spectra
"""
import numpy as np
import pylab as plt
import pspipe_utils
from pspipe_utils import pspipe_list, external_data
from pspy import so_dict
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

like_product_dir = "like_product"
freq_list = pspipe_list.get_freq_list(d)

lp, clp, errorp = {}, {}, {}

pl_data_path = os.path.join(os.path.dirname(os.path.abspath(pspipe_utils.__file__)), "data/spectra/planck")

fp_act = ["90x90", "150x150", "220x220"]
fp_pl = ["100x100", "143x143", "217x217"]

for spec in ["TT","EE"]:
    _, l_pl_dict, Db_pl_dict, err_pl_dict = external_data.get_planck_spectra(pl_data_path, spec, return_Dl=True)

    for fpa, fpp in zip(fp_act, fp_pl):

        lb, Db, sigmab = np.loadtxt(f"{like_product_dir}/spectra_xfreq_{spec}_{fpa}.dat", unpack=True)
        lb_pl, Db_pl, sigmab_pl = l_pl_dict[fpp], Db_pl_dict[fpp], err_pl_dict[fpp]
    
        if spec == "TT":
            plt.figure(figsize=(12, 8))
            plt.semilogy()
            plt.errorbar(lb_pl, Db_pl, sigmab_pl, fmt=".", label=f"Planck {fpp} legacy public data", alpha=0.5)
            plt.errorbar(lb, Db, sigmab, fmt=".", label=f"ACT {fpa} (with Planck likelihood mask)", alpha=0.5)
            plt.legend(fontsize=18)
            plt.savefig(f"ACT_and_planck_{spec}_{fpa}.png", bbox_inches="tight")
            plt.clf()
            plt.close()
            
        id_pl = np.where(lb_pl > 150)
        id = np.where(lb > 150)

        lb_pl, Db_pl, sigmab_pl = lb_pl[id_pl], Db_pl[id_pl], sigmab_pl[id_pl]
        lb, Db, sigmab  = lb[id], Db[id], sigmab[id]
        
        nbins = len(lb_pl)

        if spec == "TT":
            plt.figure(figsize=(12, 8))
            plt.plot(lb[:nbins], lb[:nbins]*0+1)
            plt.plot(lb[:nbins], Db[:nbins]/Db_pl)
            plt.savefig(f"full_sky_TF_{spec}_{fpa}.png", bbox_inches="tight")
            plt.clf()
            plt.close()

        if spec == "EE":
            plt.figure(figsize=(12, 8))
            plt.errorbar(lb[:nbins], Db[:nbins], sigmab_pl, fmt=".", label=f"Planck {fpp} legacy public error", alpha=0.5)
            plt.errorbar(lb, Db, sigmab, fmt=".", label=f"ACT {fpa} (with Planck likelihood mask)", alpha=0.5)
            plt.legend(fontsize=18)
            plt.ylim(-10, 50)
            plt.savefig(f"ACT_and_planck_{spec}_{fpa}_blinded.png", bbox_inches="tight")
            plt.clf()
            plt.close()

    
    
