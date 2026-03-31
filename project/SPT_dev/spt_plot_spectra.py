import pylab as plt
import numpy as np
import sys
from copy import deepcopy

from pspy import pspy_utils, so_dict, so_mcm, so_spectra
from pspipe_utils import log, pspipe_list
import candl
import spt_candl_data


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

lmax = d["lmax"]
binning_file = d["binning_file"]
release_dir = d["release_dir"]

mcm_dir = "mcms"
plot_dir = "plots"
pspy_utils.create_directory(plot_dir)


def correct_spt_transfer_function(lb, ps, spec_name, Bbl):

    ps_corrected = deepcopy(ps)
    
    tf_dir = f"{release_dir}/ancillary_products/specific_to_c25/"
    
    name = spec_name.replace("spt_", "")
    fa, fb = name.split("x")
    print(fa, fb)
    
    tf = {}
    for mode in ["tt", "te", "et", "ee"]:
        tf_file = f"{tf_dir}/filter_transfer_function_c25_v1_{fa}ghz{fb}ghz_{mode}.txt"
        l, tf[mode] = np.loadtxt(tf_file, unpack=True)
        
    tf["tt"] = np.dot(Bbl["spin0xspin0"], tf["tt"][:lmax])
    tf["te"] = np.dot(Bbl["spin0xspin2"], tf["te"][:lmax])
    tf["et"] = np.dot(Bbl["spin2xspin0"], tf["et"][:lmax])
    
    nbins, my_lmax = int(Bbl["spin2xspin2"].shape[0]/4), int(Bbl["spin2xspin2"].shape[1]/4)
    tf["ee"] = np.dot(Bbl["spin2xspin2"][:nbins, :my_lmax], tf["ee"][:lmax])
    
    for mode in ["tt", "te", "et", "ee"]:
        ps_corrected[mode.upper()] = ps[mode.upper()] / tf[mode]
    return lb, ps_corrected
    
    
def correct_spt_additive_bias(lb, ps, spec_name, Bbl):

    ps_corrected = deepcopy(ps)
    
    tf_dir = f"{release_dir}/ancillary_products/specific_to_c25/"
    
    name = spec_name.replace("spt_", "")
    fa, fb = name.split("x")
    print(fa, fb)
    
    additive_bias = {}
    for mode in ["tt", "te", "et", "ee"]:
        filter_artefact_bias_file = f"{tf_dir}/filtering_artifact_bias_{fa}ghz{fb}ghz_{mode}.txt"
        l,  filter_artefact_bias = np.loadtxt(filter_artefact_bias_file, unpack=True)
        inpainting_bias_file = f"{tf_dir}/inpainting_bias_{fa}ghz{fb}ghz_{mode}.txt"
        l,  inpainting_bias = np.loadtxt(inpainting_bias_file, unpack=True)
        additive_bias[mode] = (filter_artefact_bias + inpainting_bias) * l * (l + 1) / (2 * np.pi) #corrections are in Cl

        
    additive_bias["tt"] = np.dot(Bbl["spin0xspin0"], additive_bias["tt"][:lmax])
    additive_bias["te"] = np.dot(Bbl["spin0xspin2"], additive_bias["te"][:lmax])
    additive_bias["et"] = np.dot(Bbl["spin2xspin0"], additive_bias["et"][:lmax])
    
    nbins, my_lmax = int(Bbl["spin2xspin2"].shape[0]/4), int(Bbl["spin2xspin2"].shape[1]/4)
    additive_bias["ee"] = np.dot(Bbl["spin2xspin2"][:nbins, :my_lmax], additive_bias["ee"][:lmax])
    
    for mode in ["tt", "te", "et", "ee"]:
        ps_corrected[mode.upper()] = ps[mode.upper()] - additive_bias[mode]
    return lb, ps_corrected
    
candl_like = candl.Like(spt_candl_data.SPT3G_D1_TnE)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

camphuis_conv = {}
camphuis_conv["spt_095xspt_095"] = "90x90"
camphuis_conv["spt_095xspt_150"] = "90x150"
camphuis_conv["spt_095xspt_220"] = "90x220"
camphuis_conv["spt_150xspt_150"] = "150x150"
camphuis_conv["spt_150xspt_220"] = "150x220"
camphuis_conv["spt_220xspt_220"] = "220x220"


spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

Db_dict = {}
Db_dict_bias_corr = {}
Db_dict_tf_corr = {}
Db_dict_bias_tf_corr = {}
    
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

for spec_name in spec_name_list:
    
    _, Bbl = so_mcm.read_coupling(prefix=f"{mcm_dir}/{spec_name}", spin_pairs=spin_pairs)

    lb, Db_dict[spec_name] = so_spectra.read_ps(f"spectra/Dl_{spec_name}_cross.dat", spectra=spectra)
    lb, Db_dict_bias_corr[spec_name] = correct_spt_additive_bias(lb, Db_dict[spec_name], spec_name, Bbl)
    lb, Db_dict_tf_corr[spec_name] = correct_spt_transfer_function(lb, Db_dict[spec_name], spec_name, Bbl)
    
    lb, Db_dict_bias_tf_corr[spec_name] = correct_spt_transfer_function(lb, Db_dict_bias_corr[spec_name], spec_name, Bbl)

   # ps_dict[spec_name] = Db.copy()
    
cosmo_params = d["cosmo_params"]
l_th, ps_th = pspy_utils.ps_from_params(cosmo_params, type, lmax + 500)

for spec in ["TB", "EB", "BB"]:
    for spec_name in spec_name_list:
       plt.plot(l_th[:3500], ps_th[spec][:3500], color="black")
       plt.plot(lb, Db_dict_bias_tf_corr[spec_name][spec], label=f"{spec} {spec_name} (uncorrected)")
       plt.legend()
       plt.savefig(f"{plot_dir}/{spec_name}_{spec}.png", bbox_inches="tight")
       plt.clf()
       plt.close()
    

for spec in ["TT", "TE", "EE"]:
    for spec_name in spec_name_list:

        spec_to_plot = f"{spec} {camphuis_conv[spec_name]}"
        ix_of_spec = candl_like.spec_order.index(spec_to_plot)
        l_spt = candl_like.effective_ells[candl_like.bins_start_ix[ix_of_spec]:candl_like.bins_stop_ix[ix_of_spec]]
        Db = candl_like.data_bandpowers[candl_like.bins_start_ix[ix_of_spec]:candl_like.bins_stop_ix[ix_of_spec]]
        sigmab = np.sqrt(np.diag(candl_like.covariance)[candl_like.bins_start_ix[ix_of_spec]:candl_like.bins_stop_ix[ix_of_spec]])
        
        id_redo = np.where((lb>=l_spt[0]) & (lb<=l_spt[-1]+5)) # the +5 is a fudge factor becasue spt use some sort of effective bins
        id_spt = np.where((l_spt>=lb[0]) & (l_spt<=lb[-1]))


        l_spt, Db, sigmab = l_spt[id_spt], Db[id_spt], sigmab[id_spt]
        lb_redo, Db_redo = lb[id_redo], Db_dict[spec_name][spec][id_redo]
        Db_redo_tf_corr, Db_redo_bias_tf_corr = Db_dict_tf_corr[spec_name][spec][id_redo], Db_dict_bias_tf_corr[spec_name][spec][id_redo]

        plt.figure(figsize=(12,8))
        
        plt.subplot(311)
        if spec in ["TT", "EE"]:
            plt.semilogy()
        plt.errorbar(l_spt, Db, sigmab, lw=0.5, marker="o", ms=3, elinewidth=1, label=f"SPT {spec_name}")
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(r"$D_\ell$", fontsize=14)
        plt.plot(lb_redo, Db_redo_bias_tf_corr, label=f"SPT redo  {spec_name}, bias tf corrected")

        plt.legend()
        plt.subplot(312)
        plt.errorbar(l_spt, l_spt*0)
        plt.errorbar(l_spt, Db-Db_redo_bias_tf_corr, sigmab, lw=0.5, marker="o", ms=3, elinewidth=1, label=f"SPT - SPT redo, bias tf corrected {spec_name}")
        plt.legend()
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(r"$D_\ell - D^{\rm redo}_\ell$", fontsize=14)

        plt.subplot(313)
        plt.errorbar(l_spt, Db/Db_redo_bias_tf_corr, lw=0.5, marker="o", ms=3, elinewidth=1, label=f"SPT/ SPT redo {spec_name}")
        plt.xlabel(r"$\ell$", fontsize=14)
        plt.ylabel(r"$D_\ell / D^{\rm redo}_\ell$", fontsize=14)

        plt.ylim(0.95, 1.05)
        plt.savefig(f"{plot_dir}/{spec_name}_{spec}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
