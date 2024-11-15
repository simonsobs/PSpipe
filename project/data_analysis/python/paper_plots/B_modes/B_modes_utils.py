"""
"""

from pspy import so_dict, pspy_utils
from pspipe_utils import covariance
import numpy as np
import pylab as plt
import sys, os


def get_model_vecs(l_th, ps_dict, fg_dict, lmax,
                   binning_file, spec_name_list, spectra):
                              
    vec_cmb, vec_dust = [],  []
    for spec in spectra:
        lb, ps_b =  pspy_utils.naive_binning(l_th, ps_dict[spec], binning_file, lmax)
        for spec_name in spec_name_list:
            na, nb = spec_name.split("x")
            if spec in ["EB", "BE"]:
                dust = l_th * 0
            else:
                dust = fg_dict[spec.lower(), "dust", na, nb]
                
            lb, dust_b = pspy_utils.naive_binning(l_th, dust, binning_file, lmax)
            
            if (spec == "ET" or spec == "BT" or spec == "BE") & (na == nb): continue
            vec_dust = np.append(vec_dust, dust_b)
            vec_cmb = np.append(vec_cmb, ps_b)
                
    return vec_dust, vec_cmb

def get_P_mat_BB(vec_size, bin_out_dict, bin_scheme_edge, fig_name=None):
    """
    Very naive "pointing" matrix for maximum likelihood combination of the spectra
    it should be vectorized but writing the loops was trivial and the code is sufficienctly fast
    """
    P_mat = np.zeros((vec_size, len(bin_scheme_edge)))
    index1 = 0
    y_ticks, y_name = [], []
    for my_spec in bin_out_dict.keys():
        s_name, _ = my_spec
        id_spec, lb_spec = bin_out_dict[my_spec]
        # for plotting
        y_ticks += [index1]
        y_name += [s_name]
        
        for jj in lb_spec:
            for index2, my_bin in enumerate(bin_scheme_edge):
                min, max = my_bin
                if (min <= jj) & (jj <= max):
                    P_mat[index1, index2] = 1
            index1 += 1

    if fig_name is not None:
        plt.figure(figsize=(12, 12))
        plt.imshow(P_mat)
        plt.yticks(y_ticks, y_name)
        plt.xticks(np.arange(len(bin_scheme_edge)),bin_scheme_edge, rotation=90)
        plt.gca().set_aspect(0.02)
        plt.tight_layout()
        plt.savefig(f"{fig_name}", dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()
        
    return P_mat


def get_fg_sub_ML_solution_BB(lb, vec_BB, vec_dust_BB, i_cov_BB, bin_out_dict, bin_scheme_edge, fig_name=None):


    P_mat_BB = get_P_mat_BB(len(vec_BB), bin_out_dict, bin_scheme_edge, fig_name=fig_name)
    

    cov_ML_BB = covariance.get_max_likelihood_cov(P_mat_BB,
                                                  i_cov_BB,
                                                  force_sim = True,
                                                  check_pos_def = True)
                                                  
    vec_ML_BB = covariance.max_likelihood_spectra(cov_ML_BB,
                                                 i_cov_BB,
                                                 P_mat_BB,
                                                 vec_BB - vec_dust_BB)
                                                 
    lb_all = []
    for my_spec in bin_out_dict.keys():
        s_name, spectrum = my_spec
        id, lb = bin_out_dict[my_spec]
        lb_all = np.append(lb_all, lb)
        
    lb_ML_BB = covariance.max_likelihood_spectra(cov_ML_BB,
                                                 i_cov_BB,
                                                 P_mat_BB,
                                                 lb_all)
                                                 
    return lb_ML_BB, vec_ML_BB, cov_ML_BB


def get_ml_bins(bin_out_dict, bin_mean):

    """
    Find the bin range of the maximum likelihood combination of the data
    we go through all spectra in the list and choose as a minimum bin the minimum bin
    of all spectra that enters the combinaison and as a maximum the maximum bin
    of all spectra that enters the combinaison
    """
    
    min_lb_list, max_lb_list = [], []
    for my_spec in bin_out_dict.keys():
        id_spec, lb_spec = bin_out_dict[my_spec]
        min_lb_list += [np.min(lb_spec)]
        max_lb_list += [np.max(lb_spec)]

    min_lb_comb = np.min(min_lb_list, axis=0)
    max_lb_comb = np.max(max_lb_list, axis=0)

    ml_id = np.where((bin_mean >= min_lb_comb) & (bin_mean <= max_lb_comb))
    ml_lb = bin_mean[ml_id]
    
    return ml_lb


def get_P_mat(vec_size, lb_ml, bin_out_dict, fig_name=None):

    """
    Very naive "pointing" matrix for maximum likelihood combination of the spectra
    it should be vectorized but writing the loops was trivial and the code is sufficienctly fast
    """
    
    P_mat = np.zeros((vec_size, len(lb_ml)))
    
    index1 = 0
    y_ticks, y_name = [], []
    
    for my_spec in bin_out_dict.keys():
        s_name, _ = my_spec
        id_spec, lb_spec = bin_out_dict[my_spec]
        
        # for plotting
        y_ticks += [index1]
        y_name += [s_name]
        
        for jj in lb_spec:
            for index2, ii in enumerate(lb_ml):
                if ii == jj:
                    P_mat[index1, index2] = 1
            index1 += 1

    if fig_name is not None:
        plt.figure(figsize=(12,8))
        plt.imshow(P_mat, aspect="auto")
        plt.yticks(y_ticks, y_name)
        plt.xticks(np.arange(len(lb_ml))[::2], lb_ml[::2], rotation=90)
        plt.tight_layout()
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
        plt.clf()
        plt.close()
        
    return P_mat


def get_spectra_cuts(cut, lmax):

    if cut == "pre_unblinding":
        spectra_cuts = {
            "dr6_pa4_f220": dict(T=[975, lmax], P=[lmax, lmax]),
            "dr6_pa5_f150": dict(T=[475, lmax], P=[475, lmax]),
            "dr6_pa6_f150": dict(T=[475, lmax], P=[475, lmax]),
            "dr6_pa5_f090": dict(T=[475, lmax], P=[475, lmax]),
            "dr6_pa6_f090": dict(T=[475, lmax], P=[475, lmax])}
    if cut == "post_unblinding":
        spectra_cuts = {
            "dr6_pa4_f220": dict(T=[975, lmax], P=[lmax, lmax]),
            "dr6_pa5_f150": dict(T=[775, lmax], P=[775, lmax]),
            "dr6_pa6_f150": dict(T=[575, lmax], P=[575, lmax]),
            "dr6_pa5_f090": dict(T=[975, lmax], P=[975, lmax]),
            "dr6_pa6_f090": dict(T=[975, lmax], P=[975, lmax])}
            
    return spectra_cuts
