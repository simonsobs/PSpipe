"""
This script compare analytical covariances and monte carlo covariances 
"""



import matplotlib
matplotlib.use("Agg")
from pspy import pspy_utils, so_dict, so_cov
import  numpy as np
import pylab as plt
import os, sys
from pspipe_utils import pspipe_list

def write_html(filename, spec_list, multistep_path, cov_plot_dir):

    os.system('cp %s/multistep2.js %s/multistep2.js'%(multistep_path,cov_plot_dir))
    file = '%s/%s.html' % (cov_plot_dir, filename)
    g = open(file, mode="w")
    g.write('<html>\n')
    g.write('<head>\n')
    g.write('<title> %s </title>\n'% filename)
    g.write('<script src="multistep2.js"></script>\n')
    g.write('<script> add_step("sub",  ["c","v"]) </script> \n')
    g.write('<style> \n')
    g.write('body { text-align: center; } \n')
    g.write('img { width: 100%; max-width: 1200px; } \n')
    g.write('</style> \n')
    g.write('</head> \n')
    g.write('<body> \n')
    g.write('<div class=sub>\n')
    for sid1, spec1 in enumerate(spec_list):
        for sid2, spec2 in enumerate(spec_list):
            if sid1>sid2: continue
            
            n1,n2=spec1.split('x')
            n3,n4=spec2.split('x')
            
            str='%s_%s_%s.png'%(filename, spec1, spec2)
            g.write('<div class=sub>\n')
            g.write('<img src="'+str+'"  /> \n')
            g.write('</div>\n')

    g.write('</body> \n')
    g.write('</html> \n')
    g.close()


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
cov_plot_dir = "plots/covariance"

surveys = d["surveys"]
lmax = d["lmax"]
binning_file = d["binning_file"]
multistep_path = d["multistep_path"]

pspy_utils.create_directory(cov_plot_dir)

spec_list = pspipe_list.get_spec_name_list(d, char="_")


for sid1, spec1 in enumerate(spec_list):
    for sid2, spec2 in enumerate(spec_list):
        if sid1 > sid2 : continue
        
        n1, n2 = spec1.split("x")
        n3, n4 = spec2.split("x")
        
        analytic_cov = np.load(f"{cov_dir}/analytic_cov_{n1}x{n2}_{n3}x{n4}.npy")
        mc_cov = np.load(f"{cov_dir}/mc_cov_{n1}x{n2}_{n3}x{n4}.npy")
        
        bin_lo, bin_hi, lb, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
        n_bins = len(bin_hi)

        plt.figure(figsize=(15, 15))
        plt.suptitle(f"{spec1} {spec2} (press c/v to switch between covariance matrix elements)", fontsize=30)
        count = 1
        for bl in ["TTTT", "TETE", "ETET", "EEEE",
                   "TTTE", "TTEE", "TTET", "TEET",
                   "TEEE", "ETEE",
                   "EETE", "EEET", "ETTE", "ETTT",
                   "EETT", "TETT"]:
            
            mc_cov_sub = so_cov.selectblock(mc_cov, ["TT", "TE", "ET", "EE"], n_bins, block=bl)
            analytic_cov_sub= so_cov.selectblock(analytic_cov, ["TT", "TE", "ET", "EE"], n_bins, block=bl)
            
            var = mc_cov_sub.diagonal()
            analytic_var = analytic_cov_sub.diagonal()

            plt.subplot(3,6,count)
            if count == 1:
                plt.semilogy()
            plt.plot(lb[1:], var[1:], ".", label="MC %sx%s" % (bl[:2],bl[2:4]))
            plt.plot(lb[1:], analytic_var[1:], label = "Analytic %sx%s" % (bl[:2],bl[2:4]))
            if count == 1 or count == 4:
                plt.ylabel(r"$Cov_{i,i,\ell}$", fontsize=22)
            if count > 3:
                plt.xlabel(r"$\ell$", fontsize=22)
            plt.legend()
            count += 1
        plt.savefig(f"{cov_plot_dir}/covariance_pseudo_diagonal_{spec1}_{spec2}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
        
        plt.figure(figsize=(15, 15))
        plt.suptitle(f"{spec1} {spec2} (press c/v to switch between covariance matrix elements)", fontsize=30)
        count = 1
        for bl in ["TTTT", "TETE", "ETET", "EEEE"]:
            
            mc_cov_sub = so_cov.selectblock(mc_cov, ["TT", "TE", "ET", "EE"], n_bins, block=bl)
            analytic_cov_sub= so_cov.selectblock(analytic_cov, ["TT", "TE", "ET", "EE"], n_bins, block=bl)
            
            var = mc_cov_sub.diagonal()
            analytic_var = analytic_cov_sub.diagonal()

            plt.subplot(4,1,count)
            plt.plot(lb[1:], var[1:]/analytic_var[1:], label="MC/Analytic %sx%s" % (bl[:2],bl[2:4]))
            if count == 1:
                plt.ylabel(r"Ratio $Cov_{i,i,\ell}$", fontsize=22)
            if count == 4:
                plt.xlabel(r"$\ell$", fontsize=22)
            plt.legend()
            count += 1
        plt.savefig(f"{cov_plot_dir}/covariance_pseudo_diagonal_ratio_{spec1}_{spec2}.png", bbox_inches="tight")
        plt.clf()
        plt.close()
        

        analytic_corr=so_cov.cov2corr(analytic_cov)
        mc_corr=so_cov.cov2corr(mc_cov)

        plt.figure(figsize=(15, 15))
        plt.suptitle(f"{spec1} {spec2} (press c/v to switch between covariance matrix elements)", fontsize=30)
        count = 1
        for bl in ["TTTT", "TETE", "ETET", "EEEE",
                    "TTTE", "TTEE", "TTET", "TEET",
                    "TEEE", "ETEE",
                    "EETE", "EEET", "ETTE", "ETTT",
                    "EETT", "TETT"]:
            
            mc_corr_sub = so_cov.selectblock(mc_corr, ["TT", "TE", "ET", "EE"], n_bins, block=bl)
            analytic_corr_sub= so_cov.selectblock(analytic_corr, ["TT", "TE", "ET", "EE"], n_bins, block=bl)
            
            off_diag = mc_corr_sub.diagonal(1)
            analytic_off_diag = analytic_corr_sub.diagonal(1)
            
            plt.subplot(3,6,count)
            plt.plot(off_diag[1:], ".", label="MC %sx%s" % (bl[:2],bl[2:4]))
            plt.plot(analytic_off_diag[1:], label = "Analytic %sx%s" % (bl[:2],bl[2:4]))
            if count == 1 or count == 4:
                plt.ylabel(r"$Cov_{i,i+1,\ell}$", fontsize=22)
            if count > 3:
                plt.xlabel(r"$\ell$", fontsize=22)
            plt.legend()
            count += 1
        plt.savefig(f"{cov_plot_dir}/covariance_off_diagonal_{spec1}_{spec2}.png", bbox_inches="tight")
        plt.clf()
        plt.close()

filename = "covariance_pseudo_diagonal"
write_html(filename, spec_list, multistep_path, cov_plot_dir)












