from pspy import so_spectra
import pylab as plt
import pspipe
from pspipe_utils import covariance, pspipe_list, log
from pspy import so_cov, so_dict, pspy_utils
import sys
import numpy as np
import scipy.stats as ss
import os
from matplotlib import rcParams


rcParams["font.family"] = "serif"
rcParams["font.size"] = "40"
rcParams["xtick.labelsize"] = 16
rcParams["ytick.labelsize"] = 16
rcParams["axes.labelsize"] = 16
rcParams["axes.titlesize"] = 16


def pte_histo(pte_list, file_name, n_bins):
    n_samples = len(pte_list)
    bins = np.linspace(0, 1, n_bins + 1)
    min, max = np.min(pte_list), np.max(pte_list)

    id_high = np.where(pte_list > 0.99)
    id_low = np.where(pte_list < 0.01)
    nPTE_high = len(id_high[0])
    nPTE_low =  len(id_low[0])
    
    plt.figure(figsize=(8,6))
    plt.title("Isotropy test: North/South", fontsize=16)
    plt.xlabel(r"Probability to exceed (PTE)", fontsize=16)
    plt.hist(pte_list, bins=bins, label=f"n tests: {n_samples}, min: {min:.3f}, max: {max:.3f}", histtype='bar', facecolor="orange", edgecolor="black", linewidth=3)
    plt.axhline(n_samples/n_bins, color="k", ls="--", alpha=0.5)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.legend(fontsize=16)
    plt.savefig(f"{file_name}", dpi=300, bbox_inches='tight')
    plt.clf()
    plt.close()

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


sim_spec_dir_north = d["sim_spec_dir_north"]
sim_spec_dir_south = d["sim_spec_dir_south"]

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

binning_file = "BIN_ACTPOL_50_4_SC_large_bin_at_low_ell"
lmax = 8500

lmin = {}
lmin["dr6_pa4_f220", "TT"] = 1000
lmin["dr6_pa5_f090", "TT"] = 1000
lmin["dr6_pa5_f150", "TT"] = 800
lmin["dr6_pa6_f090", "TT"] = 1000
lmin["dr6_pa6_f150", "TT"] = 600

lmin["dr6_pa4_f220", "TE"] = 1000
lmin["dr6_pa5_f090", "TE"] = 1000
lmin["dr6_pa5_f150", "TE"] = 800
lmin["dr6_pa6_f090", "TE"] = 1000
lmin["dr6_pa6_f150", "TE"] = 600

lmin["dr6_pa4_f220", "ET"] = 1000
lmin["dr6_pa5_f090", "ET"] = 1000
lmin["dr6_pa5_f150", "ET"] = 800
lmin["dr6_pa6_f090", "ET"] = 1000
lmin["dr6_pa6_f150", "ET"] = 600

lmin["dr6_pa4_f220", "EE"] = 1000
lmin["dr6_pa5_f090", "EE"] = 1000
lmin["dr6_pa5_f150", "EE"] = 800
lmin["dr6_pa6_f090", "EE"] = 1000
lmin["dr6_pa6_f150", "EE"] = 600


lmin["dr6_pa4_f220", "BB"] = 1000
lmin["dr6_pa5_f090", "BB"] = 1000
lmin["dr6_pa5_f150", "BB"] = 800
lmin["dr6_pa6_f090", "BB"] = 1000
lmin["dr6_pa6_f150", "BB"] = 600

lmin["dr6_pa4_f220", "TB"] = 1000
lmin["dr6_pa5_f090", "TB"] = 1000
lmin["dr6_pa5_f150", "TB"] = 800
lmin["dr6_pa6_f090", "TB"] = 1000
lmin["dr6_pa6_f150", "TB"] = 600

lmin["dr6_pa4_f220", "BT"] = 1000
lmin["dr6_pa5_f090", "BT"] = 1000
lmin["dr6_pa5_f150", "BT"] = 800
lmin["dr6_pa6_f090", "BT"] = 1000
lmin["dr6_pa6_f150", "BT"] = 600

lmin["dr6_pa4_f220", "EB"] = 1000
lmin["dr6_pa5_f090", "EB"] = 1000
lmin["dr6_pa5_f150", "EB"] = 800
lmin["dr6_pa6_f090", "EB"] = 1000
lmin["dr6_pa6_f150", "EB"] = 600

lmin["dr6_pa4_f220", "BE"] = 1000
lmin["dr6_pa5_f090", "BE"] = 1000
lmin["dr6_pa5_f150", "BE"] = 800
lmin["dr6_pa6_f090", "BE"] = 1000
lmin["dr6_pa6_f150", "BE"] = 600




null_dir = "isotropy"
pspy_utils.create_directory(null_dir)

nsims = 300

my_ylim = {}
my_ylim["TT"] = [-200,200]
my_ylim["EB"] = [-10,10]
my_ylim["TB"] = [-10,10]
my_ylim["TE"] = [-30,30]
my_ylim["BE"] = [-10,10]
my_ylim["BT"] = [-10,10]
my_ylim["ET"] = [-30,30]
my_ylim["BB"] = [-10,10]
my_ylim["EE"] = [-10,10]


multistep_path = os.path.join(os.path.dirname(pspipe.__file__), "js")
os.system(f"cp {multistep_path}/multistep2.js .")

filename = f"isotropy.html"
g = open(filename, mode='w')
g.write('<html>\n')
g.write('<head>\n')
g.write('<title> isotropy </title>\n')
g.write('<script src="multistep2.js"></script>\n')
g.write('<script> add_step("null", ["c","v"]) </script> \n')
g.write('<script> add_step("spec", ["j","k"]) </script> \n')
g.write('<script> add_step("array", ["a","z"]) </script> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write('<h1>isotropy </h1>')
g.write('<p> In this webpage we host all plots for isotropy  tests, we have divided the observed area into two regions of the sky </p>')
g.write('<p> we have done N tests corresponding to the following  PTE distribution </p>')
g.write('<img src="' + 'isotropy/pte_isotropy.png' + '" width="50%" /> \n')
g.write('<p> you can have a look at all spectra, press a/z to change the null, c/v to change spectrum (TT,TE, .. BB) </p>')
g.write('<div class=null> \n')


pte_list = []
for my_spec in spectra:

    g.write('<div class=array>\n')

    for spec in spec_name_list:
    
        lb, Db_north = so_spectra.read_ps(f"patch_north/spectra/Dl_{spec}_cross.dat", spectra=spectra)
        lb, Db_south = so_spectra.read_ps(f"patch_south/spectra/Dl_{spec}_cross.dat", spectra=spectra)

        lth, bf_north = so_spectra.read_ps(f"patch_north/best_fits/cmb_and_fg_{spec}.dat", spectra=spectra)
        lth, bf_south = so_spectra.read_ps(f"patch_south/best_fits/cmb_and_fg_{spec}.dat", spectra=spectra)

        lb, bf_north_b = pspy_utils.naive_binning(lth, bf_north[my_spec], binning_file, lmax)
        lb, bf_south_b = pspy_utils.naive_binning(lth, bf_south[my_spec], binning_file, lmax)

        n_bins = len(lb)
        na, nb = spec.split("x")
        my_lmin = np.maximum(lmin[na, my_spec], lmin[nb, my_spec])

        diff_bf = bf_north_b - bf_south_b
        
        #Db_north[my_spec] *= 1.015
        diff = Db_north[my_spec] - Db_south[my_spec]
        
        diff_all = []
        for iii in range(nsims):
            lb, Db_north_sim = so_spectra.read_ps(f"{sim_spec_dir_north}/Dl_{spec}_cross_{iii:05d}.dat", spectra=spectra)
            lb, Db_south_sim = so_spectra.read_ps(f"{sim_spec_dir_south}/Dl_{spec}_cross_{iii:05d}.dat", spectra=spectra)
            diff_all += [Db_north_sim[my_spec] - Db_south_sim[my_spec]]
            
        diff_mean = np.mean(diff_all, axis=0)
        diff_std = np.std(diff_all, axis=0)
        diff_cov_mc = np.cov(diff_all, rowvar=False)

        id = np.where(lb > my_lmin)
        
        cov_north = np.load(f"patch_north/covariances/analytic_cov_{spec}_{spec}.npy")
        cov_south = np.load(f"patch_south/covariances/analytic_cov_{spec}_{spec}.npy")

        diff_cov = cov_north + cov_south
        diff_cov = so_cov.selectblock(diff_cov, spectra, n_bins, block=my_spec+my_spec)
        
        an_var = diff_cov.diagonal()
        mc_var = diff_cov_mc.diagonal()

        print(my_spec, spec)
        diff_cov_mc = diff_cov - np.diag(an_var) + np.diag(mc_var)

        diff_cov_mc = diff_cov_mc[np.ix_(id[0], id[0])]
        
        chi2 = (diff[id] - diff_bf[id]) @ np.linalg.inv(diff_cov_mc) @ (diff[id] - diff_bf[id])
        chi2_diag = np.sum((diff[id] - diff_bf[id]) ** 2 / diff_std[id] ** 2)
        print(my_spec, spec, "diag",chi2_diag, "full", chi2)
        
        
        ndof = len(lb[id])
        pte = 1 - ss.chi2(ndof).cdf(chi2)
        
        #plt.ylim(0,1.5)

        plt.show()
        
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1)
        plt.title(spec+ my_spec)
        plt.errorbar(lb, diff - diff_bf, diff_std, fmt="o", label=f"p = {pte:.3f}")
        plt.errorbar(lb, lb*0)
        plt.ylim(my_ylim[my_spec][0],my_ylim[my_spec][1] )
        
        xleft, xright = lb[id][0], lb[id][-1]
        plt.axvspan(xmin=0, xmax=xleft, color="gray", alpha=0.7)
        if xright != lb[-1]:
            plt.axvspan(xmin=xright, xmax=lb[-1], color="gray", alpha=0.7)

        plt.xlabel(r"$\ell$", fontsize=18)
        plt.ylabel(r"$\Delta D_\ell^\mathrm{%s}$" % (my_spec), fontsize=18)

        plt.legend(fontsize=16, loc="upper right")
        plt.title(f"north-south : {spec.replace('dr6_','')}", fontsize=16)
        plt.subplot(2,1,2)
        plt.ylim(0.5,1.5)
        plt.plot(lb, Db_north[my_spec] / Db_south[my_spec], label="data")
        plt.plot(lb, bf_north_b / bf_south_b, label="best fit (dust)")
        plt.xlabel(r"$\ell$", fontsize=18)
        plt.ylabel(r"$ D_\ell^\mathrm{%s, north} / D_\ell^\mathrm{%s, south}$" % (my_spec, my_spec), fontsize=18)
        plt.legend()
        plt.tight_layout()
        
        
        str = f"{null_dir}/spectra_{spec}_{my_spec}.png"


        plt.savefig(f"{str}")
        plt.clf()
        plt.close()
        
        g.write('<img src="' + str + '" width="50%" /> \n')

        if ("pa4_f220" in spec) & (spec != "TT"):
            continue
        if (na == nb) & (spec in ["ET","BT","BE"]):
            continue
        pte_list = np.append(pte_list, pte)
        
    g.write('</div>\n')


n_bins = 14
print(np.min(pte_list), np.max(pte_list))
pte_histo(pte_list, f"{null_dir}/pte_isotropy_all.png", n_bins)

g.write('</div> \n')
g.write('</body> \n')
g.write('</html> \n')
g.close()

