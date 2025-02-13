import numpy as np
import pylab as plt
import sys
from pspy import pspy_utils, so_map_preprocessing, so_dict, so_map
from pspipe_utils import kspace, log
import matplotlib

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])
log = log.get_logger(**d)

matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.size"] = "20"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
lmax = d["lmax"]
binning_file = d["binning_file"]
filter_dict = d[f"k_filter_dr6"]
kspace_tf_path = d["kspace_tf_path"]
window_dir =  d["window_dir"]

kspace_matrix = np.load(f"{kspace_tf_path}/kspace_matrix_dr6_pa6_f150xdr6_pa6_f150.npy")
templates = so_map.read_map(f"{window_dir}/window_dr6_pa5_f150_baseline.fits")

_, _, lb, _ = pspy_utils.read_binning_file(binning_file, lmax)
n_bins = len(lb)


#filter = kspace.get_kspace_filter(templates, filter_dict)
#_, kf_tf_analytic = so_map_preprocessing.analytical_tf(templates, filter, binning_file, lmax)

#np.save("kf_tf_analytic.npy", kf_tf_analytic)

kf_tf_analytic = np.load("kf_tf_analytic.npy")

kf_tf_mc = {}
for i, spec in enumerate(spectra):
    sub_kspace = kspace_matrix[i * n_bins: (i+1) * n_bins, i * n_bins: (i+1) * n_bins]
    kf_tf_mc[spec] = sub_kspace.diagonal()


f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
f.set_figheight(8)
f.set_figwidth(12)

#plt.subplot(2,1,1)
a0.plot(lb, kf_tf_analytic, color="black", label="analytic")
a0.errorbar(lb, kf_tf_mc["TT"], fmt="o", color="red", label=r"$\left\langle \frac{C^{TT, F}_{\ell}}{C^{TT}_{\ell}} \right\rangle$")
a0.errorbar(lb, kf_tf_mc["EE"], fmt="o", color="blue", label=r"$\left\langle \frac{C^{EE, F}_{\ell}}{C^{EE}_{\ell}} \right\rangle$")
a0.legend(fontsize=22)
a0.set_ylabel(r"$K_{\ell}$", fontsize=25)
#plt.subplot(2,1,2)
a1.set_ylim(0.98, 1.02)
a1.plot(lb, lb * 0 + 1, color="black")
a1.plot(lb, kf_tf_mc["TT"]/kf_tf_analytic, color="red", label="TT")
a1.plot(lb, kf_tf_mc["EE"]/kf_tf_analytic, color="blue", label="EE")
a1.set_xlabel(r"$\ell$", fontsize=25)
a1.set_ylabel(r"$K^{\rm sims}_{\ell}/K^{\rm analytic}_{\ell}$", fontsize=25)
plt.subplots_adjust(wspace=0, hspace=0)
a1.set_yticks([0.98, 0.99, 1, 1.01])
plt.savefig(f"kspace_tf.png", bbox_inches='tight', dpi=300)
plt.clf()
plt.close()



