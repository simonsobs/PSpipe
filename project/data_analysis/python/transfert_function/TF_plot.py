from pspy import so_spectra, so_cov, pspy_utils
import pylab as plt, numpy as np
from cobaya.run import run


freq = [90, 150, 220]
spec_in_freq = {}
spec_in_freq[90] = ["dr6_pa5_f090", "dr6_pa6_f090"]
spec_in_freq[150] = ["dr6_pa4_f150", "dr6_pa5_f150", "dr6_pa6_f150"]
spec_in_freq[220] = ["dr6_pa4_f220"]

ylmin = {}
ylmin[90] = [0.8, 1.02]
ylmin[150] = [0.2, 1.2]
ylmin[220] = [-1, 2]


for f in freq:
    for spec in spec_in_freq[f]:
        lb, TF1, std_TF1 = np.loadtxt("results/TF_%s.dat" % spec, unpack=True)
        lb, TF2, std_TF2 = np.loadtxt("results/TF_%s_cross.dat" % spec, unpack=True)
        plt.errorbar(lb, TF2, std_TF2, label=spec, fmt=".")
        plt.errorbar(lb, TF1, std_TF1, label=spec)

    plt.legend()
    plt.xlim(0, 3000)
    plt.ylim(ylmin[f])

    plt.show()
    
