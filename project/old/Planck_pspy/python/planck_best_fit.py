import numpy as np, pylab as plt
from pspy import so_dict, pspy_utils
from itertools import combinations_with_replacement as cwr
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

bestfit_dir = "best_fits"
plot_dir = "plots"

pspy_utils.create_directory(bestfit_dir)
pspy_utils.create_directory(plot_dir)

freqs = d["freqs"]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

freq_pairs=[]
for cross in cwr(freqs, 2):
    freq_pairs+=[[cross[0],cross[1]]]

clth = {}
lth, clth["TT"], clth["EE"], clth["BB"], clth["TE"] = np.loadtxt(d["theoryfile"], unpack=True)

spec_list = ["TT_100x100", "TT_143x143", "TT_143x217", "TT_217x217",
             "EE_100x100", "EE_100x143", "EE_100x217", "EE_143x143", "EE_143x217", "EE_217x217",
             "TE_100x100", "TE_100x143", "TE_100x217", "TE_143x143", "TE_143x217", "TE_217x217"]

data = np.loadtxt(d["fg_and_syst"])
lth = data[:,0]
fg = {}
for sid, spec in enumerate(spec_list):
    fg[spec] = data[:, sid+1]

fg["TT_100x143"] = 0 * np.sqrt(fg["TT_100x100"] * fg["TT_143x143"])
fg["TT_100x217"] = 0 * np.sqrt(fg["TT_100x100"] * fg["TT_217x217"])

# The foreground+syst spectra have a very strange shape at high ell, we therefore regularize them
# Note that the scales beyond the regularisation scale are not used in the paper

lth_max = 6000
l_regul_dict = {}
l_regul_dict["100x100"] = 1450
l_regul_dict["100x143"] = 1450
l_regul_dict["100x217"] = 1450
l_regul_dict["143x143"] = 2000
l_regul_dict["143x217"] = 2000
l_regul_dict["217x217"] = 2000

fg_regularised = np.zeros(lth_max)

lth = np.arange(2, lth_max + 2)
lth_padded = np.arange(0, lth_max)

fth = lth * (lth + 1) / (2 * np.pi)
fth_padded = lth_padded * (lth_padded + 1) / (2 * np.pi)

cl_th_and_fg = {}

for spec in ["TT", "EE", "TE"]:
    for f in freq_pairs:
        
        f0, f1 = f[0], f[1]
        fname = "%sx%s" % (f0, f1)
        
        l_regul = l_regul_dict[fname]

        fg_regularised[:l_regul] = fg["%s_%s" % (spec,fname)][:l_regul] / fth[:l_regul]
        fg_regularised[l_regul:] = fg_regularised[l_regul - 1]
        
        cl_th_and_fg["%s_%s" % (spec,fname)] = np.zeros(lth_max)
        cl_th_and_fg["%s_%s" % (spec,fname)][2:lth_max] = clth[spec][:lth_max-2] / fth[:lth_max-2] + fg_regularised[:lth_max-2]
        
        np.savetxt("%s/best_fit_%s_%s.dat" % (bestfit_dir, fname, spec), np.transpose([lth_padded, cl_th_and_fg["%s_%s" % (spec,fname)]]))
        
        plt.plot(lth_padded, cl_th_and_fg["%s_%s" % (spec,fname)] * fth_padded, label="%s" % fname)
    
    plt.savefig("%s/best_fit_%s.png" % (plot_dir,spec))
    plt.clf()
    plt.close()

for f in freq_pairs:
    f0, f1 = f[0], f[1]
    fname = "%sx%s" % (f0, f1)
    
    cl_th_and_fg["ET_%s" % fname] = cl_th_and_fg["TE_%s" % fname]
    cl_th_and_fg["EB_%s" % fname] = cl_th_and_fg["TE_%s" % fname]*0
    cl_th_and_fg["BE_%s" % fname] = cl_th_and_fg["TE_%s" % fname]*0
    cl_th_and_fg["TB_%s" % fname] = cl_th_and_fg["TE_%s" % fname]*0
    cl_th_and_fg["BT_%s" % fname] = cl_th_and_fg["TE_%s" % fname]*0
    cl_th_and_fg["BB_%s" % fname] = np.zeros(lth_max)
    cl_th_and_fg["BB_%s" % fname][2:lth_max] = clth["BB"][:lth_max - 2] / fth[:lth_max - 2]
    
    for spec in spectra:
        fname_revert = "%sx%s" % (f1, f0)
        cl_th_and_fg["%s_%s" % (spec, fname_revert)] = cl_th_and_fg["%s_%s" % (spec, fname)]

bestfit_mat = np.zeros((9, 9, lth_max))
fields = ["T", "E", "B"]
for f1, freq1 in enumerate(freqs):
    for f2, freq2 in enumerate(freqs):
        for c1, field1 in enumerate(fields):
            for c2, field2 in enumerate(fields):

                fname = "%sx%s" % (freq1, freq2)
                print (c1 + 3 * f1, c2 + 3 * f2, field1 + field2, fname)
                bestfit_mat[c1 + 3 * f1, c2 + 3 * f2, :lth_max] = cl_th_and_fg["%s_%s" % (field1 + field2, fname)]

np.save("%s/bestfit_matrix.npy" % bestfit_dir, bestfit_mat)
