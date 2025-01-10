import matplotlib
matplotlib.use("Agg")
from pspy import so_dict, pspy_utils
import maps_to_params_utils
import numpy as np
import pylab as plt
import sys, os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

cov_dir = "covariances"
mc_dir = "montecarlo"
cov_plot_dir = "plots/full_covariance"

pspy_utils.create_directory(cov_plot_dir)

experiments = d["experiments"]
lmax = d["lmax"]
binning_file = d["binning_file"]
multistep_path = d["multistep_path"]


bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
nbins = len(bin_hi)

spec_name = []

for id_exp1, exp1 in enumerate(experiments):
    freqs1 = d["freqs_%s" % exp1]
    for id_f1, f1 in enumerate(freqs1):
        for id_exp2, exp2 in enumerate(experiments):
            freqs2 = d["freqs_%s" % exp2]
            for id_f2, f2 in enumerate(freqs2):
                if  (id_exp1 == id_exp2) & (id_f1 >id_f2) : continue
                if  (id_exp1 > id_exp2) : continue
                spec_name += ["%s_%sx%s_%s" % (exp1, f1, exp2, f2)]

analytic_dict= {}
spectra = ["TT", "TE", "ET", "EE"]

nspec = len(spec_name)

for sid1, name1 in enumerate(spec_name):
    for sid2, name2 in enumerate(spec_name):
        if sid1 > sid2: continue
        print (name1, name2)
        na, nb = name1.split("x")
        nc, nd = name2.split("x")

        analytic_cov = np.load("%s/analytic_cov_%sx%s_%sx%s.npy" % (cov_dir, na, nb, nc, nd))
        
        for s1, spec1 in enumerate(spectra):
            for s2, spec2 in enumerate(spectra):
                
                sub_cov = analytic_cov[s1 * nbins:(s1 + 1) * nbins, s2 * nbins:(s2 + 1) * nbins]
                analytic_dict[sid1, sid2, s1, s2] = sub_cov

full_analytic_cov = np.zeros((4 * nspec * nbins, 4 * nspec * nbins))

for sid1, name1 in enumerate(spec_name):
    for sid2, name2 in enumerate(spec_name):
        if sid1 > sid2: continue
        na, nb = name1.split("x")
        nc, nd = name2.split("x")

        for s1, spec1 in enumerate(spectra):
            for s2, spec2 in enumerate(spectra):

                id_start_1 = sid1 * nbins + s1 * nspec * nbins
                id_stop_1 = (sid1 + 1) * nbins + s1 * nspec * nbins
                id_start_2 = sid2 * nbins + s2 * nspec * nbins
                id_stop_2 = (sid2 + 1) * nbins + s2 * nspec * nbins
                full_analytic_cov[id_start_1:id_stop_1, id_start_2: id_stop_2] = analytic_dict[sid1, sid2, s1, s2]

transpose = full_analytic_cov.copy().T
transpose[full_analytic_cov != 0] = 0
full_analytic_cov += transpose

np.save("%s/full_analytic_cov.npy"%cov_dir, full_analytic_cov)

block_to_delete = []
for sid, name in enumerate(spec_name):
    na, nb = name.split("x")
    for s, spec in enumerate(spectra):
        id_start = sid * nbins + s * nspec * nbins
        id_stop = (sid + 1) * nbins + s * nspec * nbins
        if (na == nb) & (spec == 'ET'):
            block_to_delete = np.append(block_to_delete, np.arange(id_start, id_stop))

full_analytic_cov = np.delete(full_analytic_cov, block_to_delete.astype(int), axis=1)
full_analytic_cov = np.delete(full_analytic_cov, block_to_delete.astype(int), axis=0)

np.save("%s/truncated_analytic_cov.npy"%cov_dir, full_analytic_cov)

print ("is matrix positive definite:", maps_to_params_utils.is_pos_def(full_analytic_cov))
print ("is matrix symmetric :", maps_to_params_utils.is_symmetric(full_analytic_cov))

size=int(full_analytic_cov.shape[0]/nbins)

full_mc_cov = np.load("%s/cov_restricted_all_cross.npy"%mc_dir)


os.system("cp %s/multistep2.js %s/multistep2.js" % (multistep_path, cov_plot_dir))
file = "%s/covariance.html" % (cov_plot_dir)
g = open(file, mode="w")
g.write('<html>\n')
g.write('<head>\n')
g.write('<title> covariance </title>\n')
g.write('<script src="multistep2.js"></script>\n')
g.write('<script> add_step("sub",  ["c","v"]) </script> \n')
g.write('<style> \n')
g.write('body { text-align: center; } \n')
g.write('img { width: 100%; max-width: 1200px; } \n')
g.write('</style> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write('<div class=sub>\n')


count=0
for ispec in range(-size+1, size):
    
    rows, cols = np.indices(full_mc_cov.shape)
    row_vals = np.diag(rows, k=ispec*nbins)
    col_vals = np.diag(cols, k=ispec*nbins)
    mat = np.ones(full_mc_cov.shape)
    mat[row_vals, col_vals] = 0
    
    str = "cov_diagonal_%03d.png" % (count)

    plt.figure(figsize=(12,8))
    plt.subplot(1,2,1)
    plt.plot(np.log(np.abs(full_analytic_cov.diagonal(ispec*nbins))))
    plt.plot(np.log(np.abs(full_mc_cov.diagonal(ispec*nbins))), '.')
    plt.legend()
    plt.subplot(1,2,2)
    plt.imshow(np.log(np.abs(full_analytic_cov*mat)))
    plt.savefig("%s/%s"%(cov_plot_dir,str))
    plt.clf()
    plt.close()
    
    g.write('<div class=sub>\n')
    g.write('<img src="'+str+'"  /> \n')
    g.write('</div>\n')
    
    count+=1

g.write('</body> \n')
g.write('</html> \n')
g.close()


