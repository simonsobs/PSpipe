from pspy import so_spectra, so_cov, pspy_utils
import pylab as plt, numpy as np
from cobaya.run import run

def get_spectra_and_std(my_spec, name, test):
    lb, ps = so_spectra.read_ps("spectra_%s/spectra/Dl_%s_cross.dat" % (my_spec, name), spectra=spectra)
    lb, noise = so_spectra.read_ps("spectra_%s/spectra/Dl_%s_noise.dat" % (my_spec, name), spectra=spectra)

    cov = np.load("covariances_%s/covariances/analytic_cov_%s_%s.npy" % (my_spec, name, name))
    cov = so_cov.selectblock(cov,
                            ["TT", "TE", "ET", "EE"],
                            n_bins = len(lb),
                            block="%s%s" % (test, test))
    std = np.sqrt(cov.diagonal())
    
    return lb, ps[test], noise[test], cov, std



spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
lmax = 7000
field = "TT"
nSims = 79

for my_spec in ["dr6_pa4_f150", "dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150",  "dr6_pa6_f090", "dr6_pa6_f150"]:

    lb, ps, nb, cov, std = get_spectra_and_std(my_spec, "%sx%s" % (my_spec, my_spec), field)
    lb, ps_uncorr, nb_uncorr, cov_uncorr, std_uncorr = get_spectra_and_std(my_spec, "%s_uncorrx%s_uncorr" % (my_spec, my_spec), field)
    lb, ps_cross, nb_cross, cov_cross, std_cross= get_spectra_and_std(my_spec, "%sx%s_uncorr" % (my_spec, my_spec), field)

    TF1 = ps/ps_uncorr
    TF2 = (ps/ps_cross)

    ps_sim_list = []
    ps_sim_uncorr_list = []
    ps_sim_cross_list = []
    ratio1_list = []
    ratio2_list = []
    
    for iii in range(nSims):
        lb, ps_sim = so_spectra.read_ps("sim_spectra_%s/sim_spectra/Dl_%sx%s_cross_%05d.dat" % (my_spec, my_spec, my_spec, iii), spectra=spectra)
        lb, ps_sim_uncorr = so_spectra.read_ps("sim_spectra_%s/sim_spectra/Dl_%s_uncorrx%s_uncorr_cross_%05d.dat" % (my_spec, my_spec, my_spec, iii), spectra=spectra)
        lb, ps_sim_cross = so_spectra.read_ps("sim_spectra_%s/sim_spectra/Dl_%sx%s_uncorr_cross_%05d.dat" % (my_spec, my_spec, my_spec, iii), spectra=spectra)

        ps_sim_list += [ps_sim[field]]
        ps_sim_uncorr_list += [ps_sim_uncorr[field]]
        ps_sim_cross_list += [ps_sim_cross[field]]
        ratio1_list += [ps_sim[field]/ps_sim_uncorr[field]]
        ratio2_list += [(ps_sim[field]/ps_sim_cross[field])]

    ps_std = np.std(ps_sim_list, axis=0)
    ps_uncorr_std = np.std(ps_sim_uncorr_list, axis=0)
    ps_cross_std = np.std(ps_sim_cross_list, axis=0)
    std_TF1 = np.std(ratio1_list, axis=0)
    std_TF2 = np.std(ratio2_list, axis=0)
    
    np.savetxt("results/TF_%s.dat" % my_spec, np.transpose([lb, TF1, std_TF1]))
    np.savetxt("results/TF_%s_cross.dat" % my_spec, np.transpose([lb, TF2, std_TF2]))



