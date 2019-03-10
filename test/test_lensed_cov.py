from pspy import so_config, so_cov, so_mpi
import os 
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sns.color_palette(palette='colorblind')

noise_uK_arcmin, fwhm_arcmin, lmin, lmax = (0., 0., 2, 500)

camb_lensed_theory_file   = '../data/cosmo2017_10K_acc3_lensedCls.dat'
camb_unlensed_theory_file = '../data/cosmo2017_10K_acc3_lenspotentialCls.dat'
output_dir                = './test_lensed_cov'

# compute lensed cov
so_cov.calc_cov_lensed(noise_uK_arcmin, fwhm_arcmin, lmin, lmax, camb_lensed_theory_file, camb_unlensed_theory_file, output_dir, overwrite=False)

def bin_matrix(cov, delta_l):
    nbin = int(np.ceil(float(lmax)/delta_l))
    for comb in cov.keys():
        shape = cov[comb].shape
        temp = np.zeros((nbin, nbin))
        for ny in range(nbin):
            for nx in range(nbin):
                lymin = delta_l*ny
                lxmin = delta_l*nx
                lymax = min(delta_l*(ny+1), shape[0]) 
                lxmax = min(delta_l*(nx+1), shape[1]) 
                temp[ny, nx] = cov[comb][lymin:lymax,lxmin:lxmax].mean()
        cov[comb] = temp         


if so_mpi.rank == 0:
    # if you just want the lensing induced part, you set include_gaussian_part=False. For here, I include the guassian part just for the plotting purpose
    include_gaussian_part=True
    # note that lensed cov mat start at ell = 0
    cov = so_cov.load_cov_lensed(os.path.join(output_dir, "covariances_CMBxCMB_v1_%d_pspy.pkl"%lmax ), include_gaussian_part)
    bin_matrix(cov, delta_l=50)
    # bin cov mats
    print("Computed lensed cov for %s"%str(cov.keys()))

    ## plotting correlation matrix
    for comb in cov.keys():
        comb1, comb2 = comb[:2]*2, comb[-2:]*2

        corr = np.nan_to_num(cov[comb].copy())
        corr = np.nan_to_num(corr /np.sqrt(np.outer(np.diag(cov[comb1]),np.diag(cov[comb2]))))

        np.fill_diagonal(corr,0.)
        corr = corr[:-2, :-2]
        vmax = np.max(np.abs(corr))*1.2
        vmin = -vmax

        fig= plt.plot()
        sns.heatmap(corr, vmin=vmin, vmax=vmax)
        plt.title('correlation {}'.format(comb.upper()))
        plt.savefig(os.path.join(output_dir,'corr_{}.png'.format(comb)))
        plt.close()
