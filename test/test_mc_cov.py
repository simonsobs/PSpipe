#--
# test/test_compare_len_unlen_cov.py
#--
# this test script compares the covariance of lensed CMB to that of Gaussian random field that matching lensed CMB spectra
#

from pspy import so_map, so_window, pspy_utils, so_mpi, so_mcm, so_config, so_spectra, sph_tools
from pixell import curvedsky, utils, enmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()
sns.palplot(sns.color_palette("colorblind"))

# create output directory
output_dir     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_mc_cov')
mcm_prefix     = os.path.join(output_dir, "mcm") 
output_path    = lambda x: os.path.join(output_dir, x) 
alm_input_dir  = '/global/cscratch1/sd/engelen/simsS1516_v0.4/data'
alm_input_temp = os.path.join(alm_input_dir, 'fullskyLensedUnabberatedCMB_alm_set01_{0:05}.fits') 

if so_mpi.rank == 0:
    pspy_utils.create_directory(output_dir)
else: pass
so_mpi.barrier()

# set up the test
ncomp                = 3
nsims                = 256
lmax                 = 5000
ra0, ra1, dec0, dec1 = (-10., 10., -10., 10.) # ra/dec in degress
res                  = 1.                     # res in arcmin
binning_file         = os.path.join(so_config.DEFAULT_DATA_DIR, 'binningFile_100_50.dat')
spin_pairs           = ['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']
spectra              = ['TT','TE','TB','ET','BT','EE','EB','BE','BB']
theory_file          = os.path.join(so_config.DEFAULT_DATA_DIR, 'cosmo2017_10K_acc3_lensedCls.dat')

window   = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
window   = so_window.create_apodization(window, 'Rectangle', 2)
window.plot(file_name=output_path('window'))

# helper function
def stats(data, axis=0, ddof=1.):
    datasize = data.shape
    mean     = np.mean(data, axis=axis)
    cov      = np.cov(data.transpose(), ddof=ddof)
    corrcoef = np.corrcoef(data.transpose())
    std      = np.std(data, axis=axis, ddof=ddof) # use the N-1 normalization
    std_mean = std/np.sqrt(datasize[0])

    return {'mean': mean, 'cov': cov, 'corrcoef': corrcoef, 'std': std, 'datasize': datasize, 'std_mean': std_mean}

# create mcm
if so_mpi.rank == 0 and not os.path.exists('{}_Bbl_spin0xspin0.npy'.format(mcm_prefix)):
    so_mcm.mcm_and_bbl_spin0and2((window, window), binning_file=binning_file, lmax=lmax, save_file=mcm_dir)
else: pass
so_mpi.barrier()

mbb_inv, Bbl = so_mcm.read_coupling(prefix=mcm_prefix, spin_pairs=spin_pairs)

# split jobs over nodes
so_mpi.init(True)
subtasks = so_mpi.taskrange(nsims-1)

template = so_map.car_template(3, ra0, ra1, dec0, dec1, res)
# use precomputed alm for lensed cmb to save times 
for sim_idx in subtasks:
    if os.path.exists(output_path('spectra_lensed_ncomp%d_%04d.dat'%(ncomp,sim_idx))): continue
    alms  = so_map.read_alm(alm_input_temp.format(sim_idx), ncomp)
    somap = so_map.alm2map(alms, template.copy())
    if sim_idx == 0:
        somap.plot(file_name=output_path('lensed'))
    print ('sim number %04d'%sim_idx)
    
    # alms from windowed patches
    alms   = sph_tools.get_alms(somap, (window, window), 0 ,lmax)
    ls, ps = so_spectra.get_spectra(alms, alms, spectra=spectra)
    lb, Db = so_spectra.bin_spectra(ls, ps, binning_file, type='Dl', lmax=lmax, mbb_inv=mbb_inv,spectra=spectra)
    so_spectra.write_ps(output_path('spectra_lensed_ncomp%d_%04d.dat'%(ncomp,sim_idx)),lb,Db,'Dl',spectra=spectra)

# generate GRF to save times 
for sim_idx in subtasks:
    if os.path.exists(output_path('spectra_grf_ncomp%d_%04d.dat'%(ncomp,sim_idx))): continue
    somap = template.copy()
    somap = somap.synfast(theory_file)
    if sim_idx == 0:
        somap.plot(file_name=output_path('grf'))
    print ('sim number %04d'%sim_idx)
    
    # alms from windowed patches
    alms   = sph_tools.get_alms(somap, (window, window), 0 ,lmax)
    ls, ps = so_spectra.get_spectra(alms, alms, spectra=spectra)
    lb, Db = so_spectra.bin_spectra(ls, ps, binning_file, type='Dl', lmax=lmax, mbb_inv=mbb_inv,spectra=spectra)
    so_spectra.write_ps(output_path('spectra_grf_ncomp%d_%04d.dat'%(ncomp,sim_idx)),lb,Db,'Dl',spectra=spectra)

if so_mpi.rank == 0:
    Db_dict    = {}
    st         = {}
    data_types = ['lensed', 'grf']
    sim_idxes  = range(0, nsims) 
    for data_type in data_types:
        Db_dict[data_type] = {}
        for spec in spectra:
            Db_dict[data_type][spec] ={}
        for sim_idx in sim_idxes:
            lb,Db=so_spectra.read_ps(output_path('spectra_%s_ncomp%d_%04d.dat'%(data_type,ncomp,sim_idx)),spectra=spectra)
            if not Db_dict[data_type].has_key('lbin'): Db_dict[data_type]['lbin'] = lb
            for spec in spectra:
                Db_dict[data_type][spec][sim_idx] = Db[spec]

        st[data_type] = {}
        nspec    = len(spectra)
        nbin     = len(Db_dict[data_type]['lbin'])
        data = np.zeros((nsims, nbin*nspec))
        ctr  = 0
        for ctr, spec in enumerate(spectra):
            data[:,ctr*nbin:(ctr+1)*nbin] = np.array(Db_dict[data_type][spectra[ctr]].values()).copy()
            st[data_type][spec]     = stats(np.array(Db_dict[data_type][spectra[ctr]].values()).copy())
        st[data_type]['full'] = stats(data)
    
    # plot spectra
    plt.figure(figsize=(20,15))
    for data_type in data_types:
        for c,spec in enumerate(spectra):
            mean=st[data_type][spec]['mean']
            std =st[data_type][spec]['std_mean']
            plt.subplot(3,3,c+1)
            plt.errorbar(lb,mean,std,fmt='o',label='%s'%data_type)
            plt.ylabel(r'$D^{%s}_{\ell}$'%spec,fontsize=20)
            plt.xlabel(r'$\ell$',fontsize=20)
            if c==0:
                plt.legend()
    plt.savefig(output_path('spectra.png'),bbox_inches='tight')
    plt.clf()
    plt.close()

    for data_type in data_types:
        for spec in spectra:
            std  = np.sqrt(np.diag(st[data_type][spec]['cov']))
            corr = st[data_type][spec]['cov']/np.outer(std,std)
            np.fill_diagonal(corr, 0.)
            ax = sns.heatmap(corr, xticklabels=10, yticklabels=10)
            plt.title('%s%s correlation matrix (%s)' %(spec.upper(), spec.upper(), data_type.upper()))
            plt.savefig(output_path('correlation_%s_%s.png'%(data_type, spec)), bbox_inches='tight')
            plt.clf()
            plt.close()
 

