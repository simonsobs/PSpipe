"""
These are tests of analytic estimation of a gaussian covariance matrix for spin0 field
It is done in CAR pixellisation.
We also provide an option for monte-carlo verification of the covariance matrix
"""
from __future__ import print_function
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_cov
import healpy as hp, numpy as np, pylab as plt
import os,time

#We start by specifying the CAR survey parameters, it will go from ra0 to ra1 and from dec0 to dec1 (all in degrees)
# It will have a resolution of 1 arcminute
ra0,ra1,dec0,dec1=-10,10,-10,10
res=2.5
ncomp=1
# clfile are the camb lensed power spectra
clfile='../data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat'
# nSplits stand for the number of splits we want to simulate
nSplits=2
# The type of power spectra we want to compute
type='Dl'
# a binningfile with format, lmin,lmax,lmean
binning_file='../data/BIN_ACTPOL_50_4_SC_low_ell_startAt2'
# the maximum multipole to consider
lmax=2000
# the number of iteration in map2alm
niter=0
# the noise on the spin0 component
rms_uKarcmin_T=1
# the apodisation lengh for the survey mask (in degree)
apo_radius_degree_survey=1
# the number of holes in the point source mask
source_mask_nholes=10
# the radius of the holes (in arcminutes)
source_mask_radius=10
# the apodisation lengh for the point source mask (in degree)
apo_radius_degree_mask=0.3
# for the CAR survey we will use an apodisation type designed for rectangle maps
apo_type='Rectangle'
# parameter for the monte-carlo simulation
DoMonteCarlo=True
n_sims=10

test_dir='result_cov_spin0'
try:
    os.makedirs(test_dir)
except:
    pass

template= so_map.car_template(ncomp,ra0,ra1,dec0,dec1,res)
# the binary template for the window functionpixels
# for CAR we set pixels inside the survey at 1 and  at the border to be zero
binary=so_map.car_template(1,ra0,ra1,dec0,dec1,res)
binary.data[:]=0
binary.data[1:-1,1:-1]=1

#we then apodize the survey mask
window=so_window.create_apodization(binary, apo_type=apo_type, apo_radius_degree=apo_radius_degree_survey)
#we create a point source mask
mask=so_map.simulate_source_mask(binary, nholes=source_mask_nholes, hole_radius_arcmin=source_mask_radius)
#... and we apodize it
mask= so_window.create_apodization(mask, apo_type='C1', apo_radius_degree=apo_radius_degree_mask)
#the window is given by the product of the survey window and the mask window
window.data*=mask.data
#the window is going to couple mode together, we compute a mode coupling matrix in order to undo this effect
mbb_inv,Bbl=so_mcm.mcm_and_bbl_spin0(window, binning_file, lmax=lmax, type='Dl')

l_th,ps_theory=pspy_utils.ps_lensed_theory_to_dict(clfile,type,lmax=lmax)
nl_th=pspy_utils.get_nlth_dict(rms_uKarcmin_T,type,lmax)

survey_id= ['Ta','Tb','Tc','Td']
survey_name=['split_0','split_1','split_0','split_1']

Clth_dict={}
for name1,id1 in zip(survey_name,survey_id):
    for name2,id2 in zip(survey_name,survey_id):
        spec=id1[0]+id2[0]
        Clth_dict[id1+id2]=ps_theory[spec]+nl_th[spec]*so_cov.delta(name1,name2)

coupling_dict=so_cov.cov_coupling_spin0(window, lmax, niter=0)
analytic_cov=so_cov.cov_spin0(Clth_dict,coupling_dict,binning_file,lmax,mbb_inv,mbb_inv)

if DoMonteCarlo==True:
    
    Db_list={}
    cov={}
    nameList=[]
    specList=[]
    for i in range(nSplits):
        nameList+=['split_%d'%i]
    for c1,name1 in  enumerate(nameList):
        for c2,name2 in enumerate(nameList):
            if c1>c2: continue
            spec_name='%sx%s'%(name1,name2)
            specList+=[spec_name]
            Db_list[spec_name]=[]

    for iii in range(n_sims):
        t=time.time()
        cmb=template.synfast(clfile)
        splitlist=[]
        for i in range(nSplits):
            split=cmb.copy()
            noise = so_map.white_noise(split,rms_uKarcmin_T=rms_uKarcmin_T)
            split.data+=noise.data
            splitlist+=[split]

        almList=[]
        for s in splitlist:
            almList+=[ sph_tools.get_alms(s,window,niter,lmax) ]

        for name1, alm1, c1  in zip(nameList,almList,np.arange(nSplits)):
            for name2, alm2, c2  in zip(nameList,almList,np.arange(nSplits)):
                if c1>c2: continue
                ls,ps= so_spectra.get_spectra(alm1,alm2)
                spec_name='%sx%s'%(name1,name2)
                lb,Db=so_spectra.bin_spectra(ls,ps,binning_file,lmax,type=type,mbb_inv=mbb_inv)
                Db_list[spec_name]+=[Db]

        print ('sim number %04d took %s second to compute'%(iii,time.time()-t))

    for spec1 in specList:
        for spec2 in specList:
            cov[spec1,spec2]=0
            for iii in range(n_sims):
                cov[spec1,spec2]+=np.outer(Db_list[spec1][iii],Db_list[spec2][iii])
            print (spec1,spec2)
            cov[spec1,spec2]=cov[spec1,spec2]/n_sims-np.outer(np.mean(Db_list[spec1],axis=0), np.mean(Db_list[spec2],axis=0))

    cov=cov['%sx%s'%(survey_name[0],survey_name[1]),'%sx%s'%(survey_name[2],survey_name[3])]

    var = cov.diagonal()
    analytic_var = analytic_cov.diagonal()

    plt.semilogy()
    plt.plot(lb[1:],var[1:],'o',label='MC ')
    plt.plot(lb[1:],analytic_var[1:],label='Analytic')
    plt.ylabel(r'$\sigma^{2}_{\ell}$',fontsize=22)
    plt.xlabel(r'$\ell$',fontsize=22)
    plt.savefig('%s/variance.png'%(test_dir))
    plt.clf()
    plt.close()

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(so_cov.cov2corr(cov))
    plt.subplot(1,2,2)
    plt.imshow(so_cov.cov2corr(analytic_cov))
    plt.savefig('%s/correlation.png'%(test_dir))
    plt.clf()
    plt.close()

