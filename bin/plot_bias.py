#!/usr/bin/env python

from __future__ import print_function
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

result_dir=d['result_dir']

l,ps_theory=pspy_utils.ps_lensed_theory_to_dict(d['clfile'],d['type'],lmax=d['lmax'])

if d['spin']=='0-2':
    ncomp=3
    spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    spin_pairs=['spin0xspin0','spin0xspin2','spin2xspin0','spin2xspin2']
    mbb_inv,Bbl=so_mcm.read_coupling(prefix='%s/test'%result_dir,spin_pairs=spin_pairs)
    ps_theory_b=so_mcm.apply_Bbl(Bbl,ps_theory,spectra=spectra)

elif d['spin']=='0':
    ncomp=1
    mbb_inv,Bbl=so_mcm.read_coupling(prefix='%s/test'%result_dir)
    spectra=None
    ps_theory_b=so_mcm.apply_Bbl(Bbl,ps_theory['TT'])


num=np.arange(d['iStart'],d['iStop'])
spec_list =  np.genfromtxt('%s/spectra_list.txt'%result_dir,unpack=True,dtype='str')

Db_dict={}
for spec_name in spec_list:
    if spectra is not None:
        for spec in spectra:
            Db_dict[spec_name,spec]=[]
        for iii in num:
            lb,Db=so_spectra.read_ps('%s/spectra_ncomp%d_%s_%04d.dat'%(result_dir,ncomp,spec_name,iii),spectra=spectra)
            for spec in spectra:
                Db_dict[spec_name,spec]+=[Db[spec]]
    else:
        Db_dict[spec_name]=[]
        for iii in num:
            lb,Db=so_spectra.read_ps('%s/spectra_ncomp%d_%s_%04d.dat'%(result_dir,ncomp,spec_name,iii),spectra=spectra)
            Db_dict[spec_name]+=[Db]

for spec_name in spec_list:
    if spectra is not None:
        plt.figure(figsize=(20,15))
        for c,spec in enumerate(spectra):
            mean=np.mean(Db_dict[spec_name,spec],axis=0)
            std=np.std(Db_dict[spec_name,spec],axis=0)
            plt.subplot(3,3,c+1)
            plt.errorbar(lb,mean,std,fmt='o',label='%s'%spec_name)
            plt.plot(lb,ps_theory_b[spec], color='lightblue',label='binned theory')
            plt.plot(l,ps_theory[spec], color='grey',label='theory')
            plt.ylabel(r'$D^{%s}_{\ell}$'%spec,fontsize=20)
            plt.xlabel(r'$\ell$',fontsize=20)
            if c==0:
                plt.legend()
        plt.savefig('%s/spectra_%s.png'%(result_dir,spec_name),bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.figure(figsize=(20,15))
        for c,spec in enumerate(spectra):
            mean=np.mean(Db_dict[spec_name,spec],axis=0)
            std=np.std(Db_dict[spec_name,spec],axis=0)
            plt.subplot(3,3,c+1)
            plt.plot(lb,lb*0)
            plt.errorbar(lb,mean-ps_theory_b[spec],std/np.sqrt(len(num)),fmt='o',label='%s'%spec_name)
            plt.ylabel(r'$D^{%s}_{\ell}-D^{%s,th}_{\ell}$'%(spec,spec),fontsize=20)
            plt.xlabel(r'$\ell$',fontsize=20)
            if c==0:
                plt.legend()
        plt.savefig('%s/spectra_bias_%s.png'%(result_dir,spec_name),bbox_inches='tight')
        plt.clf()
        plt.close()
    else:
        mean=np.mean(Db_dict[spec_name],axis=0)
        std=np.std(Db_dict[spec_name],axis=0)
        
        plt.errorbar(lb,mean,std,fmt='o',label='%s'%spec_name)
        plt.plot(lb,ps_theory_b, color='lightblue',label='binned theory')
        plt.plot(l,ps_theory['TT'], color='grey',label='theory')
        plt.ylabel(r'$D_{\ell}$',fontsize=20)
        plt.xlabel(r'$\ell$',fontsize=20)
        plt.legend()
        plt.savefig('%s/spectra_%s.png'%(result_dir,spec_name),bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.errorbar(lb,mean-ps_theory_b,std/np.sqrt(len(num)),fmt='o',label='%s'%spec_name)
        plt.plot(lb,lb*0)
        plt.ylabel(r'$D_{\ell}-D^{th}_{\ell}$',fontsize=20)
        plt.xlabel(r'$\ell$',fontsize=20)
        plt.legend()
        plt.savefig('%s/spectra_bias_%s.png'%(result_dir,spec_name),bbox_inches='tight')
        plt.clf()
        plt.close()
