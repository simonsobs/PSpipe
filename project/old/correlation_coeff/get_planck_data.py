"""
This script is used to download the public planck data
To run it: python get_planck_data.py global.dict
It will download maps, likelihood masks and beams of planck
It will also generate a binning file suitable for the estimation of the cross correlation coefficient"
"""

import numpy as np,healpy as hp,pylab as plt
from pspy import pspy_utils,so_dict
import os,sys
import wget
import tarfile
import astropy.io.fits as fits

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

# You have to spefify the data directory in which the products will be downloaded
data_dir=d['data_dir']
freqs = d['freqs']

pspy_utils.create_directory(data_dir)

# Choose what you want to download, if this is your first try, all of this should be set to True
download_maps=True
download_likelihood_mask=True
download_beams=True

if download_maps==True:
    print ('Download Planck data maps')
    maps_dir=data_dir+'/maps'
    pspy_utils.create_directory(maps_dir)
    # Planck keep inconsistent notation for the halfmission, sometimes 'halfmission-1' sometimes 'hm1'
    splits = ['halfmission-1','halfmission-2' ]
    for hm in splits:
        for f in freqs:
            url='http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_SkyMap_%s_2048_R3.01_%s.fits'%(f,hm)
            print (url)
            wget.download(url, '%s/HFI_SkyMap_%s_2048_R3.01_%s.fits'%(maps_dir,f,hm))

if download_likelihood_mask==True:
    print ('Download Planck likelihood mask')
    likelihood_mask_dir=data_dir+'/likelihood_mask'
    pspy_utils.create_directory(likelihood_mask_dir)
    splits = ['hm1','hm2' ]
    for hm in splits:
        for f in freqs:
            
            url='http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_Mask_Likelihood-temperature-%s-%s_2048_R3.00.fits'%(f,hm)
            wget.download(url, '%s/COM_Mask_Likelihood-temperature-%s-%s_2048_R3.00.fits'%(likelihood_mask_dir,f,hm))
            url='http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=COM_Mask_Likelihood-polarization-%s-%s_2048_R3.00.fits'%(f,hm)
            wget.download(url, '%s/COM_Mask_Likelihood-polarization-%s-%s_2048_R3.00.fits'%(likelihood_mask_dir,f,hm))

if download_beams==True:
    print ('Download Planck beams')
    beam_dir=data_dir+'/beams'
    pspy_utils.create_directory(beam_dir)
    url='http://pla.esac.esa.int/pla/aio/product-action?DOCUMENT.DOCUMENT_ID=HFI_RIMO_BEAMS_R3.01.tar.gz'
    wget.download(url, '%s/HFI_RIMO_BEAMS_R3.01.tar.gz'%beam_dir)
    tf = tarfile.open('%s/HFI_RIMO_BEAMS_R3.01.tar.gz'%beam_dir)
    tf.extractall(beam_dir)
    tf.close()

    spectra=['TT','EE','BB','TE']
    leakage_term={}
    for spec in spectra:
        leakage_term[spec]=['%s_2_TT'%spec, '%s_2_EE'%spec,'%s_2_BB'%spec,'%s_2_TE'%spec, '%s_2_TB'%spec,'%s_2_EB'%spec,'%s_2_ET'%spec,'%s_2_BT'%spec,'%s_2_BE'%spec]

    splits = ['hm1','hm2' ]
    my_lmax=6000
    for hm in splits:
        for f in freqs:
            Wl=fits.open('%s/BeamWf_HFI_R3.01/Wl_R3.01_plikmask_%s%sx%s%s.fits'%(beam_dir,f,hm,f,hm))
            Wl_dict={}
            num=1
            for spec in spectra:
                for leak in leakage_term[spec]:
                    Wl_dict[leak]=Wl[num].data[leak]
                num+=1

            lmax=len(Wl_dict['TT_2_TT'][0])

            bl_T=np.zeros(my_lmax)
            bl_pol=np.zeros(my_lmax)

            # We will call Planck beam the sqrt or the XX_2_XX term of the beam leakage matrix
            bl_T[:lmax]= np.sqrt(Wl_dict['TT_2_TT'][0])
            bl_pol[:lmax]= np.sqrt(Wl_dict['EE_2_EE'][0])

            # Here we 'extrapolate' slighty the Planck beam, just repeating the same value after l=max(l) in Planck
            # In practice this won't be used in the analysis
            bl_T[lmax:]=bl_T[lmax-1]
            bl_pol[lmax:]=bl_pol[lmax-1]

            l=np.arange(my_lmax)
            np.savetxt('%s/beam_T_%s_%s.dat'%(beam_dir,f,hm), np.transpose([l,bl_T]))
            np.savetxt('%s/beam_pol_%s_%s.dat'%(beam_dir,f,hm), np.transpose([l,bl_pol]))
    
# Create binning file, ensuring a decent signal to noise per bin for the Planck data
binning_dir=data_dir+'/binning'
pspy_utils.create_directory(binning_dir)
bin_size=np.zeros(100)
bin_size[0]=20
bin_size[1:8]=40
bin_size[8:42]=20
bin_size[42:46]=80
bin_size[46:48]=100
bin_size[48:]=150

bin_min=2

g = open('%s/binning.dat'%binning_dir,mode="w")
for i in range(100):
    bin_max=bin_min+bin_size[i]
    bin_mean=(bin_min+bin_max)/2
    g.write("%f %f %f\n"%(bin_min,bin_max,bin_mean))
    bin_min+=bin_size[i]+1
g.close()




