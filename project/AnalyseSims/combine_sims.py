import matplotlib
matplotlib.use('Agg')
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

experiment=['la']

freq_la=['027','039','093','145','225','280']

content=['dust','synchrotron','freefree','ame','cib','cmb','ksz','tsz']

dust_dir='/project/projectdirs/sobs/v4_sims/mbs/201904_highres_foregrounds_equatorial/4096/dust/0000/'
ame_dir='/project/projectdirs/sobs/v4_sims/mbs/201904_highres_foregrounds_equatorial/4096/ame/0000/'
freefree_dir='/project/projectdirs/sobs/v4_sims/mbs/201904_highres_foregrounds_equatorial/4096/freefree/0000/'
synchrotron_dir='/project/projectdirs/sobs/v4_sims/mbs/201904_highres_foregrounds_equatorial/4096/synchrotron/0000/'
cib_dir='/project/projectdirs/sobs/v4_sims/mbs/201905_extragalactic/4096/cib/0000/'
cmb_dir='/project/projectdirs/sobs/v4_sims/mbs/201905_extragalactic/4096/cmb/0000/'
cmb_unlensed_dir='/project/projectdirs/sobs/v4_sims/mbs/201905_extragalactic/4096/cmb_unlensed/0000/'
ksz_dir='/project/projectdirs/sobs/v4_sims/mbs/201905_extragalactic/4096/ksz/0000/'
tsz_dir='/project/projectdirs/sobs/v4_sims/mbs/201905_extragalactic/4096/tsz/0000/'
noise_dir0='/project/projectdirs/sobs/v4_sims/mbs/201901_gaussian_fg_lensed_cmb_realistic_noise/4096/noise/0000/'
noise_dir1='/project/projectdirs/sobs/v4_sims/mbs/201901_gaussian_fg_lensed_cmb_realistic_noise/4096/noise/0001/'

dust_maps=[dust_dir+'simonsobs_dust_uKCMB_la%s_nside4096_0000.fits'%f for f in freq_la]
ame_maps=[ame_dir+'simonsobs_ame_uKCMB_la%s_nside4096_0000.fits'%f for f in freq_la]
freefree_maps=[freefree_dir+'simonsobs_freefree_uKCMB_la%s_nside4096_0000.fits'%f for f in freq_la]
synchrotron_maps=[synchrotron_dir+'simonsobs_synchrotron_uKCMB_la%s_nside4096_0000.fits'%f for f in freq_la]
cib_maps=[cib_dir+'simonsobs_cib_uKCMB_la%s_nside4096_0000.fits'%f for f in freq_la]
cmb_maps=[cmb_dir+'simonsobs_cmb_uKCMB_la%s_nside4096_0000.fits'%f for f in freq_la]
cmb_unlensed_maps=[cmb_unlensed_dir+'simonsobs_cmb_unlensed_uKCMB_la%s_nside4096_0000.fits'%f for f in freq_la]
ksz_maps=[ksz_dir+'simonsobs_ksz_uKCMB_la%s_nside4096_0000.fits'%f for f in freq_la]
tsz_maps=[tsz_dir+'simonsobs_tsz_uKCMB_la%s_nside4096_0000.fits'%f for f in freq_la]

noise_maps0=[noise_dir0+'simonsobs_noise_uKCMB_la%s_nside4096_0000.fits'%f for f in freq_la]
noise_maps1=[noise_dir1+'simonsobs_noise_uKCMB_la%s_nside4096_0001.fits'%f for f in freq_la]


plot_dir='plot'
combined_map_dir='combined_maps'

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(combined_map_dir)

color_range=(200,20,20)

for exp in experiment:
    nside=d['nside_%s'%exp]
    
    for count,freq in enumerate(freq_la):
        map_all=so_map.healpix_template(ncomp=3,nside=nside)
        for cont in content:
            maps_list= '%s_maps'
            map=maps_list[count]
            
            map=so_map.read_map(map)
            if (len(map.data.shape)==1):
                map_all.data[0]+=map.data[0]
            else:
                for i in range(3):
                    map_all.data[i]+=map.data[i]

        noise0_list= 'noise_maps0'
        noise1_list= 'noise_maps1'
        
        noise_map0=so_map.read_map(noise0_list[count])
        noise_map1=so_map.read_map(noise1_list[count])

        noise_map0.data/=np.sqrt(2)
        noise_map1.data/=np.sqrt(2)

        survey_mask=so_map.read_map(d['survey_mask_%s_%s'%(exp,freq)])
        
        color_range=(200,20,20)

        map_all.data*=survey_mask.data
        map_all.plot(file_name='%s/combined_%s_%s'%(plot_dir,exp,freq),color_range=color_range)
        map_all.write_map('%s/combined_map_%s_%s.fits'%(combined_map_dir,exp,freq))
        
        color_range=(400,40,40)

        noise_map0.data+=map_all.data
        noise_map0.data*=survey_mask.data
        noise_map0.plot(file_name='%s/combined_%s_%s_noise0'%(plot_dir,exp,freq),color_range=color_range)
        noise_map0.write_map('%s/combined_map_%s_%s_noise0.fits'%(combined_map_dir,exp,freq))
        
        noise_map1.data+=map_all.data
        noise_map1.data*=survey_mask.data
        noise_map1.plot(file_name='%s/combined_%s_%s_noise1'%(plot_dir,exp,freq),color_range=color_range)
        noise_map1.write_map('%s/combined_map_%s_%s_noise1.fits'%(combined_map_dir,exp,freq))
