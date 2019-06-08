import matplotlib
matplotlib.use('Agg')
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

content=d['content']

plot_dir='plot'
combined_map_dir='combined_map'

pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(combined_map_dir)

experiment=d['experiment']


color_range=(200,20,20)

for exp in experiment:
    freqs=d['freq_%s'%exp]
    for count,freq in enumerate(freqs):
        map_all=so_map.healpix_template(ncomp=3,nside=4096)
        
        for cont in content:
            
            maps_list= d['%s_maps'%cont]
            map=maps_list[count]
            
            map=so_map.read_map(map)
            if (len(map.data.shape)==1):
                map_all.data[0]+=map.data[0]
            else:
                for i in range(3):
                    map_all.data[i]+=map.data[i]

        noise0_list= d['noise_maps0']
        noise1_list= d['noise_maps1']
        
        noise_map0=so_map.read_map(noise0_list[count])
        noise_map1=so_map.read_map(noise1_list[count])

        noise_map0.data/=np.sqrt(2)
        noise_map1.data/=np.sqrt(2)

        survey_mask_list= d['survey_masks']
        survey_mask=so_map.read_map(survey_mask_list[count])
        
        color_range=(200,20,20)

                
        map_all.data*=survey_mask.data
        map_all.plot(file_name='%s/combined_%s_%s'%(plot_dir,exp,freq),color_range=color_range)
        map_all.write_map('%s/combined_map_%s.fits'%(combined_map_dir,freq))
        
        color_range=(400,40,40)

        
        noise_map0.data+=map_all.data
        noise_map0.data*=survey_mask.data
        noise_map0.plot(file_name='%s/combined_%s_%s_noise0'%(plot_dir,exp,freq),color_range=color_range)
        noise_map0.write_map('%s/combined_map_%s_noise0.fits'%(combined_map_dir,freq))
        
        noise_map1.data+=map_all.data
        noise_map1.data*=survey_mask.data
        noise_map1.plot(file_name='%s/combined_%s_%s_noise1'%(plot_dir,exp,freq),color_range=color_range)
        noise_map1.write_map('%s/combined_map_%s_noise1.fits'%(combined_map_dir,freq))
