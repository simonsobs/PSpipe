"""
This script is used to combine cmb with extragalactic and galactic simulations.
It produces a noiseless combination of the signal maps as well as two noisy splits containing both signal and noise.
It also produce a survey mask which is a binary mask where SO observations are defined.
To run it you need to specify a dictionnary file, for example global_combine.dict provided in the:
https://github.com/simonsobs/PSpipe/tree/master/project/AnalyseSims/NERSC_run folder
The code will run as follow:
python combine_sims.py global_combine.dict
"""

import matplotlib
matplotlib.use('Agg')
from pspy import so_map,so_window,so_mcm,sph_tools,so_spectra, pspy_utils, so_dict
import healpy as hp, numpy as np, pylab as plt
import os,sys
import time

# We start by reading the info in the dictionnary
d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

experiment=d['experiment']
content=d['content']
name=d['combinaison_name']

plot_dir='maps_plot'
combined_map_dir='combined_maps'
survey_mask_dir='survey_masks'

# Create three folders, one for the plot of the simulations, one for storing the combined maps and one for the survey masks
pspy_utils.create_directory(plot_dir)
pspy_utils.create_directory(combined_map_dir)
pspy_utils.create_directory(survey_mask_dir)


# We loop on all the different experiments that we want to consider
for exp in experiment:
    # Each experiment could have its associated nside and frequency list
    nside=d['nside_%s'%exp]
    freqs=d['freq_%s'%exp]
    for count,freq in enumerate(freqs):
        #We create a template for each frequency and add all composents present in 'content'
        map_all=so_map.healpix_template(ncomp=3,nside=nside)
        for cont in content:
            maps_list= d['%s_maps'%cont]
            map=maps_list[count]
            
            map=so_map.read_map(map)
            
            #some of the component are I only while other are I,Q,U
            if (len(map.data.shape)==1):
                map_all.data[0]+=map.data
            else:
                for i in range(3):
                    map_all.data[i]+=map.data[i]
    
        # we read two noise maps, since the noise maps represent the noise properties of the full SO survey
        # we multiply them by sqrt(2)
        
        noise0_list= d['noise_maps0']
        noise1_list= d['noise_maps1']
        
        noise_map0=so_map.read_map(noise0_list[count])
        noise_map1=so_map.read_map(noise1_list[count])
        
        noise_map0.data*=np.sqrt(2)
        noise_map1.data*=np.sqrt(2)
        
        # We generate a survey mask, that represent the SO coverage
        survey_mask=so_map.healpix_template(ncomp=1,nside=nside)
        id=np.where(noise_map0.data[0]>-1*10**30)
        survey_mask.data[id]=1
        
        # plot it and write it to disk
        survey_mask.plot(file_name='%s/survey_mask_%s_%s'%(plot_dir,exp,freq))
        survey_mask.write_map('%s/survey_mask_%s_%s.fits'%(survey_mask_dir,exp,freq))

        # and multiply it with the simulation
        map_all.data*=survey_mask.data
        
        # we write the noiseless combined simulation and its plot to disk
        color_range=(200,20,20)
        map_all.plot(file_name='%s/%s_map_%s_%s'%(plot_dir,name,exp,freq),color_range=color_range)
        map_all.write_map('%s/%s_map_%s_%s.fits'%(combined_map_dir,name,exp,freq))

        # we coadd the noise simulations with signal simulation, and write them (and their plots) to disk.

        color_range=(400,40,40)
        
        noise_map0.data+=map_all.data
        noise_map0.data*=survey_mask.data
        noise_map0.plot(file_name='%s/%s_map_%s_%s_noise0'%(plot_dir,name,exp,freq),color_range=color_range)
        noise_map0.write_map('%s/%s_map_%s_%s_noise0.fits'%(combined_map_dir,name,exp,freq))
        
        noise_map1.data+=map_all.data
        noise_map1.data*=survey_mask.data
        noise_map1.plot(file_name='%s/%s_map_%s_%s_noise1'%(plot_dir,name,exp,freq),color_range=color_range)
        noise_map1.write_map('%s/%s_map_%s_%s_noise1.fits'%(combined_map_dir,name,exp,freq))
