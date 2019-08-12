import numpy as np
import pylab as plt
from pspy import so_dict,so_map,pspy_utils
import sys
import astropy.io.fits as fits
from matplotlib.pyplot import cm


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

data_dir= d['data_dir']



junk,vec=np.loadtxt('%s/spectra/data_extracted.dat'%data_dir,unpack=True)
inv_cov=np.loadtxt('%s//spectra/covmat.dat'%data_dir)
cov=np.linalg.inv(inv_cov)
err_vec=np.sqrt(cov.diagonal())

bin_min,bin_max,bin_mean=np.loadtxt('%s/binning_file/binused.dat'%data_dir,unpack=True)

bin_min=bin_min[6:]
bin_max=bin_max[6:]
bin_mean=bin_mean[6:]

min={}
max={}
min['TT','100x100']=30
max['TT','100x100']=1197
min['TT','143x143']=30
max['TT','143x143']=1996
min['TT','143x217']=30
max['TT','143x217']=2508
min['TT','217x217']=30
max['TT','217x217']=2508

min['TE','100x100']=30
max['TE','100x100']=999
min['TE','100x143']=30
max['TE','100x143']=999
min['TE','100x217']=505
max['TE','100x217']=999
min['TE','143x143']=30
max['TE','143x143']=1996
min['TE','143x217']=505
max['TE','143x217']=1996
min['TE','217x217']=505
max['TE','217x217']=1996

min['EE','100x100']=30
max['EE','100x100']=999
min['EE','100x143']=30
max['EE','100x143']=999
min['EE','100x217']=505
max['EE','100x217']=999
min['EE','143x143']=30
max['EE','143x143']=1996
min['EE','143x217']=505
max['EE','143x217']=1996
min['EE','217x217']=505
max['EE','217x217']=1996

freqArray={}
freqArray['TT']=['100x100','143x143','143x217','217x217']
freqArray['TE']=['100x100','100x143','100x217','143x143','143x217','217x217']
freqArray['EE']=['100x100','100x143','100x217','143x143','143x217','217x217']

id_start=0
id_stop=0

ps={}
error={}
for spec in ['TT','EE','TE']:
    for freq in freqArray[spec]:
        id=np.where( (bin_min>=min[spec,freq]) & (bin_max<=max[spec,freq]))
        nbin=(id[0].shape[0])
        id_stop+=nbin
        
        ps[spec,freq]=vec[id_start:id_stop]
        error[spec,freq]=err_vec[id_start:id_stop]

        id_start+=nbin
        
        lb=bin_mean[id]
        
        np.savetxt('%s/spectra/spectrum_%s_%s.dat'%(data_dir,spec,freq),np.transpose([lb,ps[spec,freq],error[spec,freq]]))
        plt.errorbar(lb,ps[spec,freq]*lb**2/(2*np.pi),error[spec,freq]*lb**2/(2*np.pi),fmt='.')
    
    plt.show()




