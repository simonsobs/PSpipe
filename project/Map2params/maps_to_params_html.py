from pspy import pspy_utils, so_dict,so_map,so_mpi,sph_tools,so_mcm,so_spectra
import  numpy as np, pylab as plt, healpy as hp
import os,sys
import so_noise_calculator_public_20180822 as noise_calc
from pixell import curvedsky,powspec
import maps_to_params_utils
import h5py


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])


plot_dir='plot'

spectra=['TT','TE','TB','ET','BT','EE','EB','BE','BB']

freqs=d['freqs']

os.system('cp multistep2.js %s/multistep2.js'%plot_dir)
fileName='%s/SO_spectra.html'%plot_dir
g = open(fileName,mode="w")
g.write('<html>\n')
g.write('<head>\n')
g.write('<title> SO spectra </title>\n')
g.write('<script src="multistep2.js"></script>\n')
g.write('<script> add_step("sub",  ["c","v"]) </script> \n')
g.write('<script> add_step("all",  ["j","k"]) </script> \n')
g.write('<script> add_step("type",  ["a","z"]) </script> \n')
g.write('</head> \n')
g.write('<body> \n')
g.write('<div class=sub> \n')

for kind in ['cross','noise','auto']:
    g.write('<div class=all>\n')
    for spec in spectra:
        for fid1,f1 in enumerate(freqs):
            for fid2,f2 in enumerate(freqs):
                if fid1>fid2: continue
                
                str='spectra_%s_%sx%s_%s.png'%(spec,f1,f2,kind)
                g.write('<div class=type>\n')
                g.write('<img src="'+str+'" width="50%" /> \n')
                g.write('<img src="'+'diff_'+str+'" width="50%" /> \n')
                g.write('<img src="'+'frac_'+str+'" width="50%" /> \n')
                g.write('</div>\n')

    g.write('</div>\n')

g.write('</div> \n')
g.write('</body> \n')
g.write('</html> \n')
g.close()
