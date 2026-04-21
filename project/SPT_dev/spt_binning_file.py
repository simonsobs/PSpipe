import numpy as np
import scipy.stats as stats
import csv

#### redo SPT binning

ell = np.arange(4001)

bin_edges = np.arange(0, 5000, 50)

lmax = bin_edges[1:].copy()
lmin = bin_edges[:-1].copy()
lmin[0]= 2
lmin[1:] += 1

lmean = (lmin + lmax)/2
np.savetxt("binning_spt.dat", np.transpose([lmin, lmax, lmean]))

#### now my binning

def generate_binning_file(file_name, start, end):

    bin_min = start
    with open(file_name, mode='w', newline='') as file:
    
        writer = csv.writer(file, delimiter=' ')
                
        while bin_min <= end:
        
            if bin_min < 1800:
                size = 75
            elif bin_min < 2500:
                size = 120
            else:
                size = 180
                
            bin_max = bin_min + size - 1
            bin_mean = (bin_min + bin_max) / 2.0
            
            writer.writerow([bin_min, bin_max, bin_mean])
            bin_min = bin_max + 1

file_name = "binning_spt_wider.dat"
start = 2
end = 5000
generate_binning_file(file_name, start, end)


