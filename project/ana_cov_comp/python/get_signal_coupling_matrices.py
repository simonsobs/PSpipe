"""
This script compute the analytical covariance matrix elements
between split power spectra
"""
import sys
import numpy as np
from pspipe_utils import log
from pspy import so_dict, so_map, pspy_utils
from itertools import product
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

couplings_dir = d['couplings_dir']
pspy_utils.create_directory(couplings_dir)

surveys = d['surveys']
arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}

if d['use_toeplitz_mcm'] == True:
    log.info('we will use the toeplitz approximation')
    l_exact, l_band, l_toep = 800, 2000, 2750
else:
    l_exact, l_band, l_toep = None, None, None

lmax = d['lmax']
niter = d['niter']

# format:
# - unroll all 'fields' i.e. (survey x array x chan x pol) is a 'field'
# - any given combination is then ('field' x 'field' x 'spintype')
# - canonical spintypes are ('00', '02', '++', '--')
#
# notes:
# - we are 'hardcoding' that all splits for a given field have the same
# analysis mask, so that we do not have to loop over splits.
# - we are taking advantage of fact that mapping of spintypes to pol
# is one-to-one for mcms, unlike covs, so we also only need to loop
# over pols, not spintypes too.

# we define the canon by the field order
fields = []
for sv1 in surveys:
    for ar1 in arrays[sv1]:
        for chan1 in arrays[sv1][ar1]:
            for pol1 in ['T', 'P']:
                fields.append((sv1, ar1, chan1, pol1))

# spintypes = ('00', '02', '++', '--')
pols2spintypes = {
    ('T', 'T'): ['00'],
    ('T', 'P'): ['02'],
    ('P', 'T'): ['02'],
    ('P', 'P'): ['++', '--']
}

# ensure no '20' terms
def canonize_mcm(f1, f2, fields):
    f1_idx = fields.index(f1)
    f2_idx = fields.index(f2)
    if f2_idx < f1_idx:
        return fields[f2_idx], fields[f1_idx]
    else:
        return fields[f1_idx], fields[f2_idx]
    
def get_mcm_fn(f1, f2, spintype):
    sv1, ar1, chan1, pol1 = f1
    sv2, ar2, chan2, pol2 = f2
    spintypes2fntags = {
        '00': '00',
        '02': '02',
        '++': 'pp',
        '--': 'mm'
    }
    return f'{sv1}_{ar1}_{chan1}_{pol1}x{sv2}_{ar2}_{chan2}_{pol2}_{spintypes2fntags[spintype]}'

def get_window(f1):
    sv1, ar1, chan1, pol1 = f1
    polstr = 'T' if pol1 == 'T' else 'pol'
    return so_map.read_map(d[f'window_{polstr}_{sv1}_{ar1}_{chan1}'])

canonized_combos = {}
for _f1, _f2 in product(fields, fields):
    f1, f2 = canonize_mcm(_f1, _f2, fields) 
    pol1, pol2 = f1[3], f2[3]
    
    for spintype in pols2spintypes[(pol1, pol2)]:
        base_fn = get_mcm_fn(f1, f2, spintype)
        full_fn = f'{couplings_dir}/{base_fn}.npy'
        
        if (f1, f2, spintype) not in canonized_combos:
            canonized_combos[(f1, f2, spintype)] = 1
            assert not os.path.isfile(full_fn), \
                f'{full_fn} exists but we should not yet have produced it in loop'
            log.info(base_fn)
            w1 = get_window(f1)
            w2 = get_window(f2)
            coupling = 1 # so_mcm.coupling_block(spintype, win1=w1, win2=w2,
                                            #  lmax=lmax, l3_pad=lmax,
                                            #  niter=niter, l_exact=l_exact,
                                            #  l_toep=l_toep, l_band=l_band)
            np.save(full_fn, coupling)
        else:
            canonized_combos[(f1, f2, spintype)] += 1
            assert os.path.isfile(full_fn), \
                f'{full_fn} does not exist but we should have produced it in loop'
            continue
np.save(f'{couplings_dir}/canonized_mcm_combos.npy', canonized_combos)

