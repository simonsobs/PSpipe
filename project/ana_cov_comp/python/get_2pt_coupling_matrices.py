"""
This script computes the analytical covariance matrix elements
between split power spectra
"""
import sys
import numpy as np
from pspipe_utils import log, covariance as psc
from pspy import so_dict, so_map, so_mcm, pspy_utils
from itertools import product, combinations_with_replacement as cwr
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

ewin_alms_dir = d['ewin_alms_dir']
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

# format:
# - unroll all 'fields' i.e. (survey x array x chan x split x pol) is a 'field'
# - any given combination is then ('field' x 'field' x 'spintype')
# - canonical spintypes are ('00', '02', '++', '--')
#
# notes:
# - we are 'hardcoding' that all splits for a given field have the same
# analysis mask, and that all pols for a given field have the same 
# sigma map.
# - we are taking advantage of fact that we have a narrow mapping of pol
# to spintypes.

# we define the canon by the windows order. we first build the fields,
# then use a mapping from fields to windows to build the canonical
# windows
field_infos = []
ewin_infos = []
for sv1 in surveys:
    for ar1 in arrays[sv1]:
        for chan1 in arrays[sv1][ar1]:
            for split1 in range(len(d[f'maps_{sv1}_{ar1}_{chan1}'])):
                for pol1 in ['T', 'P']:
                    field_info = (sv1, ar1, chan1, split1, pol1)
                    if field_info not in field_infos:
                        field_infos.append(field_info)
                    else:
                        raise ValueError(f'{field_info=} is not unique')
                    
                    ewin_info_s = psc.get_ewin_info_from_field_info(field_info, d, mode='w')
                    if ewin_info_s not in ewin_infos:
                        ewin_infos.append(ewin_info_s)
                    else:
                        pass

                    ewin_info_n = psc.get_ewin_info_from_field_info(field_info, d, mode='ws', extra='sqrt_pixar')
                    if ewin_info_n not in ewin_infos:
                        ewin_infos.append(ewin_info_n)
                    else:
                        pass

# we will reduce some of the loop by this mapping
pols2spintypes = {
    ('T', 'T'): ['00'],
    ('T', 'P'): ['02'],
    ('P', 'T'): ['02'],
    ('P', 'P'): ['++', '--']
}

canonized_combos = {}

# S S
# iterate over all pairs/orders of fields, and get the canonized window pairs
for field_info1, field_info2 in product(field_infos, repeat=2):
    (ewin_name1, _, _), \
    (ewin_name2, _, _) = psc.canonize_connected_2pt(
        psc.get_ewin_info_from_field_info(field_info1, d, mode='w'),
        psc.get_ewin_info_from_field_info(field_info2, d, mode='w'),
        ewin_infos
        ) 
    pol1, pol2 = field_info1[4], field_info2[4]
    
    for spintype in pols2spintypes[(pol1, pol2)]:
        coupling_fn = f'{couplings_dir}/{ewin_name1}x{ewin_name2}_{psc.spintypes2fntags[spintype]}_coupling.npy'

        # if we haven't yet gotten to this canonized window pair, keep track of
        # the fields, ensure we haven't calculated the coupling and calculate
        # the coupling
        if (ewin_name1, ewin_name2, spintype) not in canonized_combos:
            canonized_combos[(ewin_name1, ewin_name2, spintype)] = [(field_info1, field_info2, spintype)]
            assert not os.path.isfile(coupling_fn), \
                f'{coupling_fn} exists but we should not yet have produced it in loop'
            log.info(os.path.splitext(os.path.basename(coupling_fn))[0])
            
            w1 = np.load(f'{ewin_alms_dir}/{ewin_name1}_alm.npy')
            w2 = np.load(f'{ewin_alms_dir}/{ewin_name2}_alm.npy')
            coupling = so_mcm.coupling_block(spintype, win1=w1, win2=w2,
                                             lmax=lmax, input_alm=True,
                                             l_exact=l_exact, l_toep=l_toep,
                                             l_band=l_band)
            np.save(coupling_fn, coupling)  

        # if we have already gotten to this canonized window pair, add the
        # fields to the tracked list and ensure we've already calculated the
        # coupling 
        else:
            canonized_combos[(ewin_name1, ewin_name2, spintype)].append((field_info1, field_info2, spintype))
            assert os.path.isfile(coupling_fn), \
                f'{coupling_fn} does not exist but we should have produced it in loop'
            continue

# N N
# iterate over all pairs/orders of fields, and get the canonized window pairs
for field_info1, field_info2 in product(field_infos, repeat=2):
    sv1, sv2 = field_info1[0], field_info2[0]
    ar1, ar2 = field_info1[1], field_info2[1]
    split1, split2 = field_info1[3], field_info2[3]
    if (sv1 != sv2) or (ar1 != ar2) or (split1 != split2):
        continue
    else:
        (ewin_name1, _, _), \
        (ewin_name2, _, _) = psc.canonize_connected_2pt(
            psc.get_ewin_info_from_field_info(field_info1, d, mode='ws', extra='sqrt_pixar'),
            psc.get_ewin_info_from_field_info(field_info2, d, mode='ws', extra='sqrt_pixar'),
            ewin_infos
            ) 
        pol1, pol2 = field_info1[4], field_info2[4]
        
        for spintype in pols2spintypes[(pol1, pol2)]:
            coupling_fn = f'{couplings_dir}/{ewin_name1}x{ewin_name2}_{psc.spintypes2fntags[spintype]}_coupling.npy'

            # if we haven't yet gotten to this canonized window pair, keep track of
            # the fields, ensure we haven't calculated the coupling and calculate
            # the coupling
            if (ewin_name1, ewin_name2, spintype) not in canonized_combos:
                canonized_combos[(ewin_name1, ewin_name2, spintype)] = [(field_info1, field_info2, spintype)]
                assert not os.path.isfile(coupling_fn), \
                    f'{coupling_fn} exists but we should not yet have produced it in loop'
                log.info(os.path.splitext(os.path.basename(coupling_fn))[0])
                
                w1 = np.load(f'{ewin_alms_dir}/{ewin_name1}_alm.npy')
                w2 = np.load(f'{ewin_alms_dir}/{ewin_name2}_alm.npy')
                coupling = so_mcm.coupling_block(spintype, win1=w1, win2=w2,
                                                 lmax=lmax, input_alm=True,
                                                 l_exact=l_exact, l_toep=l_toep,
                                                 l_band=l_band)
                np.save(coupling_fn, coupling)  

            # if we have already gotten to this canonized window pair, add the
            # fields to the tracked list and ensure we've already calculated the
            # coupling 
            else:
                canonized_combos[(ewin_name1, ewin_name2, spintype)].append((field_info1, field_info2, spintype))
                assert os.path.isfile(coupling_fn), \
                    f'{coupling_fn} does not exist but we should have produced it in loop'
                continue

np.save(f'{couplings_dir}/canonized_couplings_2pt_combos.npy', canonized_combos)