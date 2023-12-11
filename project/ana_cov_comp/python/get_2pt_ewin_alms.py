"""
This script computes the alms for covariance couplings
"""
import sys
import numpy as np
from pspipe_utils import log, covariance as psc
from pspy import so_dict, so_map, sph_tools, pspy_utils
from itertools import product, combinations_with_replacement as cwr
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

ewin_alms_dir = d['ewin_alms_dir']
pspy_utils.create_directory(ewin_alms_dir)

surveys = d['surveys']
arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}

lmax = d['lmax']
niter = d['niter']

# format:
# - unroll all 'fields' i.e. (survey x array x chan x split x pol) is a 'field'
# - any given combination is then ('field' x 'field')
#
# notes:
# - we are 'hardcoding' that all splits for a given field have the same
# analysis mask, and that all pols for a given field have the same 
# sigma map.

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

canonized_combos = {}

# S S 
# iterate over all pairs/orders of fields, and get the canonized window pairs
for field_info1, field_info2 in product(field_infos, repeat=2):
    (ewin_name1, ewin_paths1, ewin_ops1), \
    (ewin_name2, ewin_paths2, ewin_ops2) = psc.canonize_connected_2pt(
        psc.get_ewin_info_from_field_info(field_info1, d, mode='w'),
        psc.get_ewin_info_from_field_info(field_info2, d, mode='w'),
        ewin_infos
        ) 
    
    alm_fn = f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_alm.npy'

    if (ewin_name1, ewin_name2) not in canonized_combos:
        canonized_combos[(ewin_name1, ewin_name2)] = [(field_info1, field_info2)]
        assert not os.path.isfile(alm_fn), \
            f'{alm_fn} exists but we should not yet have produced it in loop'
        log.info(os.path.splitext(os.path.basename(alm_fn))[0])

        ewin12_data = 1
        for i, path12 in enumerate((*ewin_paths1, *ewin_paths2)):
            ewin12 = so_map.read_map(path12)
            ewin12.data = psc.optags2ops((*ewin_ops1, *ewin_ops2)[i])(ewin12.data)
            ewin12_data *= ewin12.data
        ewin12.data = ewin12_data
        ewin12_data = None

        # note we are going to calculate window alm to 2lmax, but that's ok because
        # the lmax_limit is half the Nyquist limit
        assert lmax <= ewin12.get_lmax_limit(), \
            "the requested lmax is too high with respect to the map pixellisation"

        alm = sph_tools.map2alm(ewin12, niter=niter, lmax=2*lmax)
        np.save(alm_fn, alm)
    else:
        canonized_combos[(ewin_name1, ewin_name2)].append((field_info1, field_info2))
        assert os.path.isfile(alm_fn), \
            f'{alm_fn} does not exist but we should have produced it in loop'
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
        (ewin_name1, ewin_paths1, ewin_ops1), \
        (ewin_name2, ewin_paths2, ewin_ops2) = psc.canonize_connected_2pt(
            psc.get_ewin_info_from_field_info(field_info1, d, mode='ws', extra='sqrt_pixar'),
            psc.get_ewin_info_from_field_info(field_info2, d, mode='ws', extra='sqrt_pixar'),
            ewin_infos
            ) 
        
        alm_fn = f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_alm.npy'

        if (ewin_name1, ewin_name2) not in canonized_combos:
            canonized_combos[(ewin_name1, ewin_name2)] = [(field_info1, field_info2)]
            assert not os.path.isfile(alm_fn), \
                f'{alm_fn} exists but we should not yet have produced it in loop'
            log.info(os.path.splitext(os.path.basename(alm_fn))[0])

            ewin12_data = 1
            for i, path12 in enumerate((*ewin_paths1, *ewin_paths2)):
                ewin12 = so_map.read_map(path12)
                ewin12.data = psc.optags2ops((*ewin_ops1, *ewin_ops2)[i])(ewin12.data)
                ewin12_data *= ewin12.data
            ewin12.data = ewin12_data
            ewin12_data = None

            # we know sqrt_pixar**2 = pixar
            ewin12.data *= ewin12.data.pixsizemap()

            # note we are going to calculate window alm to 2lmax, but that's ok because
            # the lmax_limit is half the Nyquist limit
            assert lmax <= ewin12.get_lmax_limit(), \
                "the requested lmax is too high with respect to the map pixellisation"

            alm = sph_tools.map2alm(ewin12, niter=niter, lmax=2*lmax)
            np.save(alm_fn, alm)
        else:
            canonized_combos[(ewin_name1, ewin_name2)].append((field_info1, field_info2))
            assert os.path.isfile(alm_fn), \
                f'{alm_fn} does not exist but we should have produced it in loop'
            continue

np.save(f'{ewin_alms_dir}/canonized_ewin_alms_2pt_combos.npy', canonized_combos)