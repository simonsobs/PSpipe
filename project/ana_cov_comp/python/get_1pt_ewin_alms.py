# FIXME: make these comments more verbose!
"""
This script computes the "1 point" window alms for covariance couplings. 
That is, for all the windows we have (whether signal-weighted i.e. 
analaysis masks or noise-weighted i.e. analysis masks * sigma maps), it
computes the alm of that window. Later on, in get_2pt_coupling_matrices,
we will use a pair of 1pt window alms to make "2pt" couplings a.k.a. 
mode-coupling matrices. These mode coupling matrices are necessary
for obtaining pseudospectra as part of the INKA perscription.
"""
import sys
import numpy as np
from pspipe_utils import log, pspipe_list, covariance as psc
from pspy import so_dict, so_map, sph_tools, pspy_utils
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

ewin_alms_dir = d['ewin_alms_dir']
pspy_utils.create_directory(ewin_alms_dir)

sv2arrs2chans = pspipe_list.get_survey_array_channel_map(d)

lmax = d['lmax']
niter = d['niter']

# format:
# - unroll all 'fields' i.e. (survey x array x chan x split x pol) is a 'field'
#
# notes:
# - we are 'hardcoding' that all splits for a given field have the same
# analysis mask, and that all pols for a given field have the same 
# sigma map.

# we define the canon by the windows order. we first build the fields,
# then use a mapping from fields to windows to build the canonical
# windows
field_infos = []
for sv1 in sv2arrs2chans:
    for ar1 in sv2arrs2chans[sv1]:
        for chan1 in sv2arrs2chans[sv1][ar1]:
            for split1 in range(len(d[f'maps_{sv1}_{ar1}_{chan1}'])):
                for pol1 in ['T', 'P']:
                    field_info = (sv1, ar1, chan1, split1, pol1)
                    if field_info not in field_infos:
                        field_infos.append(field_info)
                    else:
                        raise ValueError(f'{field_info=} is not unique')

canonized_combos = {}

# iterate over all canonized windows
for field_info1 in field_infos:
    # S
    ewin_name1, ewin_paths1, ewin_ops1 = psc.get_ewin_info_from_field_info(
        field_info1, d, mode='w', return_paths_ops=True
        )
    
    alm_fn = f'{ewin_alms_dir}/{ewin_name1}_alm.npy'

    if ewin_name1 not in canonized_combos:
        canonized_combos[ewin_name1] = [field_info1]
        if os.path.isfile(alm_fn):
            log.info(f'{alm_fn} exists, skipping')
        else:
            log.info(ewin_name1)

            ewin1_data = 1
            for i, path1 in enumerate(ewin_paths1):
                ewin1 = so_map.read_map(path1)
                ewin1.data = psc.optags2ops[ewin_ops1[i]](ewin1.data)
                ewin1_data *= ewin1.data
            ewin1.data = ewin1_data
            ewin1_data = None

            # note we are going to calculate window alm to 2lmax, but that's ok because
            # the lmax_limit is half the Nyquist limit
            assert lmax <= ewin1.get_lmax_limit(), \
                "the requested lmax is too high with respect to the map pixellisation"

            alm = sph_tools.map2alm(ewin1, niter=niter, lmax=2*lmax)
            np.save(alm_fn, alm)
    else:
        canonized_combos[ewin_name1].append(field_info1)
        assert os.path.isfile(alm_fn), \
            f'{alm_fn} does not exist but we should have produced it in loop'

    # N
    ewin_name1, ewin_paths1, ewin_ops1 = psc.get_ewin_info_from_field_info(
        field_info1, d, mode='ws', extra='sqrt_pixar', return_paths_ops=True
        )
    
    alm_fn = f'{ewin_alms_dir}/{ewin_name1}_alm.npy'

    if ewin_name1 not in canonized_combos:
        canonized_combos[ewin_name1] = [field_info1]
        if os.path.isfile(alm_fn):
            log.info(f'{alm_fn} exists, skipping')
        else:
            log.info(ewin_name1)

            ewin1_data = 1
            for i, path1 in enumerate(ewin_paths1):
                ewin1 = so_map.read_map(path1)
                ewin1.data = psc.optags2ops[ewin_ops1[i]](ewin1.data)
                ewin1_data *= ewin1.data
            ewin1.data = ewin1_data
            ewin1_data = None

            # we know we need to multiply by sqrt_pixar
            ewin1.data *= ewin1.data.pixsizemap()**0.5

            # note we are going to calculate window alm to 2lmax, but that's ok because
            # the lmax_limit is half the Nyquist limit
            assert lmax <= ewin1.get_lmax_limit(), \
                "the requested lmax is too high with respect to the map pixellisation"

            alm = sph_tools.map2alm(ewin1, niter=niter, lmax=2*lmax)
            np.save(alm_fn, alm)
    else:
        canonized_combos[ewin_name1].append(field_info1)
        assert os.path.isfile(alm_fn), \
            f'{alm_fn} does not exist but we should have produced it in loop'

np.save(f'{ewin_alms_dir}/canonized_ewin_alms_1pt_combos.npy', canonized_combos)