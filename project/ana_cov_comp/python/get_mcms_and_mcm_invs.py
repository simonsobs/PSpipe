description = """
This script computes the mcms and mcm_invs from 2pt covariance couplings. These
matrices are needed in get_noise_model.py, get_pseudonoise.py, and 
get_pseudosignal.py, where e.g. we need to take an unbinned power spectrum and 
turn it into an unbinned pseudospectrum in accordance with the INKA
prescription. The default PSpipe products include the effects of the beam and
the binning, whereas for these operations we need the "pure" mode coupling.

It is short enough that it should always run in a one-shot job, so it 
accepts no arguments other than paramfile.
"""
import numpy as np
from pspipe_utils import log, pspipe_list, covariance as psc
from pspy import so_dict, pspy_utils
from itertools import product, combinations_with_replacement as cwr
import os
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

log = log.get_logger(**d)

ewin_alms_dir = d['ewin_alms_dir']
couplings_dir = d['couplings_dir']
pspy_utils.create_directory(couplings_dir)

sv2arrs2chans = pspipe_list.get_survey_array_channel_map(d)

lmax_pseudocov = d['lmax_pseudocov']
assert lmax_pseudocov >= d['lmax'], \
    f"{lmax_pseudocov=} must be >= {d['lmax']=}" 
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
                    
                    ewin_info_s = psc.get_ewin_info_from_field_info(field_info, d, mode='w', return_paths_ops=True)
                    if ewin_info_s not in ewin_infos:
                        ewin_infos.append(ewin_info_s)
                    else:
                        pass

                    ewin_info_n = psc.get_ewin_info_from_field_info(field_info, d, mode='ws', extra='sqrt_pixar', return_paths_ops=True)
                    if ewin_info_n not in ewin_infos:
                        ewin_infos.append(ewin_info_n)
                    else:
                        pass

canonized_combos = {}

# iterate over all pairs/orders of fields, and get the canonized window pairs
for field_info1, field_info2 in product(field_infos, repeat=2):
    # S S 
    ewin_name1, ewin_name2 = psc.canonize_connected_2pt(
        psc.get_ewin_info_from_field_info(field_info1, d, mode='w'),
        psc.get_ewin_info_from_field_info(field_info2, d, mode='w'),
        ewin_infos
        ) 
    
    mcm_fns = []
    mcm_inv_fns = []
    for fntag in ('00', '02', 'diag', 'off'): # diag is for EE, BB subblock; off is for EB, BE subblock
        mcm_fns.append(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_{fntag}_mcm.npy')
        mcm_inv_fns.append(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_{fntag}_mcm_inv.npy')
    
    if (ewin_name1, ewin_name2) not in canonized_combos:
        canonized_combos[(ewin_name1, ewin_name2)] = [(field_info1, field_info2)]

        # Need to do 00, 02 individually
        mcm_fn = mcm_fns[0] # 00
        mcm_inv_fn = mcm_inv_fns[0] # 00
        if os.path.isfile(mcm_fn) and os.path.isfile(mcm_inv_fn):
            log.info(f'{mcm_fn} exists and {mcm_inv_fn} exists, skipping')
        else:
            log.info(os.path.splitext(os.path.basename(mcm_fn))[0])
            log.info(os.path.splitext(os.path.basename(mcm_inv_fn))[0])

            mcm = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_00_coupling.npy')
            
            mcm *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
            np.save(mcm_fn, mcm)

            mcm_inv = np.linalg.inv(mcm)
            np.save(mcm_inv_fn, mcm_inv)

        mcm_fn = mcm_fns[1] # 02
        mcm_inv_fn = mcm_inv_fns[1] # 02
        if os.path.isfile(mcm_fn) and os.path.isfile(mcm_inv_fn):
            log.info(f'{mcm_fn} exists and {mcm_inv_fn} exists, skipping')
        else:
            log.info(os.path.splitext(os.path.basename(mcm_fn))[0])
            log.info(os.path.splitext(os.path.basename(mcm_inv_fn))[0])

            mcm = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_02_coupling.npy')
            
            # fill upper left diag with w2
            # strictly speaking, this shouldn't be necessary (since the upper left
            # 2-element block of the matrix is set to I2 by pspy); but, due to 
            # paranoia, we want these numbers to be the same order-of-magnitude
            # as the non-trivial part of the matrix, in case some shadowy numerics
            # happen
            w2 = np.load(f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy')
            mcm[0, 0] = w2
            mcm[1, 1] = w2

            mcm *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
            np.save(mcm_fn, mcm)

            mcm_inv = np.linalg.inv(mcm)
            np.save(mcm_inv_fn, mcm_inv)

        # Then need to do pp, mm jointly
        mcm_fn = mcm_fns[2] # diag
        mcm_inv_fn = mcm_inv_fns[2] # diag
        if os.path.isfile(mcm_fn) and os.path.isfile(mcm_inv_fn):
            log.info(f'{mcm_fn} exists and {mcm_inv_fn} exists, skipping')
        else:
            log.info(os.path.splitext(os.path.basename(mcm_fn))[0])
            log.info(os.path.splitext(os.path.basename(mcm_inv_fn))[0])

            mcm_pp = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_pp_coupling.npy')
            mcm_mm = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_mm_coupling.npy')

            # fill upper left diag with w2
            w2 = np.load(f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy')
            mcm_pp[0, 0] = w2
            mcm_pp[1, 1] = w2
            mcm_mm[0, 0] = 0
            mcm_mm[1, 1] = 0

            mcm_pp *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
            mcm_mm *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
            mcm = np.block([[mcm_pp, mcm_mm], [mcm_mm, mcm_pp]])
            np.save(mcm_fn, mcm)

            mcm_inv = np.linalg.inv(mcm)
            np.save(mcm_inv_fn, mcm_inv)

        mcm_fn = mcm_fns[3] # off 
        mcm_inv_fn = mcm_inv_fns[3] # off
        if os.path.isfile(mcm_fn) and os.path.isfile(mcm_inv_fn):
            log.info(f'{mcm_fn} exists and {mcm_inv_fn} exists, skipping')
        else:
            log.info(os.path.splitext(os.path.basename(mcm_fn))[0])
            log.info(os.path.splitext(os.path.basename(mcm_inv_fn))[0])

            mcm_pp = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_pp_coupling.npy')
            mcm_mm = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_mm_coupling.npy')
            
            # fill upper left diag with w2
            w2 = np.load(f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy')
            mcm_pp[0, 0] = w2
            mcm_pp[1, 1] = w2
            mcm_mm[0, 0] = 0
            mcm_mm[1, 1] = 0

            mcm_pp *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
            mcm_mm *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
            mcm = np.block([[mcm_pp, -mcm_mm], [-mcm_mm, mcm_pp]])
            np.save(mcm_fn, mcm)

            mcm_inv = np.linalg.inv(mcm)
            np.save(mcm_inv_fn, mcm_inv)
    else:
        canonized_combos[(ewin_name1, ewin_name2)].append((field_info1, field_info2))
        for mcm_fn, mcm_inv_fn in zip(mcm_fns, mcm_inv_fns):
            assert os.path.isfile(mcm_fn) and os.path.isfile(mcm_inv_fn), \
                f'{mcm_fn} and {mcm_inv_fn} do not exist but we should have produced them in loop'

    # N N 
    sv1, sv2 = field_info1[0], field_info2[0]
    ar1, ar2 = field_info1[1], field_info2[1]
    split1, split2 = field_info1[3], field_info2[3]
    if (sv1 != sv2) or (ar1 != ar2) or (split1 != split2):
        pass
    else:
        ewin_name1, ewin_name2 = psc.canonize_connected_2pt(
            psc.get_ewin_info_from_field_info(field_info1, d, mode='ws', extra='sqrt_pixar'),
            psc.get_ewin_info_from_field_info(field_info2, d, mode='ws', extra='sqrt_pixar'),
            ewin_infos
            ) 
        
        mcm_fns = []
        mcm_inv_fns = []
        for fntag in ('00', '02', 'diag', 'off'):
            mcm_fns.append(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_{fntag}_mcm.npy')
            mcm_inv_fns.append(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_{fntag}_mcm_inv.npy')
        
        if (ewin_name1, ewin_name2) not in canonized_combos:
            canonized_combos[(ewin_name1, ewin_name2)] = [(field_info1, field_info2)]

            # Need to do 00, 02 individually
            mcm_fn = mcm_fns[0]
            mcm_inv_fn = mcm_inv_fns[0]
            if os.path.isfile(mcm_fn) and os.path.isfile(mcm_inv_fn):
                log.info(f'{mcm_fn} exists and {mcm_inv_fn} exists, skipping')
            else:
                log.info(os.path.splitext(os.path.basename(mcm_fn))[0])
                log.info(os.path.splitext(os.path.basename(mcm_inv_fn))[0])

                mcm = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_00_coupling.npy')
                
                mcm *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
                np.save(mcm_fn, mcm)

                mcm_inv = np.linalg.inv(mcm)
                np.save(mcm_inv_fn, mcm_inv)

            mcm_fn = mcm_fns[1]
            mcm_inv_fn = mcm_inv_fns[1]
            if os.path.isfile(mcm_fn) and os.path.isfile(mcm_inv_fn):
                log.info(f'{mcm_fn} exists and {mcm_inv_fn} exists, skipping')
            else:
                log.info(os.path.splitext(os.path.basename(mcm_fn))[0])
                log.info(os.path.splitext(os.path.basename(mcm_inv_fn))[0])

                mcm = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_02_coupling.npy')

                # fill upper left diag with w2
                w2 = np.load(f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy')
                mcm[0, 0] = w2
                mcm[1, 1] = w2

                mcm *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
                np.save(mcm_fn, mcm)

                mcm_inv = np.linalg.inv(mcm)
                np.save(mcm_inv_fn, mcm_inv)

            # Then need to do pp, mm jointly
            mcm_fn = mcm_fns[2]
            mcm_inv_fn = mcm_inv_fns[2]
            if os.path.isfile(mcm_fn) and os.path.isfile(mcm_inv_fn):
                log.info(f'{mcm_fn} exists and {mcm_inv_fn} exists, skipping')
            else:
                log.info(os.path.splitext(os.path.basename(mcm_fn))[0])
                log.info(os.path.splitext(os.path.basename(mcm_inv_fn))[0])

                mcm_pp = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_pp_coupling.npy')
                mcm_mm = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_mm_coupling.npy')

                # fill upper left diag with w2
                w2 = np.load(f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy')
                mcm_pp[0, 0] = w2
                mcm_pp[1, 1] = w2
                mcm_mm[0, 0] = 0
                mcm_mm[1, 1] = 0

                mcm_pp *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
                mcm_mm *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
                mcm = np.block([[mcm_pp, mcm_mm], [mcm_mm, mcm_pp]])
                np.save(mcm_fn, mcm)

                mcm_inv = np.linalg.inv(mcm)
                np.save(mcm_inv_fn, mcm_inv)

            mcm_fn = mcm_fns[3]
            mcm_inv_fn = mcm_inv_fns[3]
            if os.path.isfile(mcm_fn) and os.path.isfile(mcm_inv_fn):
                log.info(f'{mcm_fn} exists and {mcm_inv_fn} exists, skipping')
            else:
                log.info(os.path.splitext(os.path.basename(mcm_fn))[0])
                log.info(os.path.splitext(os.path.basename(mcm_inv_fn))[0])

                mcm_pp = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_pp_coupling.npy')
                mcm_mm = np.load(f'{couplings_dir}/{ewin_name1}x{ewin_name2}_mm_coupling.npy')

                # fill upper left diag with w2
                w2 = np.load(f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy')
                mcm_pp[0, 0] = w2
                mcm_pp[1, 1] = w2
                mcm_mm[0, 0] = 0
                mcm_mm[1, 1] = 0

                mcm_pp *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
                mcm_mm *= (2*np.arange(lmax_pseudocov+1) + 1) / (4*np.pi)
                mcm = np.block([[mcm_pp, -mcm_mm], [-mcm_mm, mcm_pp]])
                np.save(mcm_fn, mcm)

                mcm_inv = np.linalg.inv(mcm)
                np.save(mcm_inv_fn, mcm_inv)
        else:
            canonized_combos[(ewin_name1, ewin_name2)].append((field_info1, field_info2))
            for mcm_fn, mcm_inv_fn in zip(mcm_fns, mcm_inv_fns):
                assert os.path.isfile(mcm_fn) and os.path.isfile(mcm_inv_fn), \
                    f'{mcm_fn} and {mcm_inv_fn} do not exist but we should have produced them in loop'

np.save(f'{couplings_dir}/canonized_mcms_and_mcm_invs_combos.npy', canonized_combos)