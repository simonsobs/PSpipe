description = """We need several inputs to the covariance computation that are
themselves functions of (up to) 4 points, so it makes sense to have a dedicated
script to obtain them. They are: the w4 scalars for 4 effective masks, and 
the corresponding window power spectra. Unlike DR6, we account for anisotropy
in the noise which is necessary to achieve percent-level accuracy in the 
analytic covariance.

In addition the usual complexities of the book-keeping associated with all the
possible combinations, an additional complexity we add are approximations that
map some 4 points to a reference 4 point to minimize the number of window
spectra (and ultimately, couplings), that need to be produced."""

import argparse
from os.path import join as opj
from itertools import product
import time

import numpy as np
import numba
from pixell import enmap, curvedsky, wcsutils

from pspipe_utils import log, pspipe_list, covariance, dict_utils
from pspy import so_dict, so_mpi, pspy_utils

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
args = parser.parse_args()

# get needed info from paramfile
d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

surveys = d["surveys"]
lmax = d["lmax"]
cov_correlation_by_noise_model = d['cov_correlation_by_noise_model']
cov_use_median_ssnn_nnss_coupling_per_block = d['cov_use_median_ssnn_nnss_coupling_per_block']
cov_use_median_nnnn_coupling_per_block = d['cov_use_median_nnnn_coupling_per_block']
dl_window_spectra = d['dl_window_spectra']
if dl_window_spectra > lmax:
    raise ValueError(f'{dl_window_spectra=} must be <= {lmax=}')

cov_dir = d['cov_dir']
pspy_utils.create_directory(cov_dir)

# define functions that will only be used in this script, but are used more than
# once (for getting the w4s and the window_spectra)
def get_effective_window_tuple_from_map(d, sv, m, pol, split='s'):
    # window doesn't depend on split
    full_window_path = d[f'window_{pol}_{sv}_{m}']
    
    if split == 's':
        return (full_window_path,)
    else:
        assert split[0] == 'n', \
            r"PSpipe indexes splits with either 's' for signal or 'n{k}' for noise from split k" 
        split = int(split[1])
        
        # ivar doesn't depend on pol
        full_map_path = d[f'maps_{sv}_{m}'][split]
        if d[f"src_free_maps_{sv}"] == True:
            full_ivar_path = full_map_path.replace('_map_srcfree', '_ivar')
        else:
            full_ivar_path = full_map_path.replace('_map', '_ivar')   
        
        return (full_window_path, full_ivar_path)
    
def get_signal_and_noise_windows_to_load_sets(list_of_tuple_of_winsets, subtasks):
    signal_windows_to_load = set() # ('/path/1',)
    noise_windows_to_load = set() # ('/path/1', '/path/2')
    for task in subtasks:
        tuple_of_winsets = list_of_tuple_of_winsets[task]

        # add item already in set to set does nothing
        for window_set in tuple_of_winsets:
            if len(window_set) == 1:
                signal_windows_to_load.add(window_set)
            elif len(window_set) == 2:
                signal_windows_to_load.add((window_set[0],))
                noise_windows_to_load.add(window_set)
            else:
                raise ValueError(f'Can only have two window paths in an effective window')
            
    return signal_windows_to_load, noise_windows_to_load

# NOTE: ivar masks will be square-root-inverted and multiplied by square-root
# pixel area when loaded
@numba.njit(parallel=True)
def prep_noise_window(signal_window, ivar_window, sqrt_pixsizemap):
    # use nan_to_num because "where" argument of reciprocal is not supported, 
    # but np.where is much slower
    sigma_window = np.nan_to_num(1/ivar_window, copy=False, nan=0., posinf=0., neginf=0.)
    sigma_window **= 0.5
    return signal_window * sigma_window * sqrt_pixsizemap

def load_effective_windows(signal_windows_to_load, noise_windows_to_load, 
                           effective_window_dict=None, pixsizemap_dict=None):
    if effective_window_dict is None:
        effective_window_dict = {} # keys are effective window tuples ('/path/1',) or ('/path/1', '/path/2'), values are the windows
    if pixsizemap_dict is None:
        pixsizemap_dict = {} # keys are (map.shape, wcsutils.describe(map.wcs)). NOTE: limited by precision of describe str

    for windows_to_load in (signal_windows_to_load, noise_windows_to_load): 
        # load signal first so they are in memory for noise. 
        # TODO: evaluate trade-off of doing s and n separately (spends memory to speed
        # up loading)
        # TODO: if worth it, rewrite so noise_windows are batched by "uneeded" 
        # signal_windows, and those signal_windows are deleted after their "child"
        # noise_windows are loaded
        for window_set in windows_to_load:
            assert window_set not in effective_window_dict, \
                f'{window_set} already in effective_window_dict, this should not ' + \
                'happen since sets should only hold unique items'

            signal_window_path = window_set[0]

            if len(window_set) == 1: # signal_window
                signal_window = enmap.read_map(signal_window_path)
                effective_window_dict[window_set] = signal_window

                # also load the pixsizemap and sqrt_pixsizemaps (we do it on-the-fly
                # rather than pre-computing the unique set because there should be 
                # very few of these)
                pixsizemap_key = (signal_window.shape, wcsutils.describe(signal_window.wcs))
                if pixsizemap_key not in pixsizemap_dict:
                    pixsizemap = signal_window.pixsizemap()
                    
                    pixsizemap_dict[pixsizemap_key] = {}
                    pixsizemap_dict[pixsizemap_key][1] = pixsizemap
                    pixsizemap_dict[pixsizemap_key][0.5] = pixsizemap ** 0.5

            elif len(window_set) == 2: # noise_window 
                # we did signal first, so this should never have KeyErrors  
                signal_window = effective_window_dict[(signal_window_path,)]
                pixsizemap_key = (signal_window.shape, wcsutils.describe(signal_window.wcs))
                sqrt_pixsizemap = pixsizemap_dict[pixsizemap_key][0.5]

                # unless doing isotropy test, an ivar will never be loaded again 
                # (e.g., with a different signal window) after being stored in an
                # effective window
                ivar_window_path = window_set[1]
                ivar_window = enmap.read_map(ivar_window_path, geometry=signal_window.geometry)

                noise_window = prep_noise_window(signal_window, ivar_window, sqrt_pixsizemap)
                effective_window_dict[window_set] = enmap.samewcs(noise_window, ivar_window)
    
    return effective_window_dict, pixsizemap_dict

# FIXME: multiplication of masks with different pixelizations/resolutions not 
# defined
@numba.njit(parallel=True)
def mult_2(w1, w2):
    return w1 * w2

# FIXME: multiplication of masks with different pixelizations/resolutions not 
# defined
@numba.njit(parallel=True)
def calc_w2(w1, w2, pixsizemap):
    return np.sum(w1 * w2 * pixsizemap)/(4*np.pi)

# FIXME: multiplication of masks with different pixelizations/resolutions not 
# defined
@numba.njit(parallel=True)
def calc_w4(w1, w2, w3, w4, pixsizemap):
    return np.sum(w1 * w2 * w3 * w4 * pixsizemap)/(4*np.pi)

def calc_wl(w1, w2, w3, w4):
    walm12 = curvedsky.map2alm(enmap.samewcs(mult_2(w1, w2), w1), lmax=lmax + dl_window_spectra)
    walm34 = curvedsky.map2alm(enmap.samewcs(mult_2(w3, w4), w3), lmax=lmax + dl_window_spectra)
    wl = curvedsky.alm2cl(walm12, walm34, dtype=np.float64)
    return wl

def calc_canonized_function(canonized_combos, subtasks, effective_window_dict, 
                            pixsizemap_dict, func, func_takes_pixsizemap_arg):
    canonized_wfs = {}
    for task in subtasks:
        can_com = canonized_combos[task]
        assert can_com not in canonized_wfs, \
            f'{can_com} already in canonized_wfs, this should not happen ' + \
            'since sets should only hold unique items'

        effective_windows = [] # the arguments to func
        for window_set in can_com: # ('/path/1', '/path/2')
            effective_window = effective_window_dict[window_set]
            effective_windows.append(effective_window)

        if func_takes_pixsizemap_arg:
            template = effective_windows[-1] # FIXME: assumes this describes all maps in the combo
            pixsizemap_key = (template.shape, wcsutils.describe(template.wcs))
            pixsizemap = pixsizemap_dict[pixsizemap_key][1]
            wf = func(*effective_windows, pixsizemap)
        else:
            wf = func(*effective_windows)
        
        canonized_wfs[can_com] = wf
    
    return canonized_wfs

# get all the possible fields. we treat noise within a given field, so it is not
# included here
field_infos = []
nsplits = {}
for sv in surveys:
    nsplits[sv] = d[f'n_splits_{sv}']
    for m in d[f'arrays_{sv}']:
        for pol in ('T', 'pol'):
            field_info = (sv, m, pol)
            if field_info not in field_infos:
                field_infos.append(field_info)
            else:
                raise ValueError(f'{field_info=} is not unique')

if cov_correlation_by_noise_model:
    mapnames2noise_model_tags = dict_utils.get_mapnames_to_noise_model_tags(d)

# use mpi to speed up iteration over all the products, since this can take
# O(3 minutes) for O(31) maps 
so_mpi.init(True)

field_info_products_2pt = list(product(field_infos, repeat=2))

subtasks = so_mpi.taskrange(imin=0, imax=len(field_info_products_2pt) - 1)

log.info(f"[Rank {so_mpi.rank}] Getting canonized connected two-point combos")
canonized_connected_combos_2pt = set() # set
canonized_sn_field_info2canonized_connected_combo_2pt = {} # dict
for task in subtasks:
    f1, f2 = field_info_products_2pt[task]
    sv1, m1, pol1 = f1 
    sv2, m2, pol2 = f2

    # "n" holds the "noise correlation group" information: f1 and f2 have 
    # correlated noise only if n1 == n2
    if cov_correlation_by_noise_model:
        n1 = (sv1, mapnames2noise_model_tags[f'{sv1}_{m1}'])
        n2 = (sv2, mapnames2noise_model_tags[f'{sv2}_{m2}'])
    else:
        n1 = sv1
        n2 = sv2

    def update_connected_2pt(split1='s', split2='s'):
        ew1 = get_effective_window_tuple_from_map(d, sv1, m1, pol1, split=split1)
        ew2 = get_effective_window_tuple_from_map(d, sv2, m2, pol2, split=split2)
        can_con_com_2pt = pspipe_list.canonize_connected_2pt(ew1, ew2)
        
        canonized_connected_combos_2pt.add(can_con_com_2pt) # add item already in set to set does nothing
        
        snf1 = (sv1, m1, pol1, split1)
        snf2 = (sv2, m2, pol2, split2)
        can_sn_field_info = pspipe_list.canonize_connected_2pt(snf1, snf2)
        
        if can_sn_field_info not in canonized_sn_field_info2canonized_connected_combo_2pt:
            canonized_sn_field_info2canonized_connected_combo_2pt[can_sn_field_info] = can_con_com_2pt
        else:
            assert canonized_sn_field_info2canonized_connected_combo_2pt[can_sn_field_info] == can_con_com_2pt, \
                f'Tried to add can_sn_field_info {can_sn_field_info} pointing to ' + \
                f'canonized connected combo {can_con_com_2pt} but already ' + \
                f'points to canonized connected combo ' + \
                f'{canonized_sn_field_info2canonized_connected_combo_2pt[can_sn_field_info]}'

    # ss
    update_connected_2pt()

    # nn
    if n1 == n2:
        for k in range(nsplits[sv1]):
            update_connected_2pt(split1=f'n{k}', split2=f'n{k}')

# need to gather all the canonized combos, and the mapping from field infos to
# canonized combos, from the separate mpi tasks
#
# canonized_sn_field_info2canonized_connected_combo_2pt is saved to disk but does nothing
# else in this script, so we gather to one rank for that
canonized_connected_combos_2pt = so_mpi.gather_set_or_dict(canonized_connected_combos_2pt,
                                                           allgather=True,
                                                           overlap_allowed=True)
# do set-to-list only once
# NOTE: careful: since the allgather updated a *set*, this list is not in the 
# same order over different tasks. need to sort it to be sure
canonized_connected_combos_2pt = sorted(list(canonized_connected_combos_2pt))


canonized_sn_field_info2canonized_connected_combo_2pt = so_mpi.gather_set_or_dict(canonized_sn_field_info2canonized_connected_combo_2pt,
                                                                                  allgather=False,
                                                                                  root=0,
                                                                                  overlap_allowed=True)

if so_mpi.rank == 0:
    np.save(opj(cov_dir, 'canonized_sn_field_info2canonized_connected_combo_2pt.npy'), canonized_sn_field_info2canonized_connected_combo_2pt)

# now calculate w2s
n_w2s = len(canonized_connected_combos_2pt)
subtasks = so_mpi.taskrange(imin=0, imax=n_w2s - 1)
log.info(f"[Rank {so_mpi.rank}] Number of w2s to compute: {len(subtasks)} (out of {n_w2s} total)")

# NOTE: only load the masks needed for this task. this spends memory to save
# time, since loading to disk is very slow, but multiplying maps with parallel
# numba is very fast.
# TODO: smart task allocation to make sure no task has too much memory requirements
# TODO: evaluate trade-off of doing s and n separately (spends memory to speed
# up loading)
# TODO: if worth-it, rewrite so noise_windows are batched by "uneeded" 
# signal_windows, and those signal_windows are deleted after their "child"
# noise_windows are loaded
log.info(f"[Rank {so_mpi.rank}] Loading windows for w2 calculation")

signal_windows_to_load, noise_windows_to_load = get_signal_and_noise_windows_to_load_sets(canonized_connected_combos_2pt, subtasks)

effective_window_dict, pixsizemap_dict = load_effective_windows(signal_windows_to_load, noise_windows_to_load)

log.info(f"[Rank {so_mpi.rank}] Calculating w2s")

canonized_w2s = calc_canonized_function(canonized_connected_combos_2pt,
                                        subtasks, effective_window_dict, 
                                        pixsizemap_dict, calc_w2, 
                                        func_takes_pixsizemap_arg=True)

canonized_w2s = so_mpi.gather_set_or_dict(canonized_w2s, allgather=True,
                                          overlap_allowed=False)

if so_mpi.rank == 0:
    np.save(opj(cov_dir, 'canonized_w2s.npy'), canonized_w2s)

# use mpi to speed up iteration over all the products, since this can take
# O(3 minutes) for O(31) maps 
field_info_products_4pt = list(product(field_infos, repeat=4))

subtasks = so_mpi.taskrange(imin=0, imax=len(field_info_products_4pt) - 1)

log.info(f"[Rank {so_mpi.rank}] Getting canonized connected four-point combos")
canonized_connected_combos_4pt = set() # set
canonized_sn_field_info2canonized_connected_combo_4pt = {} # dict
for task in subtasks:
    f1, f2, f3, f4 = field_info_products_4pt[task]
    sv1, m1, pol1 = f1 
    sv2, m2, pol2 = f2
    sv3, m3, pol3 = f3 
    sv4, m4, pol4 = f4

    # "n" holds the "noise correlation group" information: f1 and f2 have 
    # correlated noise only if n1 == n2
    if cov_correlation_by_noise_model:
        n1 = (sv1, mapnames2noise_model_tags[f'{sv1}_{m1}'])
        n2 = (sv2, mapnames2noise_model_tags[f'{sv2}_{m2}'])
        n3 = (sv3, mapnames2noise_model_tags[f'{sv3}_{m3}'])
        n4 = (sv4, mapnames2noise_model_tags[f'{sv4}_{m4}'])
    else:
        n1 = sv1
        n2 = sv2
        n3 = sv3
        n4 = sv4

    def update_connected_4pt(split1='s', split2='s', split3='s', split4='s'):
        ew1 = get_effective_window_tuple_from_map(d, sv1, m1, pol1, split=split1)
        ew2 = get_effective_window_tuple_from_map(d, sv2, m2, pol2, split=split2)
        ew3 = get_effective_window_tuple_from_map(d, sv3, m3, pol3, split=split3)
        ew4 = get_effective_window_tuple_from_map(d, sv4, m4, pol4, split=split4)
        can_con_com_4pt = pspipe_list.canonize_connected_4pt(ew1, ew2, ew3, ew4)
        
        canonized_connected_combos_4pt.add(can_con_com_4pt) # add item already in set to set does nothing
        
        snf1 = (sv1, m1, pol1, split1)
        snf2 = (sv2, m2, pol2, split2)
        snf3 = (sv3, m3, pol3, split3)
        snf4 = (sv4, m4, pol4, split4)
        can_sn_field_info = pspipe_list.canonize_connected_4pt(snf1, snf2, snf3, snf4)
        
        if can_sn_field_info not in canonized_sn_field_info2canonized_connected_combo_4pt:
            canonized_sn_field_info2canonized_connected_combo_4pt[can_sn_field_info] = can_con_com_4pt
        else:
            assert canonized_sn_field_info2canonized_connected_combo_4pt[can_sn_field_info] == can_con_com_4pt, \
                f'Tried to add can_sn_field_info {can_sn_field_info} pointing to ' + \
                f'canonized connected combo {can_con_com_4pt} but already ' + \
                f'points to canonized connected combo ' + \
                f'{canonized_sn_field_info2canonized_connected_combo_4pt[can_sn_field_info]}'

    # ssss
    update_connected_4pt()

    # ssnn
    if n3 == n4:
        for j in range(nsplits[sv3]):
            update_connected_4pt(split3=f'n{j}', split4=f'n{j}')

    # nnss
    if n1 == n2:
        for k in range(nsplits[sv1]):
            update_connected_4pt(split1=f'n{k}', split2=f'n{k}')

    # nnnn
    # NOTE: this will include "auto" noise terms, but that's ok, it's a little
    # extra compute but can be used in selecting the median w4/w2w2 for a 
    # single map for the kspace correction sims
    if n1 == n2 and n3 == n4:
        for k, j in product(range(nsplits[sv1]), range(nsplits[sv3])):
            update_connected_4pt(split1=f'n{k}', split2=f'n{k}', split3=f'n{j}', split4=f'n{j}')

# need to gather all the canonized combos, and the mapping from field infos to
# canonized combos, from the separate mpi tasks
#
# canonized_sn_field_info2canonized_connected_combo_4pt is saved to disk but does nothing
# else in this script, so we gather to one rank for that
canonized_connected_combos_4pt = so_mpi.gather_set_or_dict(canonized_connected_combos_4pt,
                                                           allgather=True,
                                                           overlap_allowed=True)

# do set-to-list only once
# NOTE: careful: since the allgather updated a *set*, this list is not in the 
# same order over different tasks. need to sort it to be sure
canonized_connected_combos_4pt = sorted(list(canonized_connected_combos_4pt))

canonized_sn_field_info2canonized_connected_combo_4pt = so_mpi.gather_set_or_dict(canonized_sn_field_info2canonized_connected_combo_4pt,
                                                                                  allgather=False,
                                                                                  root=0,
                                                                                  overlap_allowed=True)

if so_mpi.rank == 0:
    np.save(opj(cov_dir, 'canonized_sn_field_info2canonized_connected_combo_4pt.npy'), canonized_sn_field_info2canonized_connected_combo_4pt)

# now calculate w4s. based on w4/w2w2s and any approximations (using median
# couplings), we may trim the list of disconnected pairs for calculating actual
# window spectra
n_w4s = len(canonized_connected_combos_4pt)
subtasks = so_mpi.taskrange(imin=0, imax=n_w4s - 1)
log.info(f"[Rank {so_mpi.rank}] Number of w4s to compute: {len(subtasks)} (out of {n_w4s} total)")

# NOTE: only load the masks needed for this task. this spends memory to save
# time, since loading to disk is very slow, but multiplying maps with parallel
# numba is very fast.
# TODO: smart task allocation to make sure no task has too much memory requirements
# TODO: evaluate trade-off of doing s and n separately (spends memory to speed
# up loading)
# TODO: if worth-it, rewrite so noise_windows are batched by "uneeded" 
# signal_windows, and those signal_windows are deleted after their "child"
# noise_windows are loaded
log.info(f"[Rank {so_mpi.rank}] Loading windows for w4 calculation")

signal_windows_to_load, noise_windows_to_load = get_signal_and_noise_windows_to_load_sets(canonized_connected_combos_4pt, subtasks)

# delete windows from effective_window_dict if they are not needed, but keep the
# needed ones to avoid repeat loading from disk. also, delete from 
# signal_windows_to_load and noise_windows_to_load those items already in 
# effective_window_dict
window_sets_to_delete_from_effective_window_dict = []
for window_set in effective_window_dict:
    if window_set not in signal_windows_to_load | noise_windows_to_load:
        window_sets_to_delete_from_effective_window_dict.append(window_set)

for window_set in window_sets_to_delete_from_effective_window_dict:
    del effective_window_dict[window_set]

for windows_to_load in (signal_windows_to_load, noise_windows_to_load): 
    windows_to_load -= windows_to_load & effective_window_dict.keys()

# NOTE: we reuse existing pixsizemap_dict and effective_window_dict to avoid
# repeat loading from disk (we don't care if there are unneeded items in 
# pixsizemap_dict because there are so few overall)
load_effective_windows(signal_windows_to_load, noise_windows_to_load, 
                       effective_window_dict=effective_window_dict,
                       pixsizemap_dict=pixsizemap_dict)

log.info(f"[Rank {so_mpi.rank}] Calculating w4s")

canonized_w4s = calc_canonized_function(canonized_connected_combos_4pt,
                                        subtasks, effective_window_dict, 
                                        pixsizemap_dict, calc_w4, 
                                        func_takes_pixsizemap_arg=True)

canonized_w4s = so_mpi.gather_set_or_dict(canonized_w4s, allgather=True,
                                          overlap_allowed=False)

if so_mpi.rank == 0:
    np.save(opj(cov_dir, 'canonized_w4s.npy'), canonized_w4s)

# now calculate needed window spectra based on disconnected 4pt combos. this may
# be less than all the canonized disconnected 4pt combos based on approximations
# such as only calculating a representative coupling per covariance block etc.
# we may also need to load additional windows because the mpi task distribution
# over connected 4pt combos and disconnected 4pt combos has no relationship,
# TODO: fix this
# so we first find which windows we need, delete the ones in memory that we
# don't need, then add the new ones.
# 
# then we do the computation of the window spectra
subtasks = so_mpi.taskrange(imin=0, imax=len(field_info_products_4pt) - 1)

log.info(f"[Rank {so_mpi.rank}] Getting canonized disconnected four-point combos")
canonized_disconnected_combos_4pt = set() # set
canonized_sn_field_info2canonized_disconnected_combo_4pt = {} # dict
reference_sn_field_info2reference_canonized_disconnected_combo_4pt = {} # dict
for task in subtasks:
    f1, f2, f3, f4 = field_info_products_4pt[task]
    sv1, m1, pol1 = f1 
    sv2, m2, pol2 = f2
    sv3, m3, pol3 = f3 
    sv4, m4, pol4 = f4

    # "n" holds the "noise correlation group" information: f1 and f2 have 
    # correlated noise only if n1 == n2
    if cov_correlation_by_noise_model:
        n1 = (sv1, mapnames2noise_model_tags[f'{sv1}_{m1}'])
        n2 = (sv2, mapnames2noise_model_tags[f'{sv2}_{m2}'])
        n3 = (sv3, mapnames2noise_model_tags[f'{sv3}_{m3}'])
        n4 = (sv4, mapnames2noise_model_tags[f'{sv4}_{m4}'])
    else:
        n1 = sv1
        n2 = sv2
        n3 = sv3
        n4 = sv4

    # if we are using the representative coupling per covariance block
    # approximations, get the reference disconnected 4pt combo for this
    # disconnected 4pt combo
    reference_term2reference_can_discon_com_4pt = {}

    splits2sstr = lambda k, j: 's'
    splits2kstr = lambda k, j: f'n{k}'
    splits2jstr = lambda k, j: f'n{j}'
    for term in ('ssnn', 'nnss', 'nnnn'):
        use_median_coupling_per_block = False
        if term == 'ssnn' and n3 == n4 and cov_use_median_ssnn_nnss_coupling_per_block:
            use_median_coupling_per_block = True
            splits_iterator = zip([None] * nsplits[sv3], range(nsplits[sv3]))            
            split1_func = splits2sstr
            split2_func = splits2sstr
            split3_func = splits2jstr
            split4_func = splits2jstr
        elif term == 'nnss' and n1 == n2 and cov_use_median_ssnn_nnss_coupling_per_block:
            use_median_coupling_per_block = True
            splits_iterator = zip(range(nsplits[sv1]), [None] * nsplits[sv1])            
            split1_func = splits2kstr
            split2_func = splits2kstr
            split3_func = splits2sstr
            split4_func = splits2sstr
        elif term == 'nnnn' and n1 == n2 and n3 == n4 and cov_use_median_nnnn_coupling_per_block:
            use_median_coupling_per_block = True
            # NOTE: exclude auto terms this time because they could distort the 
            # median window spectrum for the actual nnnn term of this covariance
            # block
            splits_iterator = pspipe_list.get_splits_cross_iterator(sv1, nsplits[sv1], sv3, nsplits[sv3])       
            split1_func = splits2kstr
            split2_func = splits2kstr
            split3_func = splits2jstr
            split4_func = splits2jstr

        if use_median_coupling_per_block: 
            can_discon_com_4pts = []
            w4_w2_w2s = []           
            for k, j in splits_iterator:
                ew1 = get_effective_window_tuple_from_map(d, sv1, m1, pol1, split=split1_func(k, j))
                ew2 = get_effective_window_tuple_from_map(d, sv2, m2, pol2, split=split2_func(k, j))
                ew3 = get_effective_window_tuple_from_map(d, sv3, m3, pol3, split=split3_func(k, j))
                ew4 = get_effective_window_tuple_from_map(d, sv4, m4, pol4, split=split4_func(k, j))
            
                can_discon_com_4pt = pspipe_list.canonize_disconnected_4pt(ew1, ew2, ew3, ew4)
                can_discon_com_4pts.append(can_discon_com_4pt)

                can_con_com_2pt_12 = pspipe_list.canonize_connected_2pt(ew1, ew2)
                can_con_com_2pt_34 = pspipe_list.canonize_connected_2pt(ew3, ew4)
                can_con_com_4pt = pspipe_list.canonize_connected_4pt(ew1, ew2, ew3, ew4)
                w4_w2_w2 = canonized_w4s[can_con_com_4pt] / canonized_w2s[can_con_com_2pt_12] / canonized_w2s[can_con_com_2pt_34]
                w4_w2_w2s.append(w4_w2_w2)

            # for even number of terms, pick the term just above the median (as
            # opposed to just below, to be conservative -- i.e., slightly
            # stronger off-diagonals)
            reference_idx = w4_w2_w2s.index(np.quantile(w4_w2_w2s, 0.5, method='higher'))
            reference_can_discon_com_4pt = can_discon_com_4pts[reference_idx]
            reference_term2reference_can_discon_com_4pt[term] = reference_can_discon_com_4pt
    
    def update_disconnected_4pt(split1='s', split2='s', split3='s', split4='s'):
        term = covariance.get_4pt_sn_term_type(split1, split2, split3, split4)
        ref_split1, ref_split2, ref_split3, ref_split4 = term
        
        # if we are going to use the reference coupling for this split term, then
        # the block ordering actually matters (since specific per-split combos
        # with the same can_sn_field_info may appear in different blocks and thus
        # point to different references), but the split index does not (ie, only
        # the term reference label matters)
        ref_snf1 = (sv1, m1, pol1, ref_split1)
        ref_snf2 = (sv2, m2, pol2, ref_split2)
        ref_snf3 = (sv3, m3, pol3, ref_split3)
        ref_snf4 = (sv4, m4, pol4, ref_split4)
        ref_sn_field_info = (ref_snf1, ref_snf2, ref_snf3, ref_snf4)

        # if we are going to calculate the actual coupling for these 4 fields,
        # then only their can_discon_4pt field info matters
        snf1 = (sv1, m1, pol1, split1)
        snf2 = (sv2, m2, pol2, split2)
        snf3 = (sv3, m3, pol3, split3)
        snf4 = (sv4, m4, pol4, split4)
        can_sn_field_info = pspipe_list.canonize_disconnected_4pt(snf1, snf2, snf3, snf4)

        reference_can_discon_com_4pt = reference_term2reference_can_discon_com_4pt.get(term)
        if reference_can_discon_com_4pt is not None:
            can_discon_com_4pt = reference_can_discon_com_4pt

            target_sn_field_info = ref_sn_field_info
            target_sn_field_info2canonized_disconnected_combo_4pt = reference_sn_field_info2reference_canonized_disconnected_combo_4pt
            
            interloper_sn_field_info = can_sn_field_info
            interloper_sn_field_info2canonized_disconnected_combo_4pt = canonized_sn_field_info2canonized_disconnected_combo_4pt
        else:
            ew1 = get_effective_window_tuple_from_map(d, sv1, m1, pol1, split=split1)
            ew2 = get_effective_window_tuple_from_map(d, sv2, m2, pol2, split=split2)
            ew3 = get_effective_window_tuple_from_map(d, sv3, m3, pol3, split=split3)
            ew4 = get_effective_window_tuple_from_map(d, sv4, m4, pol4, split=split4)
            can_discon_com_4pt = pspipe_list.canonize_disconnected_4pt(ew1, ew2, ew3, ew4)

            target_sn_field_info = can_sn_field_info
            target_sn_field_info2canonized_disconnected_combo_4pt = canonized_sn_field_info2canonized_disconnected_combo_4pt

            interloper_sn_field_info = ref_sn_field_info
            interloper_sn_field_info2canonized_disconnected_combo_4pt = reference_sn_field_info2reference_canonized_disconnected_combo_4pt

        canonized_disconnected_combos_4pt.add(can_discon_com_4pt) # add item already in set to set does nothing

        assert interloper_sn_field_info not in interloper_sn_field_info2canonized_disconnected_combo_4pt, \
            f'Want to add target_sn_field_info {target_sn_field_info} pointing to ' + \
            f'canonized disconnected combo {can_discon_com_4pt} to ' + \
            'target_sn_field_info2canonized_disconnected_combo_4pt, but this ordered combo ' + \
            f'may also be accessed as {interloper_sn_field_info} which is already in ' + \
            f'interloper_sn_field_info2canonized_disconnected_combo_4pt and points ' + \
            f'to canonized disconnected combo ' + \
            f'{interloper_sn_field_info2canonized_disconnected_combo_4pt[interloper_sn_field_info]}'            

        if target_sn_field_info not in target_sn_field_info2canonized_disconnected_combo_4pt:
            target_sn_field_info2canonized_disconnected_combo_4pt[target_sn_field_info] = can_discon_com_4pt
        else:
            assert target_sn_field_info2canonized_disconnected_combo_4pt[target_sn_field_info] == can_discon_com_4pt, \
                f'Tried to add target_sn_field_info {target_sn_field_info} pointing to ' + \
                f'canonized disconnected combo {can_discon_com_4pt} to ' + \
                'target_sn_field_info2canonized_disconnected_combo_4pt but already ' + \
                f'points to canonized disconnected combo ' + \
                f'{target_sn_field_info2canonized_disconnected_combo_4pt[target_sn_field_info]}'

    # ssss
    update_disconnected_4pt()

    # ssnn
    if n3 == n4:
        for j in range(nsplits[sv3]):
            update_disconnected_4pt(split3=f'n{j}', split4=f'n{j}')

    # nnss
    if n1 == n2:
        for k in range(nsplits[sv1]):
            update_disconnected_4pt(split1=f'n{k}', split2=f'n{k}')

    # nnnn
    # NOTE: exclude auto terms this time because, even though we may use one of 
    # them for the kspace correction sims, all the others (for which there would
    # be many) would be wasted computation
    if n1 == n2 and n3 == n4:
        for k, j in pspipe_list.get_splits_cross_iterator(sv1, nsplits[sv1], sv3, nsplits[sv3]):
            update_disconnected_4pt(split1=f'n{k}', split2=f'n{k}', split3=f'n{j}', split4=f'n{j}')

# need to gather all the canonized combos, and the mapping from field infos to
# canonized combos, from the separate mpi tasks. we expect many repeated 
# canonized_disconnected_combos_4pt and canonized_sn_field_infos, but keys of
# reference_sn_field_info2reference_canonized_disconnected_combo_4pt should be
# unique over tasks because they aren't canonized
#
# sn_field_info2canonized_disconnected_combo_4pt dicts are saved to disk but do
# nothing else in this script, so we gather to one rank for that
canonized_disconnected_combos_4pt = so_mpi.gather_set_or_dict(canonized_disconnected_combos_4pt,
                                                              allgather=True,
                                                              overlap_allowed=True)

# do set-to-list only once
# NOTE: careful: since the allgather updated a *set*, this list is not in the 
# same order over different tasks. need to sort it to be sure
canonized_disconnected_combos_4pt = sorted(list(canonized_disconnected_combos_4pt))

canonized_sn_field_info2canonized_disconnected_combo_4pt = so_mpi.gather_set_or_dict(canonized_sn_field_info2canonized_disconnected_combo_4pt,
                                                                                     allgather=False,
                                                                                     root=0,
                                                                                     overlap_allowed=True)

reference_sn_field_info2reference_canonized_disconnected_combo_4pt = so_mpi.gather_set_or_dict(reference_sn_field_info2reference_canonized_disconnected_combo_4pt,
                                                                                               allgather=False,
                                                                                               root=0,
                                                                                               overlap_allowed=False)

if so_mpi.rank == 0:
    np.save(opj(cov_dir, 'canonized_sn_field_info2canonized_disconnected_combo_4pt.npy'), canonized_sn_field_info2canonized_disconnected_combo_4pt)
    np.save(opj(cov_dir, 'reference_sn_field_info2reference_canonized_disconnected_combo_4pt.npy'), reference_sn_field_info2reference_canonized_disconnected_combo_4pt)

# now calculate window_spectra
n_wls = len(canonized_disconnected_combos_4pt)
subtasks = so_mpi.taskrange(imin=0, imax=n_wls - 1)
log.info(f"[Rank {so_mpi.rank}] Number of wls to compute: {len(subtasks)} (out of {n_wls} total)")

# NOTE: only load the masks needed for this task. this spends memory to save
# time, since loading to disk is very slow, but multiplying maps with parallel
# numba is very fast.
# TODO: smart task allocation to make sure no task has too much memory requirements
# TODO: evaluate trade-off of doing s and n separately (spends memory to speed
# up loading)
# TODO: if worth-it, rewrite so noise_windows are batched by "uneeded" 
# signal_windows, and those signal_windows are deleted after their "child"
# noise_windows are loaded
log.info(f"[Rank {so_mpi.rank}] Loading windows for window spectra calculation")

signal_windows_to_load, noise_windows_to_load = get_signal_and_noise_windows_to_load_sets(canonized_disconnected_combos_4pt, subtasks)

# delete windows from effective_window_dict if they are not needed, but keep the
# needed ones to avoid repeat loading from disk. also, delete from 
# signal_windows_to_load and noise_windows_to_load those items already in 
# effective_window_dict
window_sets_to_delete_from_effective_window_dict = []
for window_set in effective_window_dict:
    if window_set not in signal_windows_to_load | noise_windows_to_load:
        window_sets_to_delete_from_effective_window_dict.append(window_set)

for window_set in window_sets_to_delete_from_effective_window_dict:
    del effective_window_dict[window_set]

for windows_to_load in (signal_windows_to_load, noise_windows_to_load): 
    windows_to_load -= windows_to_load & effective_window_dict.keys()

# NOTE: we reuse existing pixsizemap_dict and effective_window_dict to avoid
# repeat loading from disk (we don't care if there are unneeded items in 
# pixsizemap_dict because there are so few overall)
load_effective_windows(signal_windows_to_load, noise_windows_to_load, 
                       effective_window_dict=effective_window_dict,
                       pixsizemap_dict=pixsizemap_dict)

t0 = time.time()
log.info(f"[Rank {so_mpi.rank}] Calculating window spectra")

canonized_wls = calc_canonized_function(canonized_disconnected_combos_4pt,
                                        subtasks, effective_window_dict, 
                                        pixsizemap_dict, calc_wl, 
                                        func_takes_pixsizemap_arg=False)

log.info(f"[Rank {so_mpi.rank}] Calculated {len(subtasks)} spectra in {time.time() - t0} seconds")

canonized_wls = so_mpi.gather_set_or_dict(canonized_wls, allgather=False, root=0,
                                          overlap_allowed=False)

if so_mpi.rank == 0:
    np.save(opj(cov_dir, 'canonized_wls.npy'), canonized_wls)