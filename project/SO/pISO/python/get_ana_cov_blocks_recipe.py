description = """This is the main script for the analytic covariance computation.
Almost all the complexity is in the book-keeping. Unlike the DR6 covariance
script, this explicitly loops over the split-based sums. It is also much faster,
thanks to ducc and numba. Unlike DR6, we account for anisotropy in the noise
which is necessary to achieve percent-level accuracy in the analytic covariance.
"""

import argparse
from os.path import join as opj
from itertools import product
import time

import numpy as npy

from pspipe_utils import log, pspipe_list, covariance, dict_utils
from pspy import pspy_utils, so_dict, so_mpi

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--coupling-cache-size', type=int, default=32,
                    help='The maximum number of couplings to be calculated and '
                    'stored at one time')
args = parser.parse_args()

# get needed info from paramfile
d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

coupling_cache_size = args.coupling_cache_size

surveys = d["surveys"]
cov_correlation_by_noise_model = d['cov_correlation_by_noise_model']
cov_spin00_coupling_only = d['cov_spin00_coupling_only']

cov_dir = d['cov_dir']
plot_dir = opj(d['plots_dir'], 'covariances')
pspy_utils.create_directory(plot_dir)

canonized_sn_field_info2canonized_disconnected_combo_4pt = npy.load(opj(cov_dir, 'canonized_sn_field_info2canonized_disconnected_combo_4pt.npy'), allow_pickle=True).item()
reference_sn_field_info2reference_canonized_disconnected_combo_4pt = npy.load(opj(cov_dir, 'reference_sn_field_info2reference_canonized_disconnected_combo_4pt.npy'), allow_pickle=True).item()

def update_ducc_inputs_and_nterms(sna1, sna2, sna3, sna4,
                                  this_block_can_discon_com_4pts_and_optypes,
                                  can_sn_alm_info2nterms):
    # update ducc inputs with minimal unique couplings, and track their order
    sv1, m1, TEB1, split1 = sna1
    sv2, m2, TEB2, split2 = sna2
    sv3, m3, TEB3, split3 = sna3
    sv4, m4, TEB4, split4 = sna4

    pol1 = covariance.TEB2pol(TEB1)
    pol2 = covariance.TEB2pol(TEB2)
    pol3 = covariance.TEB2pol(TEB3)
    pol4 = covariance.TEB2pol(TEB4)

    snf1 = (sv1, m1, pol1, split1)
    snf2 = (sv2, m2, pol2, split2)
    snf3 = (sv3, m3, pol3, split3)
    snf4 = (sv4, m4, pol4, split4)
    can_discon_com_4pt = covariance.get_can_discon_com_4pt(
        snf1, snf2, snf3, snf4,
        canonized_sn_field_info2canonized_disconnected_combo_4pt,
        reference_sn_field_info2reference_canonized_disconnected_combo_4pt
        )

    # NOTE: although using uncanonized pol1, pol2, pol3, pol4, the optype is
    # insensitive to disconnected 4pt canonization
    optype = covariance.pols_disconnected_combo_4pt2ducc_optype(
        pol1, pol2, pol3, pol4, cov_spin00_coupling_only
        )
    
    # adding to set does nothing if already in set
    this_block_can_discon_com_4pts_and_optypes.add((can_discon_com_4pt, optype))

    # track the number of times this term has appeared in this cov TEB sub-block
    can_sn_alm_info = pspipe_list.canonize_disconnected_4pt(sna1, sna2, sna3, sna4)
    if can_sn_alm_info not in can_sn_alm_info2nterms:
        can_sn_alm_info2nterms[can_sn_alm_info] = 1
    else:
        can_sn_alm_info2nterms[can_sn_alm_info] += 1

# NOTE: unlike for w2s, w4s, and window spectra, here we mpi over blocks at the
# map 4pt level (i.e., not including T and pol). We handle both noise *and* T
# and pol within a block. This is because we don't want to save any large
# ellxell intermediate products to disk; rather, we want the entire
# 9xell x 9xell pseudocov block that we can immediately sandwich between two
# pseudo2datavec operators.
n_covs, ni_list, nj_list, np_list, nq_list = pspipe_list.get_covariances_list(d)
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if cov_correlation_by_noise_model:
    mapnames2noise_model_tags = dict_utils.get_mapnames_to_noise_model_tags(d)

nsplits = {}
for sv in surveys:
    nsplits[sv] = d[f'n_splits_{sv}']

# first, figure out all the "shared couplings" sets of cov blocks, such that the
# total number of couplings in each block is <= the cache size. NOTE: a nominal
# cov_block_set might get "chopped" by blindly cutting all the cov blocks into
# equal-length subtasks
so_mpi.init(True)

t0 = time.time()

subtasks = so_mpi.taskrange(imin=0, imax=n_covs - 1)

cov_block_sets2can_discon_com_4pts_and_optypes = {}
cov_block2TEB_block2can_sn_alm_info2nterms = {}

# need to initialize objects before while loop, that otherwise are re-initialized
# in the loop
cov_block_set = []
can_discon_com_4pts_and_optypes = set()
i = 0
while True:
    task = subtasks[i]
    svi, mi = ni_list[task].split('&') 
    svj, mj = nj_list[task].split('&')
    svp, mp = np_list[task].split('&')
    svq, mq = nq_list[task].split('&')
    cov_block = ((svi, mi), (svj, mj),
                 (svp, mp), (svq, mq))

    # "n" holds the "noise correlation group" information: f1 and f2 have 
    # correlated noise only if ni == nj
    if cov_correlation_by_noise_model:
        ni = (svi, mapnames2noise_model_tags[f'{svi}_{mi}'])
        nj = (svj, mapnames2noise_model_tags[f'{svj}_{mj}'])
        np = (svp, mapnames2noise_model_tags[f'{svp}_{mp}'])
        nq = (svq, mapnames2noise_model_tags[f'{svq}_{mq}'])
    else:
        ni = svi
        nj = svj
        np = svp
        nq = svq

    # we need figure out which couplings we actually need first
    #
    # for each block, see which unique couplings are needed, and then try to add
    # to existing block set of unique couplings. if resulting merged set fits in
    # the cache, go to the next block, otherwise, end set and redo this block
    this_block_can_discon_com_4pts_and_optypes = set()

    # for each cov TEB sub-block, tracks how many times a canonical 4pt combo
    # of (sv, m, TEB, split)s recurs, so it can be added once (times this count)
    # rather than each time
    TEB_block2can_sn_alm_info2nterms = {} # "alm_info" since keys are TEB instead of T and pol
    
    splits_cross_iterator_ij = pspipe_list.get_splits_cross_iterator(svi, nsplits[svi], svj, nsplits[svj])
    splits_cross_iterator_pq = pspipe_list.get_splits_cross_iterator(svp, nsplits[svp], svq, nsplits[svq])        
    for (TEBi, TEBj), (TEBp, TEBq) in product(spectra, repeat=2):    
        if (TEBi, TEBj, TEBp, TEBq) not in TEB_block2can_sn_alm_info2nterms:
            TEB_block2can_sn_alm_info2nterms[TEBi, TEBj, TEBp, TEBq] = {}
        
        can_sn_alm_info2nterms = TEB_block2can_sn_alm_info2nterms[TEBi, TEBj, TEBp, TEBq]

        for (si, sj), (sp, sq) in product(splits_cross_iterator_ij, splits_cross_iterator_pq):

            # ssss ipjq
            update_ducc_inputs_and_nterms((svi, mi, TEBi, 's'), (svp, mp, TEBp, 's'),
                                          (svj, mj, TEBj, 's'), (svq, mq, TEBq, 's'), 
                                          this_block_can_discon_com_4pts_and_optypes,
                                          can_sn_alm_info2nterms)
            
            # ssnn ipjq
            if nj == nq and sj == sq:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, 's'), (svp, mp, TEBp, 's'),
                                              (svj, mj, TEBj, f'n{sj}'), (svq, mq, TEBq, f'n{sj}'), 
                                              this_block_can_discon_com_4pts_and_optypes,
                                              can_sn_alm_info2nterms)

            # nnss ipjq
            if ni == np and si == sp:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, f'n{si}'), (svp, mp, TEBp, f'n{si}'),
                                              (svj, mj, TEBj, 's'), (svq, mq, TEBq, 's'), 
                                              this_block_can_discon_com_4pts_and_optypes,
                                              can_sn_alm_info2nterms)
                    
            # nnnn ipjq
            if ni == np and si == sp and nj == nq and sj == sq:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, f'n{si}'), (svp, mp, TEBp, f'n{si}'),
                                              (svj, mj, TEBj, f'n{sj}'), (svq, mq, TEBq, f'n{sj}'), 
                                              this_block_can_discon_com_4pts_and_optypes,
                                              can_sn_alm_info2nterms)
                    
            # ssss iqjp
            update_ducc_inputs_and_nterms((svi, mi, TEBi, 's'), (svq, mq, TEBq, 's'),
                                          (svj, mj, TEBj, 's'), (svp, mp, TEBp, 's'), 
                                          this_block_can_discon_com_4pts_and_optypes,
                                          can_sn_alm_info2nterms)
            
            # ssnn iqjp
            if nj == np and sj == sp:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, 's'), (svq, mq, TEBq, 's'),
                                              (svj, mj, TEBj, f'n{sj}'), (svp, mp, TEBp, f'n{sj}'), 
                                              this_block_can_discon_com_4pts_and_optypes,
                                              can_sn_alm_info2nterms)

            # nnss iqjp
            if ni == nq and si == sq:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, f'n{si}'), (svq, mq, TEBq, f'n{si}'),
                                              (svj, mj, TEBj, 's'), (svp, mp, TEBp, 's'), 
                                              this_block_can_discon_com_4pts_and_optypes,
                                              can_sn_alm_info2nterms)
                    
            # nnnn iqjp
            if ni == nq and si == sq and nj == np and sj == sp:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, f'n{si}'), (svq, mq, TEBq, f'n{si}'),
                                              (svj, mj, TEBj, f'n{sj}'), (svp, mp, TEBp, f'n{sj}'), 
                                              this_block_can_discon_com_4pts_and_optypes,
                                              can_sn_alm_info2nterms)

    cov_block2TEB_block2can_sn_alm_info2nterms[cov_block] = TEB_block2can_sn_alm_info2nterms

    # there are now four possibilities for what to do with this block:
    # (a) if the current block requires more couplings than the cache size 
    # limit, and the current cache is empty, then we have no recourse: 
    # ending the set, resetting the cache, and redoing the block will of course
    # never work. therefore, we first force the one block into the cache, and
    # then end the set and reset the cache. we then go on to the next block.
    # this *does* "violate" the cache limit, so we issue a warning
    # (b) like (a), if adding the current block's couplings to the cache would
    # result in a cache size more than the cache size limit, but unlike (a) if 
    # the cache is not empty, we do have a recourse: end the set, reset the
    # cache, and then redo this block with an empty cache. 
    # (c) if we are on the last block of all the subtasks, but we know we are 
    # not going to redo this block with an empty cache (i.e., not (b)), then
    # we are also on the last task of the loop. like (a) we must force the block
    # into the cache and end the set. unlike (a), we break the loop instead of
    # going on to the next block. it's possible that (a) and (c) occur at the 
    # same time, in which case (c) takes priority.
    # (d) otherwise proceed: add this block to the current cache and go on to 
    # the next block. hopefully this happens most of the time
    single_block_set = False
    end_set_and_redo_block = False
    if len(can_discon_com_4pts_and_optypes | this_block_can_discon_com_4pts_and_optypes) > coupling_cache_size:
        single_block_set = len(can_discon_com_4pts_and_optypes) == 0
        end_set_and_redo_block = len(can_discon_com_4pts_and_optypes) > 0

    end_loop = (i+1 == len(subtasks)) and not end_set_and_redo_block

    if single_block_set:
        log.warning(f"[Rank {so_mpi.rank}, Task {task}] Number of couplings for cov block {cov_block} is "
                    f"{len(this_block_can_discon_com_4pts_and_optypes)}, which exceeds the coupling cache "
                    f"size of {coupling_cache_size}. Adding to single-block-set, may result in OOM later.")

    if single_block_set or end_loop:
        cov_block_set.append(cov_block)
        can_discon_com_4pts_and_optypes.update(this_block_can_discon_com_4pts_and_optypes)

    if single_block_set or end_set_and_redo_block or end_loop:
        cov_block_sets2can_discon_com_4pts_and_optypes[tuple(cov_block_set)] = can_discon_com_4pts_and_optypes

        cov_block_set = []
        can_discon_com_4pts_and_optypes = set()
        
        if single_block_set and not end_loop:
            i += 1
            continue
        if end_set_and_redo_block:
            continue
        if end_loop:
            break
    else:
        cov_block_set.append(cov_block)
        can_discon_com_4pts_and_optypes.update(this_block_can_discon_com_4pts_and_optypes)
        i += 1

log.info(f'[Rank {so_mpi.rank}] Loop over cov block sets in {(time.time() - t0):.3f} seconds')

t0 = time.time()

cov_block_sets2can_discon_com_4pts_and_optypes = so_mpi.gather_set_or_dict(cov_block_sets2can_discon_com_4pts_and_optypes,
                                                                           allgather=False,
                                                                           root=0,
                                                                           overlap_allowed=False)

cov_block2TEB_block2can_sn_alm_info2nterms = so_mpi.gather_set_or_dict(cov_block2TEB_block2can_sn_alm_info2nterms,
                                                                       allgather=False,
                                                                       root=0,
                                                                       overlap_allowed=False)

if so_mpi.rank == 0:
    npy.save(opj(cov_dir, 'cov_block_sets2can_discon_com_4pts_and_optypes.npy'), cov_block_sets2can_discon_com_4pts_and_optypes)
    npy.save(opj(cov_dir, 'cov_block2TEB_block2can_sn_alm_info2nterms.npy'), cov_block2TEB_block2can_sn_alm_info2nterms)

log.info(f'[Rank {so_mpi.rank}] Save cov block sets in {(time.time() - t0):.3f} seconds')

# make some useful plots
if so_mpi.rank == 0:
    import matplotlib.pyplot as plt

    num_calculated_couplings_per_set = []
    mem_per_set = []
    num_cov_blocks_per_set = []
    for cov_block_sets, can_discon_com_4pts_and_optypes in cov_block_sets2can_discon_com_4pts_and_optypes.items():
        num_calculated_couplings_per_set.append(len(can_discon_com_4pts_and_optypes))

        # FIXME: get actual lmaxs
        lmax1, lmax2 = d['lmax'], d['lmax']
        mem = len(can_discon_com_4pts_and_optypes) * (lmax1+1)*(lmax2+1) * 4
        mem += 81 * (lmax1+1)*(lmax2+1) * 4
        mem /= 1e9
        mem_per_set.append(mem)

        num_cov_blocks_per_set.append(len(cov_block_sets))
    num_calculated_couplings_per_set = npy.array(num_calculated_couplings_per_set)
    mem_per_set = npy.array(mem_per_set)
    num_cov_blocks_per_set = npy.array(num_cov_blocks_per_set)

    fig, ax1 = plt.subplots()
    ax1.hist(mem_per_set, histtype='step', bins=30, color='C0')
    ax1.semilogy()
    mean = npy.mean(mem_per_set)
    ax1.axvline(mean, linestyle='--', color='C0', label = f'Mean memory (GB) per cov block set: {mean:0.3f}')
    ax1.tick_params(axis='x', color='C0', labelcolor='C0')
    ax1.set_xlabel('Memory (GB) per cov block set')
    ax1.set_ylabel('Number of cov block sets')

    ax2 = ax1.twiny()
    ax2.hist(num_calculated_couplings_per_set, histtype='step', bins=30, color='C1')
    ax2.semilogy()
    mean = npy.mean(num_calculated_couplings_per_set)
    ax2.axvline(mean, linestyle='--', color='C1', label = f'Mean number of couplings per cov block set: {mean:0.3f}')
    ax2.tick_params(axis='x', color='C1', labelcolor='C1')
    ax2.set_xlabel('Number of couplings per cov block set')
    ax2.set_ylabel('Number of cov block sets')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines2 + lines1, labels2 + labels1)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.set_facecolor('none')
    plt.savefig(opj(plot_dir, 'num_couplings_and_mem_per_cov_block_set.png'))
    plt.close()

    num_calculated_couplings_per_cov_block = num_calculated_couplings_per_set / num_cov_blocks_per_set
    weights = num_cov_blocks_per_set

    _ = plt.hist(num_calculated_couplings_per_cov_block, histtype='step', bins=30, weights=weights)
    plt.semilogy()
    mean = npy.mean(num_calculated_couplings_per_cov_block * weights) / npy.mean(weights)
    plt.axvline(mean, linestyle='--', color='C0', label = f'Mean couplings per cov block: {mean:0.3f}')
    plt.legend()
    plt.xlabel('Number of couplings per cov block')
    plt.ylabel('Number of cov blocks')
    plt.savefig(opj(plot_dir, 'num_couplings_per_cov_block.png'))
    plt.close()

    nterms = []
    for cov_block, TEB_block2can_sn_alm_info2nterms in cov_block2TEB_block2can_sn_alm_info2nterms.items():
        nt = 0
        for TEB_block, can_sn_alm_info2nterms in TEB_block2can_sn_alm_info2nterms.items():
            nt += len(can_sn_alm_info2nterms)
        nterms.append(nt)

    _ = plt.hist(nterms, histtype='step', bins=30)
    plt.semilogy()
    mean = npy.mean(nterms)
    plt.axvline(mean, linestyle='--', color='C0', label = f'Mean added terms per cov block: {mean:0.3f}')
    plt.legend()
    plt.xlabel('Number of added terms per cov block')
    plt.ylabel('Number of cov blocks')
    plt.savefig(opj(plot_dir, 'num_added_terms_per_cov_block.png'))
    plt.close()