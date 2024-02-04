"""
The script get_split_averaged_unbinned_pseudo_cov_blocks.py takes our
ingredients to construct covariance blocks. To reduce space, it averages over
split crosses, so each "block" it produces is a canonical combo of 4
(survey, array, channel, pol) fields.

The averaging over split crosses is over all split1!=split2 crosses. E.g., for
4 splits, there are 12 such crosses, so the covariance needs to average over
144 per-split blocks.

This script reads the files that will be loaded for each block in 
get_split_averaged_unbinned_pseudo_cov_blocks.py, and determines and efficient
grouping of blocks such that each group shares a lot of the same files. That
way, when get_split_averaged_unbinned_pseudo_cov_blocks.py is run, it can do
minimal I/O.
"""
import sys
import numpy as np
from pspipe_utils import log, covariance as psc
from pspy import so_dict, pspy_utils
from itertools import product, permutations
import os
import matplotlib.pyplot as plt

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

target_ngroups = int(sys.argv[2])

couplings_dir = d['couplings_dir']
covariances_dir = d['covariances_dir']
plot_dir = os.path.join(d['plot_dir'], 'covariances')
pspy_utils.create_directory(covariances_dir)
pspy_utils.create_directory(plot_dir)

surveys = d['surveys']
arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}

# format:
# - unroll all 'fields' i.e. (survey x array x chan x pol) is a 'field'
# - any given combination is then ('field'x'field' X 'field'x'field x 'spintype')
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
sv_ar_chans = [] # necessary for indexing signal model
coadd_infos = [] # no splits, can't think of a better name
ewin_infos = []
for sv1 in surveys:
    for ar1 in arrays[sv1]:
        for chan1 in arrays[sv1][ar1]:
            sv_ar_chans.append((sv1, ar1, chan1)) 
            for split1 in range(len(d[f'maps_{sv1}_{ar1}_{chan1}'])):
                for pol1 in ('T', 'E', 'B'):
                    field_info = (sv1, ar1, chan1, split1, pol1)
                    
                    coadd_info = (sv1, ar1, chan1, pol1)
                    if coadd_info not in coadd_infos:
                        coadd_infos.append(coadd_info)
                    else:
                        pass # coadd_infos are not unique because of splits
                    
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
    ('T', 'T', 'T', 'T'): '00',

    ('T', 'T', 'T', 'P'): '00',
    ('T', 'T', 'P', 'T'): '00',
    ('T', 'P', 'T', 'T'): '00',
    ('P', 'T', 'T', 'T'): '00',

    ('T', 'P', 'T', 'P'): '00',
    ('T', 'P', 'P', 'T'): '00',
    ('P', 'T', 'T', 'P'): '00',
    ('P', 'T', 'P', 'T'): '00',

    ('T', 'T', 'P', 'P'): '02',
    ('P', 'P', 'T', 'T'): '02',

    ('T', 'P', 'P', 'P'): '02',
    ('P', 'T', 'P', 'P'): '02',
    ('P', 'P', 'T', 'P'): '02',
    ('P', 'P', 'P', 'T'): '02',

    ('P', 'P', 'P', 'P'): '++'
}

canonized_combos = {}

for coadd_info1, coadd_info2, coadd_info3, coadd_info4 in product(coadd_infos, repeat=4):

    # canonize the coadded fields
    coadd_infoi, coadd_infoj, coadd_infop, coadd_infoq = psc.canonize_disconnected_4pt(
        coadd_info1, coadd_info2, coadd_info3, coadd_info4, coadd_infos
    )

    if (coadd_infoi, coadd_infoj, coadd_infop, coadd_infoq) not in canonized_combos:
        canonized_combos[(coadd_infoi, coadd_infoj, coadd_infop, coadd_infoq)] = [(coadd_info1, coadd_info2, coadd_info3, coadd_info4)]
    else:
        canonized_combos[(coadd_infoi, coadd_infoj, coadd_infop, coadd_infoq)].append((coadd_info1, coadd_info2, coadd_info3, coadd_info4))

np.save(f'{covariances_dir}/canonized_split_averaged_unbinned_pseudo_cov_combos.npy', canonized_combos)

canonized_couplings = list(np.load(f'{couplings_dir}/canonized_couplings_4pt_combos.npy', allow_pickle=True).item().keys())
canonized_blocks = list(canonized_combos.keys())
coup2block = np.zeros((len(canonized_blocks), len(canonized_couplings)), dtype=int)

# # main loop, we will add all S, N terms together here
# # iterate over all pairs/orders of coadded fields
for i, (coadd_infoi, coadd_infoj, coadd_infop, coadd_infoq) in enumerate(canonized_blocks):

    pseudo_cov_fn = f"{covariances_dir}/pseudo_cov_{'_'.join(coadd_infoi)}x{'_'.join(coadd_infoj)}"
    pseudo_cov_fn += f"_{'_'.join(coadd_infop)}x{'_'.join(coadd_infoq)}.npy"

    if os.path.isfile(pseudo_cov_fn):
        log.info(f'{pseudo_cov_fn} exists, skipping')
    else:
        log.info(f'Adding {pseudo_cov_fn} to to-do list')

        svi, ari, chani, poli = coadd_infoi
        svj, arj, chanj, polj = coadd_infoj
        svp, arp, chanp, polp = coadd_infop
        svq, arq, chanq, polq = coadd_infoq

        # TP is needed to grab canonized windows and couplings
        TPi, pi = psc.pol2pol_info(poli)  # e.g. 'E' --> 'P', 1
        TPj, pj = psc.pol2pol_info(polj)
        TPp, pp = psc.pol2pol_info(polp)
        TPq, pq = psc.pol2pol_info(polq)

        spintype_ipjq = pols2spintypes[(TPi, TPp, TPj, TPq)]
        spintype_iqjp = pols2spintypes[(TPi, TPq, TPj, TPp)]

        # iterate over all split crosses (ij, pq)
        nspliti = len(d[f'maps_{svi}_{ari}_{chani}'])
        nsplitj = len(d[f'maps_{svj}_{arj}_{chanj}'])
        nsplitp = len(d[f'maps_{svp}_{arp}_{chanp}'])
        nsplitq = len(d[f'maps_{svq}_{arq}_{chanq}'])
        
        # if svi == svj, loop over split combos excl. crosses, else include crosses
        if svi == svj:
            assert nspliti == nsplitj, \
                f'{svi=} and {svj=} are equal but {nspliti=} and {nsplitj=} are not'
            split_ij_iterator = list(permutations(range(nspliti), r=2))
        else:
            split_ij_iterator = list(product(range(nspliti), range(nsplitj)))

        # if svp == svq, loop over split combos excl. crosses, else include crosses
        if svp == svq:
            assert nsplitp == nsplitq, \
                f'{svp=} and {svq=} are equal but {nsplitp=} and {nsplitq=} are not'
            split_pq_iterator = list(permutations(range(nsplitp), r=2))
        else:
            split_pq_iterator = list(product(range(nsplitp), range(nsplitq)))
        
        for si, sj in split_ij_iterator:
            for sp, sq in split_pq_iterator:
                # need for getting canonized ewin_infos and couplings
                field_infoi = (svi, ari, chani, si, TPi)
                field_infoj = (svj, arj, chanj, sj, TPj)
                field_infop = (svp, arp, chanp, sp, TPp)
                field_infoq = (svq, arq, chanq, sq, TPq)

                # there are eight terms, some of which we will skip due to 
                # noise crosses (e.g. if si != sp). we will add each term
                # to pseudo_cov

                # Sip Sjq
                ewin_name1, ewin_name2, ewin_name3, ewin_name4 = psc.canonize_disconnected_4pt(
                    psc.get_ewin_info_from_field_info(field_infoi, d, mode='w'),
                    psc.get_ewin_info_from_field_info(field_infop, d, mode='w'),
                    psc.get_ewin_info_from_field_info(field_infoj, d, mode='w'),
                    psc.get_ewin_info_from_field_info(field_infoq, d, mode='w'),
                    ewin_infos
                    ) 
                coupling_info = (ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_ipjq)
                j = canonized_couplings.index(coupling_info)
                coup2block[i, j] += 1

                # Sip Njq
                if (svj != svq) or (arj != arq) or (sj != sq):
                    pass
                else:
                    ewin_name1, ewin_name2, ewin_name3, ewin_name4 = psc.canonize_disconnected_4pt(
                        psc.get_ewin_info_from_field_info(field_infoi, d, mode='w'),
                        psc.get_ewin_info_from_field_info(field_infop, d, mode='w'),
                        psc.get_ewin_info_from_field_info(field_infoj, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infoq, d, mode='ws', extra='sqrt_pixar'),
                        ewin_infos
                        ) 
                    coupling_info = (ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_ipjq)
                    j = canonized_couplings.index(coupling_info)
                    coup2block[i, j] += 1

                # Nip Sjq
                if (svi != svp) or (ari != arp) or (si != sp):
                    pass
                else:
                    ewin_name1, ewin_name2, ewin_name3, ewin_name4 = psc.canonize_disconnected_4pt(
                        psc.get_ewin_info_from_field_info(field_infoi, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infop, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infoj, d, mode='w'),
                        psc.get_ewin_info_from_field_info(field_infoq, d, mode='w'),
                        ewin_infos
                        ) 
                    coupling_info = (ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_ipjq)
                    j = canonized_couplings.index(coupling_info)
                    coup2block[i, j] += 1

                # Nip Njq
                if (svi != svp) or (ari != arp) or (si != sp) or (svj != svq) or (arj != arq) or (sj != sq):
                    pass
                else:
                    ewin_name1, ewin_name2, ewin_name3, ewin_name4 = psc.canonize_disconnected_4pt(
                        psc.get_ewin_info_from_field_info(field_infoi, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infop, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infoj, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infoq, d, mode='ws', extra='sqrt_pixar'),
                        ewin_infos
                        ) 
                    coupling_info = (ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_ipjq)
                    j = canonized_couplings.index(coupling_info)
                    coup2block[i, j] += 1

                # Siq Sjp
                ewin_name1, ewin_name2, ewin_name3, ewin_name4 = psc.canonize_disconnected_4pt(
                    psc.get_ewin_info_from_field_info(field_infoi, d, mode='w'),
                    psc.get_ewin_info_from_field_info(field_infoq, d, mode='w'),
                    psc.get_ewin_info_from_field_info(field_infoj, d, mode='w'),
                    psc.get_ewin_info_from_field_info(field_infop, d, mode='w'),
                    ewin_infos
                    ) 
                coupling_info = (ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_iqjp)
                j = canonized_couplings.index(coupling_info)
                coup2block[i, j] += 1
                
                # Siq Njp
                if (svj != svp) or (arj != arp) or (sj != sp):
                    pass
                else:
                    ewin_name1, ewin_name2, ewin_name3, ewin_name4 = psc.canonize_disconnected_4pt(
                        psc.get_ewin_info_from_field_info(field_infoi, d, mode='w'),
                        psc.get_ewin_info_from_field_info(field_infoq, d, mode='w'),
                        psc.get_ewin_info_from_field_info(field_infoj, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infop, d, mode='ws', extra='sqrt_pixar'),
                        ewin_infos
                        ) 
                    coupling_info = (ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_iqjp)
                    j = canonized_couplings.index(coupling_info)
                    coup2block[i, j] += 1

                # Niq Sjp
                if (svi != svq) or (ari != arq) or (si != sq):
                    pass
                else:
                    ewin_name1, ewin_name2, ewin_name3, ewin_name4 = psc.canonize_disconnected_4pt(
                        psc.get_ewin_info_from_field_info(field_infoi, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infoq, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infoj, d, mode='w'),
                        psc.get_ewin_info_from_field_info(field_infop, d, mode='w'),
                        ewin_infos
                        ) 
                    coupling_info = (ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_iqjp)
                    j = canonized_couplings.index(coupling_info)
                    coup2block[i, j] += 1
                    
                # Niq Njp
                if (svi != svq) or (ari != arq) or (si != sq) or (svj != svp) or (arj != arp) or (sj != sp):
                    pass
                else:
                    ewin_name1, ewin_name2, ewin_name3, ewin_name4 = psc.canonize_disconnected_4pt(
                        psc.get_ewin_info_from_field_info(field_infoi, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infoq, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infoj, d, mode='ws', extra='sqrt_pixar'),
                        psc.get_ewin_info_from_field_info(field_infop, d, mode='ws', extra='sqrt_pixar'),
                        ewin_infos
                        ) 
                    coupling_info = (ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_iqjp)
                    j = canonized_couplings.index(coupling_info)
                    coup2block[i, j] += 1

np.save(f'{covariances_dir}/canonized_couplings2blocks.npy', coup2block)

# now that we know which blocks load which couplings and how many times
# per coupling, we try to group blocks to minimize the number of total
# times a given group of blocks needs to read couplings from disk
target_num_ops = coup2block.sum() // target_ngroups

groups = []
group_total_reads = []
group_total_ops = []

ungrouped = np.arange(len(coup2block), dtype=int)
while(len(ungrouped) > 0):    
    # get the row of coup2block, among rows that haven't yet been assigned, that has the largest sum 
    maxidx = ungrouped[np.argmax(coup2block.sum(axis=1)[ungrouped])] 
    
    # get the dot products with this row among rows that haven't yet been assigned (including this row)
    dot_prods = coup2block[ungrouped] @ coup2block[maxidx]

    # idxs into ungrouped sorted by highest dot products with high-norm row of coup2block
    ungrouped_idxs_by_high_to_low_dot_prods = np.argsort(dot_prods)[::-1]

    # total operations by highest dot products
    total_ops_by_high_to_low_dot_prods = coup2block.sum(axis=1)[ungrouped[ungrouped_idxs_by_high_to_low_dot_prods]] 
    cum_total_ops = np.cumsum(total_ops_by_high_to_low_dot_prods)

    try:
        # idx of first ungrouped row of coup2block, when sorted by high to low dot prods, 
        # where the cumulative total number of ops is greater than the target
        cut_idx = np.where(cum_total_ops >= target_num_ops)[0].min() 
    except ValueError:
        # if not enough ungrouped rows left to reach target, grab everything remaining
        cut_idx = len(ungrouped_idxs_by_high_to_low_dot_prods) - 1

    ungrouped_idxs = ungrouped_idxs_by_high_to_low_dot_prods[:cut_idx + 1]
    group_idxs = ungrouped[ungrouped_idxs]
    total_reads = (coup2block[group_idxs].sum(axis=0) > 0).sum()
    total_ops = coup2block[group_idxs].sum()

    groups.append(group_idxs)
    group_total_reads.append(total_reads)
    group_total_ops.append(total_ops)

    assert total_ops == cum_total_ops[cut_idx], \
        f'Expected the total operations in this group to be equal to the ' + \
        f'cumulative total operations at the cut index, but got ' + \
        f'{total_ops=} and {cum_total_ops[cut_idx]=}' 
    
    ungrouped = np.delete(ungrouped, ungrouped_idxs)

np.savez(f'{covariances_dir}/canonized_split_averaged_unbinned_pseudo_cov_blocks_recipe.npz',
         *groups,
         group_total_reads=np.array(group_total_reads),
         group_total_ops=np.array(group_total_ops)
         )

plt.figure()
_ = plt.hist(group_total_reads, histtype='step', bins=25, label=f'min={np.min(group_total_reads)}, median={np.median(group_total_reads)}, max={np.max(group_total_reads)}')
plt.legend()
plt.title('total reads per group')
plt.savefig(f'{plot_dir}/group_total_reads.png')

plt.figure()
_ = plt.hist(group_total_ops, histtype='step', bins=25, label=f'min={np.min(group_total_ops)}, median={np.median(group_total_ops)}, max={np.max(group_total_ops)}')
plt.legend()
plt.title('total terms to calculate per group')
plt.savefig(f'{plot_dir}/group_total_ops.png')