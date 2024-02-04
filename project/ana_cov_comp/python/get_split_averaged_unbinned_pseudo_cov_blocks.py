"""
This script takes our ingredients to construct covariance blocks. To reduce space,
it averages over split crosses, so each "block" it produces is a canonical combo
of 4 (survey, array, channel, pol) fields.

The averaging over split crosses is over all split1!=split2 crosses. E.g., for
4 splits, there are 12 such crosses, so the covariance needs to average over
144 per-split blocks.
"""
import sys
import numpy as np
import numba
from pspipe_utils import log, covariance as psc
from pspy import so_dict
from itertools import product, permutations
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

ewin_alms_dir = d['ewin_alms_dir']
bestfit_dir = d['best_fits_dir']
noise_model_dir = d['noise_model_dir']
couplings_dir = d['couplings_dir']
covariances_dir = d['covariances_dir']

surveys = d['surveys']
arrays = {sv: d[f'arrays_{sv}'] for sv in surveys}
lmax = d['lmax']

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

# hold the pseudospectra on-disk
signal_mat = np.load(f'{bestfit_dir}/pseudo_tfed_beamed_signal_model.npy')
noise_mats = {}
for sv1 in surveys:
    for ar1 in arrays[sv1]:
        for chan1 in arrays[sv1][ar1]:
            for split1 in range(len(d[f'maps_{sv1}_{ar1}_{chan1}'])):
                key = (sv1, ar1, split1)
                if key not in noise_mats:
                    noise_mats[key] = np.load(f'{noise_model_dir}/{sv1}_{ar1}_set{split1}_pseudo_tfed_spec.npy')

# hold the w2 factors on disk
w2s = {}
canonized_w2s = list(np.load(f'{ewin_alms_dir}/canonized_ewin_alms_2pt_combos.npy', allow_pickle=True).item().keys())
for (ewin_name1, ewin_name2) in canonized_w2s:
    w2_fn = f'{ewin_alms_dir}/{ewin_name1}x{ewin_name2}_w2.npy'
    w2s[(ewin_name1, ewin_name2)] = np.load(w2_fn)

# hold only the couplings we need on disk
recipe = np.load(f'{covariances_dir}/canonized_split_averaged_unbinned_pseudo_cov_blocks_recipe.npz')
ngroups = len(recipe.files) - 2 # 2 files are not group indices

assert len(sys.argv) == 3, 'must submit a job array with one array parameter'
group_idx = int(sys.argv[2])
log.info(f'computing only the covariance block group: {group_idx} of {ngroups}')

couplings = {}
group = recipe[f'arr_{group_idx}']
canonized_couplings2blocks = np.load(f'{covariances_dir}/canonized_couplings2blocks.npy')
canonized_couplings_idxs_to_load = np.where(canonized_couplings2blocks[group].sum(axis=0) > 0)[0]
canonized_couplings = list(np.load(f'{couplings_dir}/canonized_couplings_4pt_combos.npy', allow_pickle=True).item().keys())
canonized_couplings_to_load = [canonized_couplings[idx] for idx in canonized_couplings_idxs_to_load]
for (ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype) in canonized_couplings_to_load:
    coupling_fn = f'{couplings_dir}/{ewin_name1}x{ewin_name2}x{ewin_name3}x{ewin_name4}_{psc.spintypes2fntags[spintype]}_coupling.npy'
    couplings[(ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype)] = np.load(coupling_fn)

canonized_blocks = list(np.load(f'{covariances_dir}/canonized_split_averaged_unbinned_pseudo_cov_combos.npy', allow_pickle=True).item().keys())

# numba can help speed up the basic array operations ~2x
@numba.njit
def add_term_to_pseudo_cov(pseudo_cov, w2_12, w2_34, C12, C34, coupling):
    pseudo_cov += (1/w2_12/w2_34) * (C12 + C12[:, None]) * (C34 + C34[:, None]) * coupling

# # main loop, we will add all S, N terms together here
# # iterate over all pairs/orders of coadded fields
for i in group:
    (coadd_infoi, coadd_infoj, coadd_infop, coadd_infoq) = canonized_blocks[i]

    pseudo_cov_fn = f"{covariances_dir}/pseudo_cov_{'_'.join(coadd_infoi)}x{'_'.join(coadd_infoj)}"
    pseudo_cov_fn += f"_{'_'.join(coadd_infop)}x{'_'.join(coadd_infoq)}.npy"

    if os.path.isfile(pseudo_cov_fn):
        log.info(f'{pseudo_cov_fn} exists, skipping')
    else:
        log.info(f'Generating {pseudo_cov_fn}')

        svi, ari, chani, poli = coadd_infoi
        svj, arj, chanj, polj = coadd_infoj
        svp, arp, chanp, polp = coadd_infop
        svq, arq, chanq, polq = coadd_infoq

        # sac is needed to grab signal theory spectra
        saci = sv_ar_chans.index((svi, ari, chani)) # e.g. 'dr6_pa5_f150' --> 2
        sacj = sv_ar_chans.index((svj, arj, chanj))
        sacp = sv_ar_chans.index((svp, arp, chanp))
        sacq = sv_ar_chans.index((svq, arq, chanq))

        # c is needed to grab noise theory spectra
        ci = arrays[svi][ari].index(chani)
        cj = arrays[svj][arj].index(chanj)
        cp = arrays[svp][arp].index(chanp)
        cq = arrays[svq][arq].index(chanq)

        # TP is needed to grab canonized windows and couplings, and
        # p is needed to grab signal and noise theory spectra
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
        
        pseudo_cov = np.zeros((lmax + 1, lmax + 1), dtype=np.float64)
        for si, sj in split_ij_iterator:
            for sp, sq in split_pq_iterator:
                log.info(f'Adding split sub-block for cov of spec({si=}x{sj=}) with spec({sp=}x{sq=})')

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
                w2_12 = w2s[(ewin_name1, ewin_name2)]
                w2_34 = w2s[(ewin_name3, ewin_name4)]
                coupling = couplings[(ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_ipjq)]

                Sip = signal_mat[saci, pi, sacp, pp] 
                Sjq = signal_mat[sacj, pj, sacq, pq]

                add_term_to_pseudo_cov(pseudo_cov, w2_12, w2_34, Sip, Sjq, coupling)

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
                    w2_12 = w2s[(ewin_name1, ewin_name2)]
                    w2_34 = w2s[(ewin_name3, ewin_name4)]
                    coupling = couplings[(ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_ipjq)]

                    Njq = noise_mats[(svj, arj, sj)][cj, pj, cq, pq]
                    
                    add_term_to_pseudo_cov(pseudo_cov, w2_12, w2_34, Sip, Njq, coupling)

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
                    w2_12 = w2s[(ewin_name1, ewin_name2)]
                    w2_34 = w2s[(ewin_name3, ewin_name4)]
                    coupling = couplings[(ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_ipjq)]

                    Nip = noise_mats[(svi, ari, si)][ci, pi, cp, pp]
                    
                    add_term_to_pseudo_cov(pseudo_cov, w2_12, w2_34, Nip, Sjq, coupling)

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
                    w2_12 = w2s[(ewin_name1, ewin_name2)]
                    w2_34 = w2s[(ewin_name3, ewin_name4)]
                    coupling = couplings[(ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_ipjq)]

                    add_term_to_pseudo_cov(pseudo_cov, w2_12, w2_34, Nip, Njq, coupling)

                # Siq Sjp
                ewin_name1, ewin_name2, ewin_name3, ewin_name4 = psc.canonize_disconnected_4pt(
                    psc.get_ewin_info_from_field_info(field_infoi, d, mode='w'),
                    psc.get_ewin_info_from_field_info(field_infoq, d, mode='w'),
                    psc.get_ewin_info_from_field_info(field_infoj, d, mode='w'),
                    psc.get_ewin_info_from_field_info(field_infop, d, mode='w'),
                    ewin_infos
                    ) 
                w2_12 = w2s[(ewin_name1, ewin_name2)]
                w2_34 = w2s[(ewin_name3, ewin_name4)]
                coupling = couplings[(ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_iqjp)]

                Siq = signal_mat[saci, pi, sacq, pq]
                Sjp = signal_mat[sacj, pj, sacp, pp]

                add_term_to_pseudo_cov(pseudo_cov, w2_12, w2_34, Siq, Sjp, coupling)

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
                    w2_12 = w2s[(ewin_name1, ewin_name2)]
                    w2_34 = w2s[(ewin_name3, ewin_name4)]
                    coupling = couplings[(ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_iqjp)]

                    Njp = noise_mats[(svj, arj, sj)][cj, pj, cp, pp] 
                    
                    add_term_to_pseudo_cov(pseudo_cov, w2_12, w2_34, Siq, Njp, coupling)

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
                    w2_12 = w2s[(ewin_name1, ewin_name2)]
                    w2_34 = w2s[(ewin_name3, ewin_name4)]
                    coupling = couplings[(ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_iqjp)]

                    Niq = noise_mats[(svi, ari, si)][ci, pi, cq, pq]
                    
                    add_term_to_pseudo_cov(pseudo_cov, w2_12, w2_34, Niq, Sjp, coupling)

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
                    w2_12 = w2s[(ewin_name1, ewin_name2)]
                    w2_34 = w2s[(ewin_name3, ewin_name4)]
                    coupling = couplings[(ewin_name1, ewin_name2, ewin_name3, ewin_name4, spintype_iqjp)]
                    
                    add_term_to_pseudo_cov(pseudo_cov, w2_12, w2_34, Niq, Njp, coupling)

        # .25 is leftover from symmetrization, 4pi is due to convention definition
        # of coupling
        pseudo_cov *= 0.25 / (4*np.pi) / len(split_ij_iterator) / len(split_pq_iterator)
        np.save(pseudo_cov_fn, pseudo_cov)