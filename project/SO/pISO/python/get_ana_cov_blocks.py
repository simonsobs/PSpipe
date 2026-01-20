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
import numba

from pspipe_utils import log, pspipe_list, covariance, dict_utils
from pspy import so_dict, so_mcm, so_mpi, so_spectra

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
cov_spin00_coupling_only = d['cov_spin00_coupling_only']

bestfit_dir = d["best_fits_dir"]
noise_dir = opj(bestfit_dir, 'noise')
mcm_dir = d['mcm_dir']
cov_dir = d['cov_dir']

canonized_sn_field_info2canonized_connected_combo_2pt = npy.load(opj(cov_dir, 'canonized_sn_field_info2canonized_connected_combo_2pt.npy'), allow_pickle=True).item()
canonized_w2s = npy.load(opj(cov_dir, 'canonized_w2s.npy'), allow_pickle=True).item()

canonized_sn_field_info2canonized_connected_combo_4pt = npy.load(opj(cov_dir, 'canonized_sn_field_info2canonized_connected_combo_4pt.npy'), allow_pickle=True).item()
canonized_w4s = npy.load(opj(cov_dir, 'canonized_w4s.npy'), allow_pickle=True).item()

canonized_sn_field_info2canonized_disconnected_combo_4pt = npy.load(opj(cov_dir, 'canonized_sn_field_info2canonized_disconnected_combo_4pt.npy'), allow_pickle=True).item()
reference_sn_field_info2reference_canonized_disconnected_combo_4pt = npy.load(opj(cov_dir, 'reference_sn_field_info2reference_canonized_disconnected_combo_4pt.npy'), allow_pickle=True).item()
canonized_wls = npy.load(opj(cov_dir, 'canonized_wls.npy'), allow_pickle=True).item()

def update_pseudospectra_dict(f1, f2, pseudospectra_dict=None):
    if pseudospectra_dict is None:
        pseudospectra_dict = {}

    sv1, m1, n1 = f1 # T
    sv2, m2, n2 = f2 # E

    split_iterator = ['s']
    dir_iterator = [bestfit_dir]
    fn_template_iterator = ['pseudo_cmb_and_fg_{spec_name}.dat']

    if n1 == n2:
        for k in range(nsplits[sv1]):
            split_iterator.append(f'n{k}')
            dir_iterator.append(noise_dir)
            fn_template_iterator.append('pseudo_noise_{spec_name}_' + f'set{k}.dat')

    for split, dir, fn_template in zip(split_iterator, dir_iterator, fn_template_iterator):
        try:
            spec_name = f"{sv1}_{m1}x{sv2}_{m2}" 
            _, ps_dict = so_spectra.read_ps(opj(dir, fn_template.format(spec_name=spec_name)),
                                            spectra, return_type='Cl',
                                            return_dtype=npy.float32)
        except FileNotFoundError:
            spec_name = f"{sv2}_{m2}x{sv1}_{m1}" # E T
            _, ps_dict = so_spectra.read_ps(opj(dir, fn_template.format(spec_name=spec_name)),
                                            spectra, return_type='Cl',
                                            return_dtype=npy.float32)
            ps_dict = {spec[::-1]: val for spec, val in ps_dict.items()} # ET -> TE
        
        for (TEB1, TEB2), val in ps_dict.items():
            sna1 = sv1, m1, TEB1, split
            sna2 = sv2, m2, TEB2, split 
            can_sn_alm_info = pspipe_list.canonize_connected_2pt(sna1, sna2)
            
            if can_sn_alm_info not in pseudospectra_dict:
                pseudospectra_dict[can_sn_alm_info] = val
        
    return pseudospectra_dict

def TEB2pol(TEB):
    if TEB == 'T':
        return 'T'
    elif TEB in ('E', 'B'):
        return 'pol'
    else:
        raise ValueError('Only valid strs are T, E, or B')
    
def get_can_discon_com_4pt(snf1, snf2, snf3, snf4):
    sv1, m1, pol1, split1 = snf1
    sv2, m2, pol2, split2 = snf2
    sv3, m3, pol3, split3 = snf3
    sv4, m4, pol4, split4 = snf4
    can_sn_field_info = pspipe_list.canonize_disconnected_4pt(snf1, snf2, snf3, snf4)
    can = can_sn_field_info in canonized_sn_field_info2canonized_disconnected_combo_4pt

    ref_split1, ref_split2, ref_split3, ref_split4 = covariance.get_4pt_sn_term_type(split1, split2, split3, split4)
    ref_snf1 = (sv1, m1, pol1, ref_split1)
    ref_snf2 = (sv2, m2, pol2, ref_split2)
    ref_snf3 = (sv3, m3, pol3, ref_split3)
    ref_snf4 = (sv4, m4, pol4, ref_split4)
    ref_sn_field_info = (ref_snf1, ref_snf2, ref_snf3, ref_snf4)
    ref = ref_sn_field_info in reference_sn_field_info2reference_canonized_disconnected_combo_4pt

    if can and ref:
        raise ValueError(
            f'For sn_field_info {snf1, snf2, snf3, snf4}, the possible reference disconnected '
            f'sn_field_info {ref_sn_field_info} is in '
            'reference_sn_field_info2reference_canonized_disconnected_combo_4pt '
            f'and canonized disconnected sn_field_info {can_sn_field_info} is in '
            'canonized_sn_field_info2canonized_disconnected_combo_4pt'
            )        

    if can:
        can_discon_com_4pt = canonized_sn_field_info2canonized_disconnected_combo_4pt[can_sn_field_info]
    if ref:
        can_discon_com_4pt = reference_sn_field_info2reference_canonized_disconnected_combo_4pt[ref_sn_field_info]
    
    return can_discon_com_4pt
    
# FIXME: we need this for now pending ducc api changes (see below)
def pols_disconnected_combo_4pt2ducc_mat_spintype_idx(pol1, pol2, pol3, pol4):
    spin2_1 = int(pol1 == 'pol' and pol2 == 'pol')
    spin2_2 = int(pol3 == 'pol' and pol4 == 'pol')

    # if 0, then the spintype is 00, which is ducc_mat index 0
    # if 1, then the spintype is 02 (or 20), which is ducc_mat index 1
    # if 2, then the spintype is ++, which is ducc_mat index 2
    return spin2_1 + spin2_2

@numba.njit(parallel=True)
def add_term_to_pseudo_cov_block(pseudo_cov_block, num_terms, w4_1234, w4_coupling, w2_12, w2_34, C12, C34, coupling):
    # important to cast the scalar to the right type before multiplication, 
    # which is a little faster than having it figure out the casting on-the-fly
    prefactor = coupling.dtype.type(num_terms * w4_1234 / (4 * w2_12 * w2_34 * w4_coupling))

    C12_2d = npy.expand_dims(C12, 0)
    C12_2d = npy.broadcast_to(C12_2d, coupling.shape)

    C34_2d = npy.expand_dims(C34, 0)
    C34_2d = npy.broadcast_to(C34_2d, coupling.shape)
    pseudo_cov_block += prefactor * (C12_2d + C12_2d.T) * (C34_2d + C34_2d.T) * coupling

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

# now cov blocks
so_mpi.init(True)

subtasks = so_mpi.taskrange(imin=0, imax=n_covs - 1)
log.info(f"[Rank {so_mpi.rank}] Number of covs to compute: {len(subtasks)} (out of {n_covs} total)")

cov_block2couplings_preshape = {}
cov_block2TEB_block2can_sn_alm_info2nterms = {}

for task in subtasks:
    svi, mi = ni_list[task].split('&') 
    svj, mj = nj_list[task].split('&')
    svp, mp = np_list[task].split('&')
    svq, mq = nq_list[task].split('&')

    log.info(f"[Rank {so_mpi.rank}, Task {task}]: Calculating ana. cov. for {svi}_{mi}x{svj}_{mj}, {svp}_{mp}x{svq}_{mq}")

    t_block = time.time()

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

    # gather our ingredients for this block
    #
    # first spectra    
    pseudospectra_dict = {}
    update_pseudospectra_dict((svi, mi, ni), (svp, mp, np), pseudospectra_dict=pseudospectra_dict)
    update_pseudospectra_dict((svj, mj, nj), (svq, mq, nq), pseudospectra_dict=pseudospectra_dict)
    
    update_pseudospectra_dict((svi, mi, ni), (svq, mq, nq), pseudospectra_dict=pseudospectra_dict)
    update_pseudospectra_dict((svj, mj, nj), (svp, mp, np), pseudospectra_dict=pseudospectra_dict)
            
    # now couplings. we need figure out which couplings we actually need first
    #
    # FIXME: the general case with ducc is too tricky to implement right now;
    # instead, we only support two simpler cases: spin00 (for arbitrary windows),
    # and all spins but only when T window and pol window are the same for all
    # fields in this block. otherwise, we need to call the function 3 times, 
    # with spin00-alone, and two mixed spins, and each of the mixed spins 
    # waste somte computation. ideally we want ducc to not force you to package
    # spins together

    # can_discon_com_4pts2specs_for_ducc to track indices in output ducc
    # array. use list instead of set because we need to track insertion order,
    # but a dict is overkill
    can_discon_com_4pts2specs_for_ducc = [] 
    specs_for_ducc = []

    # FIXME: a cludge around the all-spins case for ducc right now
    if not cov_spin00_coupling_only:
        ducc_all_spin_one_coup_check = {}

    # for each cov TEB sub-block, tracks how many times a canonical 4pt combo
    # of (sv, m, TEB, split)s recurs, so it can be added once (times this count)
    # rather than each time
    TEB_block2can_sn_alm_info2nterms = {} # "alm_info" since keys are TEB instead of T and pol
    
    def update_ducc_inputs_and_nterms(sna1, sna2, sna3, sna4, can_sn_alm_info2nterms):
        # update ducc inputs with minimal unique couplings, and track their
        # order
        sv1, m1, TEB1, split1 = sna1
        sv2, m2, TEB2, split2 = sna2
        sv3, m3, TEB3, split3 = sna3
        sv4, m4, TEB4, split4 = sna4

        pol1 = TEB2pol(TEB1)
        pol2 = TEB2pol(TEB2)
        pol3 = TEB2pol(TEB3)
        pol4 = TEB2pol(TEB4)

        snf1 = (sv1, m1, pol1, split1)
        snf2 = (sv2, m2, pol2, split2)
        snf3 = (sv3, m3, pol3, split3)
        snf4 = (sv4, m4, pol4, split4)
        can_discon_com_4pt = get_can_discon_com_4pt(snf1, snf2, snf3, snf4)

        if can_discon_com_4pt not in can_discon_com_4pts2specs_for_ducc:
            can_discon_com_4pts2specs_for_ducc.append(can_discon_com_4pt)
            specs_for_ducc.append(canonized_wls[can_discon_com_4pt])

        # we are iterating over (T, pol) to make sure there is one coupling for
        # this term otherwise, i.e., the key is 4pt sn_field_info ex (T, pol)
        if not cov_spin00_coupling_only:
            f1 = sv1, m1, split1
            f2 = sv2, m2, split2
            f3 = sv3, m3, split3
            f4 = sv4, m4, split4
            if (f1, f2, f3, f4) not in ducc_all_spin_one_coup_check:
                ducc_all_spin_one_coup_check[f1, f2, f3, f4] = can_discon_com_4pt
            else:
                assert ducc_all_spin_one_coup_check[f1, f2, f3, f4] == can_discon_com_4pt, \
                    'For now, cannot support pol-dependent windows if doing spinny ' + \
                    'coupling flavors'

        # track the number of times this term has appeared in this cov TEB
        # sub-block
        can_sn_alm_info = pspipe_list.canonize_disconnected_4pt(sna1, sna2, sna3, sna4)
        if can_sn_alm_info not in can_sn_alm_info2nterms:
            can_sn_alm_info2nterms[can_sn_alm_info] = 1
        else:
            can_sn_alm_info2nterms[can_sn_alm_info] += 1

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
                                          can_sn_alm_info2nterms)
            
            # ssnn ipjq
            if nj == nq and sj == sq:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, 's'), (svp, mp, TEBp, 's'),
                                              (svj, mj, TEBj, f'n{sj}'), (svq, mq, TEBq, f'n{sj}'),
                                              can_sn_alm_info2nterms)

            # nnss ipjq
            if ni == np and si == sp:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, f'n{si}'), (svp, mp, TEBp, f'n{si}'),
                                              (svj, mj, TEBj, 's'), (svq, mq, TEBq, 's'),
                                              can_sn_alm_info2nterms)
                    
            # nnnn ipjq
            if ni == np and si == sp and nj == nq and sj == sq:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, f'n{si}'), (svp, mp, TEBp, f'n{si}'),
                                              (svj, mj, TEBj, f'n{sj}'), (svq, mq, TEBq, f'n{sj}'),
                                              can_sn_alm_info2nterms)
                    
            # ssss iqjp
            update_ducc_inputs_and_nterms((svi, mi, TEBi, 's'), (svq, mq, TEBq, 's'),
                                          (svj, mj, TEBj, 's'), (svp, mp, TEBp, 's'),
                                          can_sn_alm_info2nterms)
            
            # ssnn iqjp
            if nj == np and sj == sp:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, 's'), (svq, mq, TEBq, 's'),
                                              (svj, mj, TEBj, f'n{sj}'), (svp, mp, TEBp, f'n{sj}'),
                                              can_sn_alm_info2nterms)

            # nnss iqjp
            if ni == nq and si == sq:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, f'n{si}'), (svq, mq, TEBq, f'n{si}'),
                                              (svj, mj, TEBj, 's'), (svp, mp, TEBp, 's'),
                                              can_sn_alm_info2nterms)
                    
            # nnnn iqjp
            if ni == nq and si == sq and nj == np and sj == sp:
                update_ducc_inputs_and_nterms((svi, mi, TEBi, f'n{si}'), (svq, mq, TEBq, f'n{si}'),
                                              (svj, mj, TEBj, f'n{sj}'), (svp, mp, TEBp, f'n{sj}'),
                                              can_sn_alm_info2nterms)

    # set up and perform the ducc calculation
    specs_for_ducc = npy.array(specs_for_ducc)[:, None, :] # (nspecs, nl) -> (nspecs, 1, nl)
    if cov_spin00_coupling_only:
        # just line all canonized wls up and send to ducc
        spec_index = (0, 0, 0, 0)
        mat_index = (0, -1, -1, -1, -1)
    else:
        # repeat wl as 00, 02, and 22 spin pairs because we need the 00, 02, ++
        # coupling flavors for each wl
        specs_for_ducc = npy.tile(specs_for_ducc, (1, 3, 1)) # (nspecs, 1, nl) -> (nspecs, 3, nl)
        spec_index = (0, 1, 1, 2)
        mat_index = (0, 1, -1, 2, -1)

    cov_block2couplings_preshape[(svi, mi), (svj, mj), 
                                 (svp, mp), (svq, mq)] = specs_for_ducc.shape[:-1]
    cov_block2TEB_block2can_sn_alm_info2nterms[(svi, mi), (svj, mj), 
                                               (svp, mp), (svq, mq)] = TEB_block2can_sn_alm_info2nterms

    t0 = time.time()
    coups = so_mcm.ducc_couplings(specs_for_ducc, lmax, spec_index=spec_index,
                                  mat_index=mat_index, dtype=npy.float32,
                                  coupling=True, pspy_index_convention=True)
    
    log.info(f'[Rank {so_mpi.rank}, Task {task}]: Calculated {specs_for_ducc.shape[0]}x{specs_for_ducc.shape[1]} couplings in {(time.time() - t0):.3f} seconds')            

    # now add all terms together
    pseudo_cov = npy.zeros((len(spectra) * (lmax - 2), len(spectra) * (lmax - 2)), dtype=npy.float32)

    t0 = time.time()
    for ridx, (TEBi, TEBj) in enumerate(spectra):
        for cidx, (TEBp, TEBq) in enumerate(spectra):
            pseudo_cov_block = pseudo_cov[ridx*(lmax - 2):(ridx+1)*(lmax - 2), cidx*(lmax - 2):(cidx+1)*(lmax - 2)]

            can_sn_alm_info2nterms = TEB_block2can_sn_alm_info2nterms[TEBi, TEBj, TEBp, TEBq]
            for can_sn_alm_info, nterms in can_sn_alm_info2nterms.items():
                (sna1, sna2, sna3, sna4) = can_sn_alm_info
                sv1, m1, TEB1, split1 = sna1
                sv2, m2, TEB2, split2 = sna2
                sv3, m3, TEB3, split3 = sna3
                sv4, m4, TEB4, split4 = sna4

                if nterms is not None:                    
                    pol1 = TEB2pol(TEB1)
                    pol2 = TEB2pol(TEB2)
                    pol3 = TEB2pol(TEB3)
                    pol4 = TEB2pol(TEB4)

                    snf1 = (sv1, m1, pol1, split1)
                    snf2 = (sv2, m2, pol2, split2)
                    snf3 = (sv3, m3, pol3, split3)
                    snf4 = (sv4, m4, pol4, split4)

                    # get the w4 for this actual can_sn_alm_info
                    can_con_sn_field_info_1234 = pspipe_list.canonize_connected_4pt(snf1, snf2, snf3, snf4)
                    can_con_com_4pt_1234 = canonized_sn_field_info2canonized_connected_combo_4pt[can_con_sn_field_info_1234]
                    w4_1234 = canonized_w4s[can_con_com_4pt_1234]

                    # get the w4 for the coupling that will be used for this term
                    can_discon_com_4pt_coupling = get_can_discon_com_4pt(snf1, snf2, snf3, snf4)
                    can_con_com_4pt_coupling = pspipe_list.canonize_connected_4pt(*can_discon_com_4pt_coupling)
                    w4_coupling = canonized_w4s[can_con_com_4pt_coupling]
                    
                    # get the w2 factors
                    can_con_sn_field_info_12 = pspipe_list.canonize_connected_2pt(snf1, snf2)
                    can_con_com_2pt_12 = canonized_sn_field_info2canonized_connected_combo_2pt[can_con_sn_field_info_12]
                    w2_12 = canonized_w2s[can_con_com_2pt_12]

                    can_con_sn_field_info_34 = pspipe_list.canonize_connected_2pt(snf3, snf4)
                    can_con_com_2pt_34 = canonized_sn_field_info2canonized_connected_combo_2pt[can_con_sn_field_info_34]
                    w2_34 = canonized_w2s[can_con_com_2pt_34]

                    # get the spectra
                    # sna pairs already canonized since discon 4pt is canonized
                    C12 = pseudospectra_dict[sna1, sna2]
                    C34 = pseudospectra_dict[sna3, sna4]
                
                    # get the coupling itself
                    ducc_mat_coup_idx = can_discon_com_4pts2specs_for_ducc.index(can_discon_com_4pt_coupling)
                    if cov_spin00_coupling_only:
                        ducc_mat_spintype_idx = 0 
                    else:
                        ducc_mat_spintype_idx = pols_disconnected_combo_4pt2ducc_mat_spintype_idx(pol1, pol2, pol3, pol4)

                    coupling = coups[ducc_mat_coup_idx, ducc_mat_spintype_idx]

                    add_term_to_pseudo_cov_block(pseudo_cov_block, nterms, w4_1234,
                                                 w4_coupling, w2_12, w2_34, C12, C34,
                                                 coupling)

    log.info(f'[Rank {so_mpi.rank}, Task {task}]: Added terms to pseudo cov in {(time.time() - t0):.3f} seconds')

    # convert from pseudo to spec cov
    # NOTE: cast the pseudo2datavec because they are small so casting is fast,
    # whereas casting pseudo_cov is slow (~8 seconds). matmul automatically 
    # promotes, so we would cast pseudo_cov if we do nothing. by degrading
    # pseudo2datavec, we lose some accuracy, but this is all approximate anyway
    t0 = time.time()

    pseudo_cov = so_mcm.get_spec2spec_sparse_dict_mat_from_dense_mat(pseudo_cov, spectra, skip_empty=False)

    spec_name_ij = f"{svi}_{mi}x{svj}_{mj}"
    pseudo2datavec_ij = npy.load(opj(f'{mcm_dir}', f'pseudo2datavec_{spec_name_ij}.npy'), allow_pickle=True).item()
    pseudo2datavec_ij = so_mcm.sparse_dict_mat_astype(pseudo2datavec_ij, npy.float32)

    spec_name_pq = f"{svp}_{mp}x{svq}_{mq}"
    pseudo2datavec_pq_T = npy.load(opj(f'{mcm_dir}', f'pseudo2datavec_{spec_name_pq}.npy'), allow_pickle=True).item()
    pseudo2datavec_pq_T = so_mcm.sparse_dict_mat_astype(pseudo2datavec_pq_T, npy.float32)
    pseudo2datavec_pq_T = so_mcm.sparse_dict_mat_transpose(pseudo2datavec_pq_T)

    # do sparse math. we want the dense array in the end
    ana_cov = so_mcm.sparse_dict_mat_matmul_sparse_dict_mat(pseudo2datavec_ij, pseudo_cov)
    ana_cov = so_mcm.sparse_dict_mat_matmul_sparse_dict_mat(ana_cov, pseudo2datavec_pq_T,
                                                            dense=True, dtype=npy.float32)
    
    # finalize: need to divide the split factor from each side and cast to double
    ana_cov /= (len(splits_cross_iterator_ij) * len(splits_cross_iterator_pq))
    ana_cov = ana_cov.astype(npy.float64, copy=False)

    log.info(f'[Rank {so_mpi.rank}, Task {task}]: Calculated pseudo-to-power-spectrum cov in {(time.time() - t0):.3f} seconds')
    
    npy.save(opj(cov_dir, f'analytic_cov_{spec_name_ij}_{spec_name_pq}.npy'), ana_cov)
    log.info(f'[Rank {so_mpi.rank}, Task {task}]: Calculated ana. cov. for {svi}_{mi}x{svj}_{mj}, {svp}_{mp}x{svq}_{mq} in {(time.time() - t_block):.3f} seconds')

    # cleanup
    pseudo_cov = None
    pseudo2datavec_ij = None
    pseudo2datavec_pq_T
    ana_cov = None

# these may be useful to check later
cov_block2couplings_preshape = so_mpi.gather_set_or_dict(cov_block2couplings_preshape,
                                                         allgather=False,
                                                         root=0,
                                                         overlap_allowed=False)

cov_block2TEB_block2can_sn_alm_info2nterms = so_mpi.gather_set_or_dict(cov_block2TEB_block2can_sn_alm_info2nterms,
                                                                       allgather=False,
                                                                       root=0,
                                                                       overlap_allowed=False)

if so_mpi.rank == 0:
    npy.save(opj(cov_dir, 'cov_block2couplings_preshape.npy'), cov_block2couplings_preshape)
    npy.save(opj(cov_dir, 'cov_block2TEB_block2can_sn_alm_info2nterms.npy'), cov_block2TEB_block2can_sn_alm_info2nterms)