description = """
This script computes the mode coupling matrices and the binning matrices Bbl
for the different surveys and arrays.
"""

import argparse
from os.path import join as opj
import numpy as np
from pixell import curvedsky

from pspipe_utils import log, pspipe_list, misc, kspace
from pspy import pspy_utils, so_dict, so_map, so_spectra, so_mcm, so_mpi, sph_tools

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--old', action='store_true', # default False, type bool
                    help='Calculate using old pspy fortran code instead of ducc.')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

if args.old:
    log.warning('using old pspy fortran code will soon be deprecated')

bestfit_dir = d["best_fits_dir"]
cov_dir = d['cov_dir']
pspy_utils.create_directory(cov_dir)

pseudosignal_dir = opj(cov_dir, 'pseudosignal')
pspy_utils.create_directory(pseudosignal_dir)

mcm_dir = d['mcm_dir']
pspy_utils.create_directory(mcm_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

surveys = d["surveys"]
niter = d['niter']
lmax = d["lmax"]
type = d['type']
if type == "Dl":
    doDl = 1
if type == "Cl":
    doDl = 0
if type not in ["Dl", "Cl"]:
    raise ValueError("Unkown 'type' value! Must be either 'Dl' or 'Cl'")
binning_file = d["binning_file"]
binned_mcm = d["binned_mcm"]

apply_kspace_filter = d['apply_kspace_filter']
templates = {}
filter_dicts = {}
for sv in surveys:
    maps = d[f'arrays_{sv}']
    templates[sv] = so_map.read_map(d[f"window_kspace_{sv}_{maps[0]}"])
    if templates[sv].pixel == "CAR":
        if apply_kspace_filter:
            filter_dicts[sv] = d[f"k_filter_{sv}"]
        else:
            filter_dicts[sv] = None
    else:
        filter_dicts[sv] = None

if d["use_toeplitz_mcm"] == True:
    assert args.old, 'can only toeplitz with pspy for now' # FIXME
    log.info("we will use the toeplitz approximation")
    l_exact, l_band, l_toep = 800, 2000, 2750
else:
    l_exact, l_band, l_toep = None, None, None

n_mcms, sv1_list, m1_list, sv2_list, m2_list = pspipe_list.get_spectra_list(d)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_mcms - 1)
log.info(f"[Rank {so_mpi.rank}] Number of mcm matrices to compute: {len(subtasks)}")

if args.old:
    for task in subtasks:
        sv1, m1, sv2, m2 = sv1_list[task], m1_list[task], sv2_list[task], m2_list[task]

        l, bl1 = misc.read_beams(d[f"beam_T_{sv1}_{m1}"], d[f"beam_pol_{sv1}_{m1}"])

        win1_T = so_map.read_map(d[f"window_T_{sv1}_{m1}"])
        win1_pol = so_map.read_map(d[f"window_pol_{sv1}_{m1}"])

        l, bl2 = misc.read_beams(d[f"beam_T_{sv2}_{m2}"], d[f"beam_pol_{sv2}_{m2}"])

        win2_T = so_map.read_map(d[f"window_T_{sv2}_{m2}"])
        win2_pol = so_map.read_map(d[f"window_pol_{sv2}_{m2}"])

        log.info(f"[Rank {so_mpi.rank}, {task:02d}] Computing mcm for {sv1}_{m1} x {sv2}_{m2} the old-fashioned way")

        mbb_inv, Bbl = so_mcm.mcm_and_bbl_spin0and2(win1=(win1_T, win1_pol),
                                                    win2=(win2_T, win2_pol),
                                                    bl1=(bl1["T"], bl1["E"]),
                                                    bl2=(bl2["T"], bl2["E"]),
                                                    binning_file=binning_file,
                                                    niter=niter,
                                                    lmax=lmax,
                                                    type=type,
                                                    l_exact=l_exact,
                                                    l_band=l_band,
                                                    l_toep=l_toep,
                                                    binned_mcm=binned_mcm,
                                                    save_file=opj(f"{mcm_dir}", f"{sv1}_{m1}x{sv2}_{m2}"))

# TODO: Ideally, we would split up mcm and Bbl, since mcm is just ultimately
# used for pseudo2datavec, which is related to Bbl (theory2datavec =
# pseudo2datavec @ theory2pseudo). I.e., we probably want a pseudo2datavec
# script followed by a Bbl script
else:
    # ducc can batch the matrix calculations rather than one at a time, so
    # instead in the loop we just group the inputs, and call the calculation
    # once outside the loop. finally, we need to loop again to apply binning
    # (since the binning function does one matrix at a time) and save the
    # outputs by name individually (also one matrix at a time)

    # TODO: avoid calculation for repeated windows
    specs_for_ducc = []
    for task in subtasks:
        sv1, m1, sv2, m2 = sv1_list[task], m1_list[task], sv2_list[task], m2_list[task]
        pols = ('T', 'pol')

        log.info(f"[Rank {so_mpi.rank}, {task:02d}]: Preparing data for {sv1}_{m1} x {sv2}_{m2}")

        # only calculate the stuff we need to avoid numerical differences
        m2win_fn = {}
        for pol in pols:
            m2win_fn[sv1, m1, pol] = d[f"window_{pol}_{sv1}_{m1}"]
            m2win_fn[sv2, m2, pol] = d[f"window_{pol}_{sv2}_{m2}"] # no repeated keys

        win_fn2win = {}
        for win_fn in m2win_fn.values():
            if win_fn not in win_fn2win:
                win_fn2win[win_fn] = so_map.read_map(win_fn) # no repeated computation (or keys)

        # TODO: make DRY code with so_mcm for preparing inputs
        lmax_limit = np.inf
        for win in win_fn2win.values():
            _lmax_limit = win.get_lmax_limit() * 2 # this is OK
            if _lmax_limit < lmax_limit:
                lmax_limit = _lmax_limit
        if lmax > lmax_limit:
            raise ValueError("the requested lmax is too high with respect to the map pixellisation")
        maxl = np.minimum(2*lmax, lmax_limit).astype(int)

        win_fn2walm = {}
        for win_fn, win in win_fn2win.items():
            if win_fn not in win_fn2walm:
                win_fn2walm[win_fn] = sph_tools.map2alm(win, niter=niter, lmax=maxl, dtype=np.complex128) # no repeated computation (or keys)

        can_win_fn_2pt2cl = {}
        for poli in pols:
            win_fni = m2win_fn[sv1, m1, poli]
            walmi = win_fn2walm[win_fni]
            for polj in pols:
                win_fnj = m2win_fn[sv2, m2, polj]
                walmj = win_fn2walm[win_fnj]

                can_win_fn_2pt_ij = pspipe_list.canonize_connected_2pt(win_fni, win_fnj)
                if can_win_fn_2pt_ij not in can_win_fn_2pt2cl:
                    can_win_fn_2pt2cl[can_win_fn_2pt_ij] = curvedsky.alm2cl(walmi, walmj, dtype=np.float64) # no repeated computation (or keys)

        # tabulate inputs for ducc, avoid numerical differences in specs_for_ducc
        spec_for_ducc = []
        for i in range(2):
            for j in range(2):
                poli, polj = pols[i], pols[j]
                win_fni, win_fnj = m2win_fn[sv1, m1, poli], m2win_fn[sv2, m2, polj]
                can_win_fn_2pt_ij = pspipe_list.canonize_connected_2pt(win_fni, win_fnj)
                spec_for_ducc.append(can_win_fn_2pt2cl[can_win_fn_2pt_ij])
        specs_for_ducc.append(spec_for_ducc)
        
    log.info(f"[Rank {so_mpi.rank}]: Computing mcm matrices using ducc")
    
    specs_for_ducc = np.array(specs_for_ducc).reshape(len(subtasks)*4, maxl + 1) # (nspec, 4, nl) -> (nspec*4, nl)

    mcms = so_mcm.ducc_couplings(specs_for_ducc, lmax, len(subtasks)*[0, 1, 1, 4], # 00, 02, 02, ++, --
                                 dtype=np.float64, coupling=False,
                                 pspy_index_convention=True)

    # nl goes from l = 2 to lmax
    mcms = mcms.reshape(len(subtasks), 5, lmax-2, lmax-2) # (nspec*5, nl, nl) -> (nspec, 5, nl, nl)

    # get the pseudo2data and theory2data matrices, including binning etc.
    # TODO: recompute this for every spectrum individually!
    bin_lo, bin_hi, _, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_hi)

    for t, task in enumerate(subtasks):
        log.info(f"[Rank {so_mpi.rank}, {task:02d}] Computing bbl and other products")

        sv1, m1, sv2, m2 = sv1_list[task], m1_list[task], sv2_list[task], m2_list[task]
        spec_name = f"{sv1}_{m1}x{sv2}_{m2}"

        # get the beams
        # TODO: generalize this into a whole (9nl x 9nl) operator, a la W_l^{WXYZ}
        l1_T, bl1_T = misc.prep_beams(d[f"beam_T_{sv1}_{m1}"], norm='mono', return_err=False)
        l1_P, bl1_P = misc.prep_beams(d[f"beam_pol_{sv1}_{m1}"], norm='mono', return_err=False)
        l2_T, bl2_T = misc.prep_beams(d[f"beam_T_{sv2}_{m2}"], norm='mono', return_err=False)
        l2_P, bl2_P = misc.prep_beams(d[f"beam_pol_{sv2}_{m2}"], norm='mono', return_err=False)
        for _l in (l1_P, l2_T, l2_P):
            assert np.all(l1_T[:lmax] == _l[:lmax]), f'bls assumed to have same ell'
        assert l1_T[0] == 0, f'bls assumed to start at l=0, got l={l1_T[0]}'

        bl = []
        for bl1 in (bl1_T, bl1_P):
            for bl2 in (bl2_T, bl2_P):
                bl.append(bl1[2:lmax] * bl2[2:lmax]) # TODO: reconsider pspipe conventions

        # get the tf. will be 1 if nothing is being filtered, so OK to do this in all cases
        # TODO: implement something like 2111.01113
        l, tf = kspace.build_analytic_kspace_filter_diag(sv1, sv2, lmax, templates, filter_dicts)
        assert l[0] == 0, f'Tf assumed to start at l=0, got l={l[0]}'
        tf = tf[2:lmax] # TODO: reconsider pspipe conventions

        # get the total response. check that it is nonzero in all bins. get the minimum,
        # maximum l of the total response, and a slice object corresponding to this
        total_response = tf * bl # (nl,) * (4, nl) = (4, nl)
        assert total_response.shape == (4, lmax-2), \
            f'expected total_response.shape=(4, {lmax}-2), got {total_response.shape=}'

        l = np.arange(2, lmax) # assumes 2:lmax ordering
        for ibin in range(nbins):
            loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))[0] # this is common idiom for PSpipe binning
            for spin_idx in range(4):
                assert not np.allclose(total_response[spin_idx, loc], 0), \
                    f'bin index {ibin} with bin_lo={bin_lo[ibin]} and bin_hi={bin_hi[ibin]} ' + \
                    f'has zero total_response from tf * bl for {spin_idx=} of {spec_name=}'

        # TODO: right now, for simplicity, I am enforcing equal binning on all TT, TE, etc.,
        # by making the most conservative selection. would be better to allow *everything*
        # to be customizable, so kspace filters are defined at the map and T vs. pol level, 
        # rather than at the survey level. NB this is already the case for the beams, for 
        # example, hence why this loop is necessary even now
        mask = True
        for spin_idx in range(4):
            mask = np.logical_and(mask, np.logical_not(np.isclose(total_response[spin_idx], 0)))
        nonzero_response_l = l[mask]
        assert np.all(np.diff(nonzero_response_l) == 1), \
            'nonzero entries are split into multiple chunks, should be contiguous'

        nonzero_response_l_lo = nonzero_response_l[0]
        nonzero_response_l_hi = nonzero_response_l[-1]
        assert nonzero_response_l_lo < bin_lo[1], \
            'lowest nonzero entry not in or below lowest bin, this should be impossible'
        assert nonzero_response_l_hi > bin_hi[-2], \
            'highest nonzero entry not in or above highest bin, this should be impossible'

        nonzero_response_sel = np.s_[nonzero_response_l_lo-2:nonzero_response_l_hi-2 + 1] # assumes 2:lmax ordering
        assert np.all(l[nonzero_response_sel] == nonzero_response_l), \
            f'got {l[nonzero_response_sel]=} but expected {nonzero_response_l=}'

        # you might expect here that we would apply the total_response on the right to build up the
        # theory2pseudo operator, but we may (in the case of unbinned mcms) need to invert the mcms before
        # applying the inverse of the total_response on the left, because the total_response might be 0 
        # (especially due to the analytic per-ell tf). so we defer applying the total_response later as
        # needed

        # we need to get the best-fit pseudosignal spectra for the covariance. we do
        # that here to avoid recalculating all the unbinned mcms again in a
        # different script
        l, signal_dict = so_spectra.read_ps(opj(bestfit_dir, f'cmb_and_fg_{spec_name}.dat'),
                                            spectra=spectra, return_type='Cl',
                                            return_dtype=np.float32)
        assert l[0] == 2, f'Bestfit spectra assumed to start at l=2, got l={l[0]}'

        # trim to match mcm and apply total_response as in forward model
        # hijack this function; we need to make total_response 3d
        # TODO: promote total_response to a dense (9nl, 9nl) operator 
        total_response_dict = so_mcm.get_spec2spec_sparse_dict_mat_from_spin2spin_array(total_response[:, None, :], spectra) # (4, nl) -> (4, 1, nl)
        for k in signal_dict.keys():
            # we get the diagonal "blocks", and remove spurious extra dim
            signal_dict[k] = total_response_dict[k][k][0] * signal_dict[k][:lmax-2] # (1, nl)[0] * (nl,) = (nl,)

        # the fully realized mcm matrix would be a lot of memory
        pseudosignal_dict = so_mcm.spin2spin_array_matmul_sparse_dict_vec(mcms[t], spectra, signal_dict)
        so_spectra.write_ps(opj(pseudosignal_dir, f'pseudo_cmb_and_fg_{spec_name}.dat'),
                            l[:lmax-2], pseudosignal_dict, 'Cl', spectra=spectra)

        # NOTE: if binned_mcm, then we need to include the total_response before inversion, because
        # it will be binned before being inverted. the above check -- that the total_response is
        # at least nonzero in all bins -- should help ensure that it's invertible
        total_response = np.repeat(total_response, (1, 1, 1, 2), axis=0) # (4, nl) -> (5, nl)

        if binned_mcm:
            Pbl = so_spectra.get_binning_matrix(bin_lo, bin_hi, lmax, type) # b x (lmax - 2)
            mxx = np.zeros((5, nbins, nbins)) # b x b
            Bbl = np.zeros((5, nbins, lmax)) # b x lmax

            for spin_idx in range(5):
                # multiply by tf * bl on the right
                mcms_t_i = mcms[t, spin_idx] * total_response[spin_idx]

                # bins both indices of mll to get mxx, that will then be inverted later.
                # compute Mbb' = (Pbl Mll' * Tl' Ql'b')
                so_mcm.mcm_fortran.bin_mcm(mcms_t_i.T,
                                           bin_lo,
                                           bin_hi,
                                           bin_size,
                                           mxx[i].T,
                                           doDl)

                # compute (Pbl @ theory2pseudo) = (Pbl Mll' * Tl')
                so_mcm.mcm_fortran.binning_matrix(mcms_t_i.T,
                                                  bin_lo,
                                                  bin_hi,
                                                  bin_size,
                                                  Bbl[i].T,
                                                  doDl)

        # NOTE: if not binned_mcm, we wait to include the total_response until after the
        # inversion, because the total_response may be 0 for some ells (and even if not,
        # it's "easy" to invert it, so we don't want to put too much stress on linalg.inv).
        # HOWEVER we do need to "trim" the binning to intake only those ells where the
        # total_response is nonzero
        else:
            # we need to adjust bin_lo and bin_hi so that the binning does not include
            # any ells where the total_response is 0
            _bin_lo = bin_lo.copy()
            _bin_hi = bin_hi.copy()
            _bin_size = bin_size.copy()

            _bin_lo[0] = max(nonzero_response_l_lo, bin_lo[0]) # don't trim if not needed
            _bin_hi[-1] = min(nonzero_response_l_hi, bin_hi[-1]) # don't trim if not needed
            _bin_size[0] -= (_bin_lo[0] - bin_lo[0]) # reduce the bin size by the right amount (perhaps 0)
            _bin_size[-1] -= (bin_hi[-1] - _bin_hi[-1]) # reduce the bin size by the right amount (perhaps 0)

            Pbl = so_spectra.get_binning_matrix(_bin_lo, _bin_hi, lmax, type) # b x (lmax - 2)
            mxx = mcms[t] # l x l
            Bbl = np.zeros((nbins, lmax)) # b x lmax

            so_mcm.mcm_fortran.binning_matrix(np.eye(mcms.shape[-1]).T,
                                              _bin_lo,
                                              _bin_hi,
                                              _bin_size,
                                              Bbl.T,
                                              doDl)

        # invert the mcm and apply binning. NOTE: mbb, mll, Pbl follow
        # (nbin, nbin), (2:lmax, 2:lmax), and (nbin, 2:lmax) shape/ordering
        # respectively, while Bbl follows (nbin, 2:lmax+2) shape/ordering
        mxx_inv = so_mcm.invert_mcm(mxx)

        # compute pseudo2data
        if binned_mcm:
            # NOTE: total response on the right of mcm before binning and inversion, 
            # i.e. compute inv(Mbb') Pb'l = inv(Pbl Mll' * Tl' Ql'b') Pb'l.
            # Cl->Dl + binning happens immediately after pseudo-Cl
            mbl_inv = mxx_inv @ Pbl
        else:
            # NOTE: apply inverse of total response on the left of inv_mcm, 
            # i.e. compute (Pbl" / Tl" inv(Ml"l).
            # Cl->Dl + binning happens after deconvolution
            Pbl = (Pbl[:, nonzero_response_sel] / total_response[:, None, nonzero_response_sel]) # (nbin, _nl) * (5, 1, _nl) -> (5, nbin, _nl)
            mbl_inv = Pbl @ mxx_inv[:, nonzero_response_sel, :] # (5, nbin, _nl) @ (5, _nl, lmax-2) = (5, nbin, lmax-2)

        # finish the Bbl (theory2data) computation
        if binned_mcm:
            # computes Bbl' = inv(Mbb') (Pb'l Mll' * Tl') = inv(Pbl Mll' * Tl' Ql'b') (Pb'l Mll' * Tl'),
            Bbl[:3] = mxx_inv[:3] @ Bbl[:3]
            np.einsum('mnab,nbl->mal',
                      np.array([[mxx_inv[3], mxx_inv[4]], [mxx_inv[4], mxx_inv[3]]]),
                      Bbl[3:],
                      out=Bbl[3:])

        log.info(f"[Rank {so_mpi.rank}, {task:02d}] Saving mcm matrix for {sv1}_{m1} x {sv2}_{m2}")

        np.save(opj(f"{mcm_dir}", spec_name + "_mode_coupling_inv.npy") , mbl_inv)
        np.save(opj(f"{mcm_dir}", spec_name + "_Bbl.npy"), Bbl)
