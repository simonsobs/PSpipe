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

mcm_dir = d['mcm_dir']
pspy_utils.create_directory(mcm_dir)

bestfit_dir = d["best_fits_dir"]

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

# TODO: rewrite to not repeat for equivalent windows. Ideally, we would split up
# mcm and Bbl, since mcm is just ultimately used for pseudo2datavec, which is 
# related to Bbl (theory2datavec = pseudo2datavec @ theory2pseudo). I.e., we 
# probably want a pseudo2datavec script followed by a Bbl script
else:
    # ducc can batch the matrix calculations rather than one at a time, so
    # instead in the loop we just group the inputs, and call the calculation
    # once outside the loop. finally, we need to loop again to apply binning
    # (since the binning function does one matrix at a time) and save the
    # outputs by name individually (also one matrix at a time)
    specs_for_ducc = []
    bls = []
    for task in subtasks:
        sv1, m1, sv2, m2 = sv1_list[task], m1_list[task], sv2_list[task], m2_list[task]
        mapnames = ((sv1, m1), (sv2, m2))
        pols = ('T', 'pol')

        log.info(f"[Rank {so_mpi.rank}, {task:02d}]: Preparing data for {sv1}_{m1} x {sv2}_{m2}")

        # only calculate the stuff we need to avoid numerical differences
        m2win_fn = {}
        for (sv, m) in mapnames:
            for pol in pols:
                m2win_fn[sv, m, pol] = d[f"window_{pol}_{sv}_{m}"] # no repeated keys

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
        for win_fn1, walm1 in win_fn2walm.items():
            for win_fn2, walm2 in win_fn2walm.items():
                can_win_fn_2pt = pspipe_list.canonize_connected_2pt(win_fn1, win_fn2)
                if can_win_fn_2pt not in can_win_fn_2pt2cl:
                    can_win_fn_2pt2cl[can_win_fn_2pt] = curvedsky.alm2cl(walm1, walm2, dtype=np.float64) # no repeated computation (or keys)

        # beams have no numerical differences due to multithreading
        _, bl1 = misc.read_beams(d[f"beam_T_{sv1}_{m1}"], d[f"beam_pol_{sv1}_{m1}"])
        _, bl2 = misc.read_beams(d[f"beam_T_{sv2}_{m2}"], d[f"beam_pol_{sv2}_{m2}"])
        
        bl1 = (bl1["T"], bl1["E"])
        bl2 = (bl2["T"], bl2["E"])

        # tabulate inputs for ducc, avoid numerical differences in specs_for_ducc
        spec_for_ducc = []
        bl = []
        for i in range(2):
            for j in range(2):
                (svi, mi), (svj, mj) = mapnames[i], mapnames[j]
                poli, polj = pols[i], pols[j]
                win_fni, win_fnj = m2win_fn[svi, mi, poli], m2win_fn[svj, mj, polj]
                can_win_fn_2pt_ij = pspipe_list.canonize_connected_2pt(win_fni, win_fnj)
                spec_for_ducc.append(can_win_fn_2pt2cl[can_win_fn_2pt])
                bl.append(bl1[i][2:lmax] * bl2[j][2:lmax]) # TODO: reconsider pspipe conventions
        specs_for_ducc.append(spec_for_ducc)
        bls.append(bl)
        
    log.info(f"[Rank {so_mpi.rank}]: Computing mcm matrices using ducc")
    
    specs_for_ducc = np.array(specs_for_ducc).reshape(len(subtasks)*4, maxl + 1) # (nspec, 4, nl) -> (nspec*4, nl)
    bls = np.repeat(bls, (1, 1, 1, 2), axis=1) # (nspec, 4, nl) -> (nspec, 5, nl)
    bls = bls[..., None, :] # (nspec, 5, nl) -> (nspec, 5, 1, nl)

    mcms = so_mcm.ducc_couplings(specs_for_ducc, lmax, len(subtasks)*[0, 1, 1, 4], # 00, 02, 02, ++, --
                                 dtype=np.float64, coupling=False,
                                 pspy_index_convention=True)
    mcms = mcms.reshape(len(subtasks), 5, lmax-2, lmax-2) # (nspec*5, nl, nl) -> (nspec, 5, nl, nl)

    # apply total-diagonal beams on the right
    mcms *= bls

    # get the binned mcms
    bin_lo, bin_hi, _, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_hi)
    Pbl = so_spectra.get_binning_matrix(bin_lo, bin_hi, lmax, type)

    for t, task in enumerate(subtasks):
        log.info(f"[Rank {so_mpi.rank}, {task:02d}] Computing bbl and other products")

        sv1, m1, sv2, m2 = sv1_list[task], m1_list[task], sv2_list[task], m2_list[task]
        spec_name = f"{sv1}_{m1}x{sv2}_{m2}"

        # we need to get the best-fit pseudosignal spectra for the covariance. we do
        # that here to avoid recalculating all the unbinned mcms again in a
        # different script. NOTE: we need beamed (and tf'ed, if necessary)
        # pseudo Cls, but the mcms above already have the beam, so we just need
        # to apply the mcm (and tf, if necessary)
        l, tf = kspace.build_analytic_kspace_filter_diag(sv1, sv2, lmax, templates,
                                                         filter_dicts, dtype=np.float32)
        assert l[0] == 0, f'Tf assumed to start at l=0, got l={l[0]}'

        l, signal_dict = so_spectra.read_ps(opj(bestfit_dir, f'cmb_and_fg_{spec_name}.dat'),
                                            spectra=spectra, return_type='Cl',
                                            return_dtype=np.float32)
        assert l[0] == 2, f'Bestfit spectra assumed to start at l=2, got l={l[0]}'

        # trim to match mcm
        l = l[:lmax-2]
        for k in signal_dict.keys():
            signal_dict[k] = tf[2:lmax] * signal_dict[k][:lmax-2]

        # the fully realized mcm matrix would be a lot of memory. also, don't
        # need to copy blocks since just being used in math
        mcm_dict = so_mcm.get_spec2spec_sparse_dict_mat_from_spin2spin_array(mcms[t], spectra)
        pseudosignal_dict = so_mcm.sparse_dict_mat_matmul_sparse_dict_vec(mcm_dict, signal_dict)
        so_spectra.write_ps(opj(bestfit_dir, f'pseudo_cmb_and_fg_{spec_name}.dat'),
                            l, pseudosignal_dict, 'Cl', spectra=spectra)

        # now do the binning 
        if binned_mcm:
            mxx = np.zeros((5, nbins, nbins)) # b x b
            Bbl = np.zeros((5, nbins, lmax))

            for i in range(5):
                so_mcm.mcm_fortran.bin_mcm(mcms[t, i].T,
                                           bin_lo,
                                           bin_hi,
                                           bin_size,
                                           mxx[i].T,
                                           doDl)

                so_mcm.mcm_fortran.binning_matrix(mcms[t, i].T,
                                                  bin_lo,
                                                  bin_hi,
                                                  bin_size,
                                                  Bbl[i].T,
                                                  doDl)
        else:
            mxx = mcms[t] # l x l
            Bbl = np.zeros((nbins, lmax))

            so_mcm.mcm_fortran.binning_matrix(np.eye(mcms.shape[-1]).T,
                                              bin_lo,
                                              bin_hi,
                                              bin_size,
                                              Bbl.T,
                                              doDl)

        # invert the mcm and apply binning. NOTE: mbb, mll, Pbl follow
        # (nbin, nbin), (2:lmax, 2:lmax), and (nbin, 2:lmax) shape/ordering
        # respectively, while Bbl follows (nbin, 2:lmax+2) shape/ordering
        mxx_inv = so_mcm.invert_mcm(mxx)

        if binned_mcm:
            mbl_inv = mxx_inv @ Pbl # Cl->Dl + binning happens immediately after pseudo-Cl
        else:
            mbl_inv = Pbl @ mxx_inv # Cl->Dl + binning happens after deconvolution

        # finish the Bbl computation for binned_mcm
        if binned_mcm:
            Bbl[:3] = mxx_inv[:3] @ Bbl[:3]
            np.einsum('mnab,nbl->mal',
                      np.array([[mxx_inv[3], mxx_inv[4]], [mxx_inv[4], mxx_inv[3]]]),
                      Bbl[3:],
                      out=Bbl[3:])

        log.info(f"[Rank {so_mpi.rank}, {task:02d}] Saving mcm matrix for {sv1}_{m1} x {sv2}_{m2}")

        np.save(opj(f"{mcm_dir}", spec_name + "_mode_coupling_inv.npy") , mbl_inv)
        np.save(opj(f"{mcm_dir}", spec_name + "_Bbl.npy"), Bbl)