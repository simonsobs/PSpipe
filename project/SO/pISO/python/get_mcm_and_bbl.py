description = """
This script computes the mode coupling matrices and the binning matrices Bbl
for the different surveys and arrays.
"""

import argparse
import numpy as np
from pixell import curvedsky

from pspipe_utils import log, pspipe_list, misc
from pspy import pspy_utils, so_dict, so_map, so_mcm, so_mpi, sph_tools

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

surveys = d["surveys"]
niter = d['niter']
lmax = d["lmax"]
l3_pad = d['l3_pad']
type = d['type']
if type == "Dl":
    doDl = 1
if type == "Cl":
    doDl = 0
if type not in ["Dl", "Cl"]:
    raise ValueError("Unkown 'type' value! Must be either 'Dl' or 'Cl'")
binning_file = d["binning_file"]
binned_mcm = d["binned_mcm"]

if d["use_toeplitz_mcm"] == True:
    assert args.old, 'can only toeplitz with pspy for now' # FIXME
    log.info("we will use the toeplitz approximation")
    l_exact, l_band, l_toep = 800, 2000, 2750
else:
    l_exact, l_band, l_toep = None, None, None

n_mcms, sv1_list, ar1_list, sv2_list, ar2_list = pspipe_list.get_spectra_list(d)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=n_mcms - 1)
log.info(f"[Rank {so_mpi.rank}] number of mcm matrices to compute: {len(subtasks)}")

specs_for_ducc = []
bls = []
for task in subtasks:
    task = int(task)
    sv1, ar1, sv2, ar2 = sv1_list[task], ar1_list[task], sv2_list[task], ar2_list[task]
    log.info(f"[{task:02d}] mcm matrix for {sv1}_{ar1} x {sv2}_{ar2}")

    l, bl1 = misc.read_beams(d[f"beam_T_{sv1}_{ar1}"], d[f"beam_pol_{sv1}_{ar1}"])

    win1_T = so_map.read_map(d[f"window_T_{sv1}_{ar1}"])
    win1_pol = so_map.read_map(d[f"window_pol_{sv1}_{ar1}"])

    l, bl2 = misc.read_beams(d[f"beam_T_{sv2}_{ar2}"], d[f"beam_pol_{sv2}_{ar2}"])

    win2_T = so_map.read_map(d[f"window_T_{sv2}_{ar2}"])
    win2_pol = so_map.read_map(d[f"window_pol_{sv2}_{ar2}"])

    if args.old:
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
                                                    save_file=f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}")
    else:
        # ducc can batch the matrix calculations rather than one at a time, so
        # instead in the loop we just group the inputs, and call the calculation
        # once outside the loop. finally, we need to loop again to apply binning
        # (since the binning function does one matrix at a time) and save the
        # outputs by name individually (also one matrix at a time)

        # TODO: make DRY code with so_mcm for preparing inputs
        lmax_limit = np.inf
        for win in (win1_T, win1_pol, win2_T, win2_pol):
            _lmax_limit = win.get_lmax_limit()
            if _lmax_limit < lmax_limit:
                lmax_limit = _lmax_limit
        if lmax > lmax_limit:
            raise ValueError("the requested lmax is too high with respect to the map pixellisation")
        maxl = np.minimum(lmax + l3_pad, lmax_limit)

        win1_alm_T = sph_tools.map2alm(win1_T, niter=niter, lmax=maxl, dtype=np.complex128)
        win1_alm_P = sph_tools.map2alm(win1_pol, niter=niter, lmax=maxl, dtype=np.complex128)
        win1 = (win1_alm_T, win1_alm_P)

        win2_alm_T = sph_tools.map2alm(win2_T, niter=niter, lmax=maxl, dtype=np.complex128)
        win2_alm_P = sph_tools.map2alm(win2_pol, niter=niter, lmax=maxl, dtype=np.complex128)
        win2 = (win2_alm_T, win2_alm_P)

        bl1 = (bl1["T"], bl1["E"])
        bl2 = (bl2["T"], bl2["E"])

        spec_for_ducc = []
        bl = []
        for i in range(2):
            for j in range(2):
                spec_for_ducc.append(curvedsky.alm2cl(win1[i], win2[j]))
                bl.append(bl1[i][2:lmax] * bl2[j][2:lmax]) # TODO: reconsider pspipe conventions
        specs_for_ducc.append(spec_for_ducc)
        bls.append(bl)

if not args.old:
    specs_for_ducc = np.array(specs_for_ducc)
    bls = np.repeat(bls, (1, 1, 1, 2), axis=1) # (nspec, 4, nl) -> (nspec, 5, nl)
    bls = bls[..., None, :] # (nspec, 5, nl) -> (nspec, 5, 1, nl)

    mcm = so_mcm.ducc_couplings(specs_for_ducc, lmax, spec_index=(0, 1, 2, 3),
                                mat_index=(0, 1, 2, 3, 4), dtype=np.float64,
                                coupling=False, pspy_index_convention=True)

    # apply total-diagonal beams on the right
    mcm *= bls

    # get the binned mcms
    bin_lo, bin_hi, _, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
    nbins = len(bin_hi)

    # loop over map pairs for the post-processing and saving
    for t, task in enumerate(subtasks):
        sv1, ar1, sv2, ar2 = sv1_list[task], ar1_list[task], sv2_list[task], ar2_list[task]
    
        if binned_mcm:
            mbb_array = np.zeros((5, nbins, nbins))
            Bbl_array = np.zeros((5, nbins, lmax))

            for i in range(5):
                so_mcm.mcm_fortran.bin_mcm(mcm[t, i].T,
                                           bin_lo,
                                           bin_hi,
                                           bin_size,
                                           mbb_array[i].T,
                                           doDl)

                so_mcm.mcm_fortran.binning_matrix(mcm[t, i].T,
                                                  bin_lo,
                                                  bin_hi,
                                                  bin_size,
                                                  Bbl_array[i].T,
                                                  doDl)
        else:
            mbb_array = mcm[t]
            Bbl_array = np.zeros((5, nbins, lmax))
            
            for i in range(4): # leave the last (the '--' entry) zero
                so_mcm.mcm_fortran.binning_matrix(np.eye(mcm.shape[-1]).T,
                                                  bin_lo,
                                                  bin_hi,
                                                  bin_size,
                                                  Bbl_array[i].T,
                                                  doDl)

        # TODO: the 2x2 block wastes 2x the space and computation on zeros
        mbb = so_mcm.get_coupling_dict(mbb_array)
        Bbl = so_mcm.get_coupling_dict(Bbl_array)

        mbb_inv = {}
        for spin_pair in mbb:
            mbb_inv[spin_pair] = np.linalg.inv(mbb[spin_pair])
            if binned_mcm:
                Bbl[spin_pair] = np.dot(mbb_inv[spin_pair], Bbl[spin_pair])

        prefix = f"{mcm_dir}/{sv1}_{ar1}x{sv2}_{ar2}"
        so_mcm.save_coupling(prefix, mbb_inv, Bbl, spin_pairs=mbb_inv.keys())