description = """parametric nulls"""

import sys
from os.path import join as opj
import argparse

import numpy as np
from scipy import stats
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

from pspy import so_dict, so_spectra, so_mcm, so_cov, pspy_utils
from pspipe_utils import log, covariance, dict_utils, null_utils, pspipe_list

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
parser.add_argument('--absolute-params', action='store_true',
                    help='Params are absolute not nulled')
args = parser.parse_args()

d = so_dict.so_dict()
d.read_from_file(args.paramfile)
log = log.get_logger(**d)

null_params = not args.absolute_params

spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_", from_spec_nullgroups=d['spectra_list_from_spec_nullgroups'])
spec_dir = d['spec_dir']
bestfit_dir = d['best_fits_dir']
mcm_dir = d['mcm_dir']
cov_dir = d['cov_dir']
null_test_dir = d["nulls_dir"]
plot_dir = opj(d['plots_dir'], 'nulls')
pspy_utils.create_directory(null_test_dir)
pspy_utils.create_directory(plot_dir)

lmax = d['lmax']
Dl = d['type']
binned_mcm = d['binned_mcm']

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
bin_low, bin_high, bin_mean, bin_size = pspy_utils.read_binning_file(d['binning_file'], lmax)
nbins = len(bin_mean)

pspipespec_mpair2theoryl = {}
for spec_name in spec_name_list:
    l, _theory_dict = so_spectra.read_ps(opj(bestfit_dir, f"cmb_and_fg_{spec_name}.dat"), spectra=spectra,
                                    return_type='Dl' if Dl else 'Cl', return_dtype=np.float64)
    assert l[0] == 2, f'expected l[0]=2, got {l[0]=}'

    for pspipespec, theoryl in _theory_dict.items():
        _theory_dict[pspipespec] = theoryl[:lmax]   
        pspipespec_mpair2theoryl[pspipespec, spec_name] = theoryl[:lmax]   
    
# get full data vector for this sim, append to list
x_ar_data_vec = np.load(opj(spec_dir, "x_ar_data_vec.npy"))
x_ar_theory_vec = np.load(opj(bestfit_dir, "x_ar_theory_vec.npy"))
x_ar_res_vec = (x_ar_data_vec - x_ar_theory_vec)[:, None] # for slicing and matrix math

x_ar_cov = np.load(opj(cov_dir, "x_ar_analytic_cov.npy"))

log.info(f'{x_ar_data_vec.shape=}, {x_ar_data_vec.dtype=}, {x_ar_cov.shape=}, {x_ar_cov.dtype=}')

spec2nullgroup2nullflag_mpairs = pspipe_list.get_spec2nullgroup2nullflag_mpairs(d)

# TODO: improve this. should work for lat_iso alone and dr6xlat_iso
_, sv_list, map_list = pspipe_list.get_arrays_list(d)
sv_map_list = ['_'.join(sv_m) for sv_m in zip(sv_list, map_list)]

spectra_cuts = {
    "legacy_f100": {'T': [300, 1500], 'P': [300, 1500]},
    "legacy_f143": {'T': [300, 2000], 'P': [300, 2000]},
    "legacy_f217": {'T': [300, 2500], 'P': [300, 2500]},
    "legacy_f353": {'T': [300, lmax], 'P': [300, lmax]},
    "dr6_pa4_f220": {'T': [975, lmax], 'P': [lmax, lmax]},
    "dr6_pa5_f090": {'T': [975, lmax], 'P': [975, lmax]},
    "dr6_pa5_f150": {'T': [775, lmax], 'P': [775, lmax]},
    "dr6_pa6_f090": {'T': [975, lmax], 'P': [975, lmax]},
    "dr6_pa6_f150": {'T': [575, lmax], 'P': [575, lmax]},
    "lat_iso_i1_f090": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_i1_f150": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_i3_f090": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_i3_f150": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_i4_f090": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_i4_f150": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_i6_f090": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_i6_f150": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_c1_f220": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_c1_f280": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_i5_f220": {'T': [300, lmax], 'P': [300, lmax]},
    "lat_iso_i5_f280": {'T': [300, lmax], 'P': [300, lmax]},
    }

paramtypes2nullgroup2maps_nullflag_splinetype_splinekwargs_norms_xknots = {
    'delta_T': {
        'f090':[[m for m in sv_map_list if ('f090' in m) and ('lat_iso' in m)], null_params, 'cubic', {'bc_type': 'natural'}, [-0.01, 0.01/1000], [2, 600, 1200, 1800, 3600, lmax+1]],
        'f150':[[m for m in sv_map_list if ('f150' in m) and ('lat_iso' in m)], null_params, 'cubic', {'bc_type': 'natural'}, [-0.01, 0.01/1000], [2, 600, 1200, 1800, 3600, lmax+1]],
        'f220':[[m for m in sv_map_list if ('f220' in m) and ('lat_iso' in m)], null_params, 'cubic', {'bc_type': 'natural'}, [-0.01, 0.01/1000], [2, 600, 1200, 1800, 3600, lmax+1]],
        'f280':[[m for m in sv_map_list if ('f280' in m) and ('lat_iso' in m)], null_params, 'cubic', {'bc_type': 'natural'}, [-0.01, 0.01/1000], [2, 600, 1200, 1800, 3600, lmax+1]],
        },
    'delta_P': {
        'MF':[[m for m in sv_map_list if ('f090' in m or 'f150' in m) and ('lat_iso' in m)], null_params, 'cubic', {'bc_type': 'natural'}, [0.01, 0.01/1000], [2, 600, 1200, 1800, 3600, lmax+1]],
        'UHF':[[m for m in sv_map_list if ('f220' in m or 'f280' in m) and ('lat_iso' in m)], null_params, 'cubic', {'bc_type': 'natural'}, [0.01, 0.01/1000], [2, 600, 1200, 1800, 3600, lmax+1]],
        },
    'gamma_TE': {
        'MF':[[m for m in sv_map_list if ('f090' in m or 'f150' in m) and ('lat_iso' in m)], null_params, 'cubic', {'bc_type': 'natural'}, [0.01, 0.01/1000], [2, 600, 1200, 1800, 3600, lmax+1]],
        'UHF':[[m for m in sv_map_list if ('f220' in m or 'f280' in m) and ('lat_iso' in m)], null_params, 'cubic', {'bc_type': 'natural'}, [0.01, 0.01/1000], [2, 600, 1200, 1800, 3600, lmax+1]],
        },
    'gamma_TB': {
        'MF':[[m for m in sv_map_list if ('f090' in m or 'f150' in m) and ('lat_iso' in m)], null_params, 'cubic', {'bc_type': 'natural'}, [0.01, 0.01/1000], [2, 600, 1200, 1800, 3600, lmax+1]],
        'UHF':[[m for m in sv_map_list if ('f220' in m or 'f280' in m) and ('lat_iso' in m)], null_params, 'cubic', {'bc_type': 'natural'}, [0.01, 0.01/1000], [2, 600, 1200, 1800, 3600, lmax+1]],
        },
    'gamma_EB': {
        'MF': [[m for m in sv_map_list if ('f090' in m or 'f150' in m) and ('lat_iso' in m)], null_params, 'linear', {}, [2*np.pi/180], [1000]],
        'UHF': [[m for m in sv_map_list if ('f220' in m or 'f280' in m) and ('lat_iso' in m)], null_params, 'linear', {}, [2*np.pi/180], [1000]],
    }
}

add_delta_P_to_delta_T = False
###

### Get the null likelihood ###
# this gets the typical pspipe indexing of the x_ar_datavec 
spectra_to_use = spectra # use all spectra for now

bin_out_dict, indices = covariance.get_indices(bin_low,
                                               bin_high,
                                               bin_mean,
                                               spec_name_list,
                                               spectra_cuts=spectra_cuts,
                                               spectra_order=spectra,
                                               selected_spectra=spectra_to_use,
                                               use_bin_edges=True)

# this orders into spec_bin blocks (which are sparse for nulls)
spec_bin2validmpairs_idxsintopspipevec = {}

for (validmpair, pspipespec), (idxs, binmeans) in bin_out_dict.items():
    # Zip the arrays to process individual data points
    for idx, b in zip(idxs, binmeans):
        if pspipespec in ['ET', 'BT', 'BE']: # put in spec-major order
            _spec = pspipespec[::-1]
            _validmpair = 'x'.join(validmpair.split('x')[::-1])
        else:
            _spec = pspipespec
            _validmpair = validmpair
        
        key = (_spec, b)
        
        if key not in spec_bin2validmpairs_idxsintopspipevec:
            spec_bin2validmpairs_idxsintopspipevec[key] = ([], [])
        
        spec_bin2validmpairs_idxsintopspipevec[key][0].append(_validmpair)
        spec_bin2validmpairs_idxsintopspipevec[key][1].append(indices[idx]) # indices are after the cuts

# this gets the matrices that perform the nulling

# NOTE: in below we always are doing matrices with a data2null convention
# NOTE: even though diagonal in the block spec_bin indices, this makes it so we can use our sparse linalg stuff. alternatively could have done one-off for loops
spec_nullgroup_bin2spec_bin2selproj = {} # (N-1) x all_mpairs, where N = num of mpairs in this nullgroup that are also in this specbin

# first iterate over specbin2mpairs_idxs:
#   - for each specbin, get the spec and therefore the null groups.
#     - for each null_group, get the intersection of the mpairs in the null_group and the mpairs at this specbin
#     - form P from the length of the intersection, and mpair_selector from the pairwise indices of the intersection (in the null2data direction, even though we don't use this)
for spec_bin, (validmpairs, _) in spec_bin2validmpairs_idxsintopspipevec.items():
    spec, b = spec_bin 
    if spec in spec2nullgroup2nullflag_mpairs:
        for nullgroup, (nullflag, nullgroupmpairs) in spec2nullgroup2nullflag_mpairs[spec].items():
            validnullgroupmpairs, _, validnullgroupmpairidxsintovalidmpairs = np.intersect1d(nullgroupmpairs, validmpairs, return_indices=True)

            # get sel: from validmpairs to validnullmpairs
            selproj = np.zeros((len(validnullgroupmpairs), len(validmpairs)))
            for validnullmpairidx, validmpairidx in enumerate(validnullgroupmpairidxsintovalidmpairs):
                selproj[validnullmpairidx, validmpairidx] = 1

            # get proj: from validmpairs to null columnspace 
            if nullflag:
                N = len(validnullgroupmpairs)
                if N <= 1:
                    continue
                P = null_utils.get_orthogonal_projector(np.ones((N, 1)))
                selproj = P.T @ selproj # allmpairs to null columnspace

            key = (spec, nullgroup, b)
            if key in spec_nullgroup_bin2spec_bin2selproj:
                raise ValueError(f'Already encountered {key}')
            spec_nullgroup_bin2spec_bin2selproj[key] = {spec_bin: selproj}

# i can use specbin2mpairs_idxs to also get the blocked datavector and cov
spec_bin2datavec = {}
spec_bin2spec_bin2datacov = {}
for spec_bin1, (_, idxs1) in spec_bin2validmpairs_idxsintopspipevec.items():
    spec_bin2datavec[spec_bin1] = x_ar_data_vec[idxs1]
    spec_bin2spec_bin2datacov[spec_bin1] = {}
    for spec_bin2, (_, idxs2) in spec_bin2validmpairs_idxsintopspipevec.items():
        spec_bin2spec_bin2datacov[spec_bin1][spec_bin2] = x_ar_cov[np.ix_(idxs1, idxs2)]

# also need transpose for cov
spec_nullgroup_bin2spec_bin2selproj_T = so_mcm.sparse_dict_mat_transpose(spec_nullgroup_bin2spec_bin2selproj, copy=True) # copy for faster striding?

# get the nulls and nullcov
spec_nullgroup_bin2nulldatavec = so_mcm.sparse_dict_mat_matmul_sparse_dict_vec(spec_nullgroup_bin2spec_bin2selproj, spec_bin2datavec)
spec_nullgroup_bin2spec_nullgroup_bin2nulldatacov = so_mcm.sparse_dict_mat_matmul_sparse_dict_mat(spec_nullgroup_bin2spec_bin2selproj, spec_bin2spec_bin2datacov)
spec_nullgroup_bin2spec_nullgroup_bin2nulldatacov = so_mcm.sparse_dict_mat_matmul_sparse_dict_mat(spec_nullgroup_bin2spec_nullgroup_bin2nulldatacov, spec_nullgroup_bin2spec_bin2selproj_T)

# get chi2 of the data
nulldatavec = np.concatenate(list(spec_nullgroup_bin2nulldatavec.values()), dtype=np.float64)
nulldatacov = so_mcm.sparse_dict_mat2dense_array(spec_nullgroup_bin2spec_nullgroup_bin2nulldatacov, np.float64)

inv_nulldatacov = np.linalg.inv(nulldatacov)
chi2 = nulldatavec @ inv_nulldatacov @ nulldatavec
log.info(f'{chi2=:0.3f} for {len(nulldatavec)} dof, pte={stats.chi2.sf(chi2, len(nulldatavec)):0.3g}')

np.linalg.cholesky(nulldatacov)
###

### Get the systematic model ###
paramtypes2nparamsaknots_to_mparamsyl = {}
nparamlist = []

paramtypes2numparams = {}
total_params = 0
for paramtype, nullgroup2maps_nullflag_splinetype_splinekwargs_norms_xknots in paramtypes2nullgroup2maps_nullflag_splinetype_splinekwargs_norms_xknots.items():

    nullgroup2mlnk = {}
    numparams = 0
    for nullgroup, (ngmaps, nullflag, splinetype, splinekwargs, norms, xknots) in nullgroup2maps_nullflag_splinetype_splinekwargs_norms_xknots.items():
        # get mapping from amplitudes (aknots) to yknots
        nsplines = len(xknots)
        
        constvec = np.full(len(xknots), norms[0])[:, None] # column vector
        aknots_to_yknots = constvec
        
        if nsplines > 1:
            m = norms[1] # 1% per 1000 ell
            dx = xknots[-1] - xknots[0]
            dy = m * dx
            xint = 1000 # try to make slope and cal roughly uncorrelated, pick an easy number
            slopefunc = lambda x: m*(x - xint)
            slopevec = np.array([slopefunc(x) for x in xknots])[:, None] # column vector
            aknots_to_yknots = np.append(aknots_to_yknots, slopevec, axis=1)

        if nsplines > 2:
            remaindervecs = null_utils.get_orthogonal_projector(np.concatenate([constvec, slopevec], axis=1))
            for i, norm in enumerate(norms[2:]):
                remaindervecs[:, i] *= norm
            aknots_to_yknots = np.append(aknots_to_yknots, remaindervecs, axis=1)

        # get mapping from yknots to yl
        l = np.arange(2, lmax+2) # to match against Bbl
        _l = np.arange(xknots[0], xknots[-1]+1) # to match against the xknots, outside this range will be "constant extrapolated"
        yknots_to_yl = np.zeros((len(l), len(xknots)))
        for i, xknot in enumerate(xknots):
            _yknotsi = np.zeros(len(xknots))
            _yknotsi[i] = 1
            _yli = np.zeros(lmax)
            if splinetype == 'linear':
                _yli[_l - l[0]] = np.interp(_l, xknots, _yknotsi, **splinekwargs)
            if splinetype == 'cubic':
                _yli[_l - l[0]] = CubicSpline(xknots, _yknotsi, extrapolate=False, **splinekwargs)(_l)

            # "constant extrapolation"
            _yli[:_l[0]-l[0]] = _yli[_l[0] - l[0]]
            _yli[_l[-1]-l[0]:] = _yli[_l[-1]-l[0]]

            yknots_to_yl[:, i] = _yli

        aknots_to_yl = yknots_to_yl @ aknots_to_yknots

        # get possible mapping from null amplitudes (nparams) to amplitudes (mparams)
        constvec = np.ones(len(ngmaps))[:, None] # column vector
        nparams_to_mparams = null_utils.get_orthogonal_projector(constvec)
        if not nullflag:
            nparams_to_mparams = np.append(constvec, nparams_to_mparams, axis=1)

        mlnk = np.einsum('mn,lk->mlnk', nparams_to_mparams, aknots_to_yl)
        nullgroup2mlnk[nullgroup] = mlnk
        numparams += np.prod(mlnk.shape[-2:])

    paramtypes2numparams[paramtype] = numparams
    total_params += numparams

    nparamsaknots_to_mparamsyl = np.zeros((len(sv_map_list), lmax, numparams))
    nullgroup_paramstartidx = 0
    for nullgroup, (ngmaps, nullflag, _, _, _, xknots) in nullgroup2maps_nullflag_splinetype_splinekwargs_norms_xknots.items():
        mlnk = nullgroup2mlnk[nullgroup]
        for i, (n, k) in enumerate(np.ndindex(mlnk.shape[-2:])):
            for ngmapidx, ngmap in enumerate(ngmaps):
                mapsindex = sv_map_list.index(ngmap)
                nparamsaknots_to_mparamsyl[mapsindex, :, nullgroup_paramstartidx + i] = mlnk[ngmapidx, :, n, k]
            
            if not nullflag:
                if n == 0:
                    nname = 'constmap'
                else:
                    nname = f'diffmap{n-1}'
            else:
                nname = f'diffmap{n}'

            if k == 0:
                kname = 'constell'
            elif k == 1:
                kname = 'slopeell'
            else:
                kname = f'diffknot{k-2}'
            nparamlist.append(f'{paramtype}_{nullgroup}_{nname}_{kname}')

        nullgroup_paramstartidx += np.prod(mlnk.shape[-2:])

    paramtypes2nparamsaknots_to_mparamsyl[paramtype] = nparamsaknots_to_mparamsyl

# need thing that stores how each m projects into each mpair.
# complications are: need templates at spectrum level, and need to mix paramtypes.
# will do this block by block so can load each Bbl once and not hold them all in memory
def polout_polin2paramtypes_signs(polout, polin, add_delta_P_to_delta_T=False):
    if polout + polin == 'TT':
        return [('delta_T',), (1,)]
    if add_delta_P_to_delta_T:
        if polout + polin in ('EE', 'BB'):
            return [('delta_T', 'delta_P'), (1, 1)]
    else:
        if polout + polin in ('EE', 'BB'):
            return [('delta_P',), (1,)]
    if polout + polin in ('TE', 'ET'):
        return [('gamma_TE',), (1,)]
    if polout + polin in ('TB', 'BT'):
        return [('gamma_TB',), (1,)]
    if polout + polin == 'BE':
        return [('gamma_EB',), (1,)]
    if polout + polin == 'EB':
        return [('gamma_EB',), (-1,)]

paramtypes = list(paramtypes2nullgroup2maps_nullflag_splinetype_splinekwargs_norms_xknots.keys())
pspipespec_mpair_t2nk_to_b = {}
for mpair in spec_name_list:
    mi, mj = mpair.split('x')

    # paramtype_map_to_pspipespec (for this mpair). need to fill all 
    # pspipespecs to be multiplied by Bbl which has specxspec stuff. 
    # (assume it never has mpairxmpair stuff...)
    pspipespec2t2lnk = {}
    for (poli, polj) in spectra:
        pspipespec2t2lnk[poli + polj] = {}

        tm2l = {} # the reason lm depends on t is because the l template could change
        
        if mi in sv_map_list: # cludge for spec_name_list not being based on my selected maps
            for polp in 'TEB':
                ts, signs = polout_polin2paramtypes_signs(poli, polp, add_delta_P_to_delta_T=add_delta_P_to_delta_T) # turning p into i
                for (t, sign) in zip(ts, signs):
                    if (t, mi) not in tm2l:
                        tm2l[t, mi] = np.zeros(lmax)
                    tm2l[t, mi] += sign * pspipespec_mpair2theoryl[polp + polj, mpair] # turning p into i
        if mj in sv_map_list: # cludge for spec_name_list not being based on my selected maps
            for polq in 'TEB':
                ts, signs = polout_polin2paramtypes_signs(polj, polq, add_delta_P_to_delta_T=add_delta_P_to_delta_T) # turning q into j
                for (t, sign) in zip(ts, signs):
                    if (t, mj) not in tm2l:
                        tm2l[t, mj] = np.zeros(lmax)
                    tm2l[t, mj] += sign * pspipespec_mpair2theoryl[poli + polq, mpair] # turning q into j

        for (t, m), l in tm2l.items():
            try:
                mlnk = paramtypes2nparamsaknots_to_mparamsyl[t]
            except KeyError:
                continue # we are setting this missing paramtype to 0

            if t not in pspipespec2t2lnk[poli + polj]:
                pspipespec2t2lnk[poli + polj][t] = 0
            pspipespec2t2lnk[poli + polj][t] += l[:, None] * mlnk[sv_map_list.index(m)] # spec, mpair, and t specific to the matrices themselves

    Bbl = np.load(opj(mcm_dir, f'{mpair}_Bbl.npy'))
    if binned_mcm:
        pspipespec2pspipespec2Bbl = so_mcm.get_spec2spec_sparse_dict_mat_from_spin2spin_array(Bbl, spectra)
    else:
        pspipespec2pspipespec2Bbl = so_mcm.get_block_diagonal_sparse_dict_mat_from_array(Bbl, spectra)

    for pspipespecij in pspipespec2pspipespec2Bbl:
        for pspipespecpq, Bbl in pspipespec2pspipespec2Bbl[pspipespecij].items():
            for t, lnk in pspipespec2t2lnk[pspipespecpq].items():
                key = (pspipespecij, mpair, t)
                if key not in pspipespec_mpair_t2nk_to_b:
                    pspipespec_mpair_t2nk_to_b[key] = 0
                
                nk_to_b = Bbl @ pspipespec2t2lnk[pspipespecpq][t]
                pspipespec_mpair_t2nk_to_b[key] += nk_to_b

# finally, get the pspipe-ordered operator
x_ar_knots_to_data = np.zeros((len(x_ar_data_vec), total_params))
data_idx = 0
for spec in spectra:
    for mpair in spec_name_list:
        mi, mj = mpair.split('x')
        if spec in ['ET', 'BT', 'BE'] and mi == mj:
            continue
        param_idx = 0
        for t, numparams in paramtypes2numparams.items():
            try:
                nk_to_b = pspipespec_mpair_t2nk_to_b[spec, mpair, t]
                numdata = nk_to_b.shape[0]
                assert numparams == nk_to_b.shape[1], \
                    f'expected {numparams=}, got {nk_to_b.shape[1]=}'
                x_ar_knots_to_data[data_idx:data_idx + numdata, param_idx:param_idx + numparams] = nk_to_b
            except KeyError: # otherwise leave matrix as 0's
                numdata = nbins # TODO: make robust against differing number of bins per spectrum including if spectrum is missing 
            param_idx += numparams
        data_idx += numdata 

spec_bin2knots_to_data = {}
for spec_bin, (_, idxs) in spec_bin2validmpairs_idxsintopspipevec.items():
    spec_bin2knots_to_data[spec_bin] = x_ar_knots_to_data[idxs]

spec_nullgroup_bin2knots_to_nulldata = so_mcm.sparse_dict_mat_matmul_sparse_dict_vec(spec_nullgroup_bin2spec_bin2selproj, spec_bin2knots_to_data, dense=True)
log.info(f'{spec_nullgroup_bin2knots_to_nulldata.shape=}, {nulldatavec.shape=}, {nulldatacov.shape=}')

inv_optnkcov = spec_nullgroup_bin2knots_to_nulldata.T @ inv_nulldatacov @ spec_nullgroup_bin2knots_to_nulldata
optnkcov = np.linalg.inv(inv_optnkcov)
optnk = optnkcov @ spec_nullgroup_bin2knots_to_nulldata.T @ inv_nulldatacov @ nulldatavec
chi2 = optnk @ inv_optnkcov @ optnk

np.linalg.cholesky(optnkcov)

print(f'cond={np.linalg.cond(optnkcov):0.3f}')
print(f'{chi2=:0.3f} for {len(optnk)} dof, pte={stats.chi2.sf(chi2, len(optnk)):0.3g}')

np.save(opj(null_test_dir, "optnk.npy"), optnk)
np.save(opj(null_test_dir, "optnkcov.npy"), optnkcov)

x_ar_parametricnull_data_vec = x_ar_knots_to_data @ optnk
x_ar_parametricnull_cov = x_ar_knots_to_data @ optnkcov @ x_ar_knots_to_data.T

np.save(opj(null_test_dir, "x_ar_parametricnull_data_vec.npy"), x_ar_parametricnull_data_vec)
np.save(opj(null_test_dir, "x_ar_parametricnull_cov.npy"), x_ar_parametricnull_cov)
###

### Plot derived things ###
plt.imshow(so_cov.cov2corr(optnkcov, remove_diag=True))
plt.colorbar()
plt.title('knots correlation matrix')
plt.savefig(opj(plot_dir, 'knots_correlation_matrix.png'))
plt.close()

paramtype2optnk = {}
paramtype2paramtype2optnkcov = {}

param_idxi = 0
for ti in paramtypes:
    numparamsi = paramtypes2numparams[ti]
    paramtype2optnk[ti] = optnk[param_idxi:param_idxi + numparamsi]
    
    paramtype2paramtype2optnkcov[ti] = {}
    
    param_idxj = 0
    for tj in paramtypes:
        numparamsj = paramtypes2numparams[tj]
        paramtype2paramtype2optnkcov[ti][tj] = optnkcov[param_idxi:param_idxi + numparamsi, param_idxj:param_idxj + numparamsj]

        param_idxj += numparamsj
    
    param_idxi += numparamsi

paramtype2optsplines = {}
paramtype2optsplineserr = {}
for t in paramtypes:
   mlnk = paramtypes2nparamsaknots_to_mparamsyl[t]
   mlnk = mlnk.reshape(*mlnk.shape[:2], -1)
   paramtype2optsplines[t] = np.einsum('mlk,k->ml', mlnk, paramtype2optnk[t])
   paramtype2optsplineserr[t] = np.einsum('mlK,Kk,mlk->ml', mlnk, paramtype2paramtype2optnkcov[t][t], mlnk)

paramtypes2plotnorms_plotylims_plotylabels = {
    'delta_T': (0.01, 25, '%'), 
    'delta_P': (0.01, 25, '%'),
    'gamma_TE': (0.01, 10, '%'),
    'gamma_TB': (0.01, 10, '%'),
    'gamma_EB': (2*np.pi/180, 5, 'deg')
}

l = np.arange(2, lmax+2)
for paramtype, y in paramtype2optsplines.items():
    for m in range(len(sv_map_list)):
        yerr = paramtype2optsplineserr[paramtype]

        plotnorm, ylim, ylabel = paramtypes2plotnorms_plotylims_plotylabels[paramtype]

        plt.figure(figsize=(8, 4))
        plt.axhline(0, color='k', alpha=0.5)
        plt.plot(l, y[m] / plotnorm, color='C0')
        plt.fill_between(l, (y[m] - yerr[m]**0.5) / plotnorm, (y[m] + yerr[m]**0.5) / plotnorm, alpha=0.3, color='C0', edgecolor=None)
        plt.fill_between(l, (y[m] - 2*yerr[m]**0.5) / plotnorm, (y[m] + 2*yerr[m]**0.5) / plotnorm, alpha=0.3, color='C0', edgecolor=None)
        plt.grid()
        plt.xlim(0, 7000)
        if ylim is not None:
            plt.ylim(-ylim, ylim)
        plt.ylabel(ylabel)
        plt.title(f'{paramtype, sv_map_list[m]}')
        plt.savefig(opj(plot_dir, f'{paramtype}_{sv_map_list[m]}.png'))
        plt.close()
###