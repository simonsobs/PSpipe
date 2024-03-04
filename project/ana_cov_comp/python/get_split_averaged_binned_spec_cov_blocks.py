"""
This script stitches together the previously produced lowest-level split-
averaged covariance blocks (at the pseudospectrum level) into array-level
powerspectrum blocks. To do so it sandwiches two pseudo-to-spec operators
that capture the PSpipe, including mode-decoupling, binning (with possible)
Dl factors, and kspace deconvolving.
"""
import sys
import numpy as np
from pspipe_utils import log, pspipe_list, covariance as psc
from pspy import so_dict, pspy_utils
from itertools import product
import os

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

log = log.get_logger(**d)

pspipe_ops_dir = d['pspipe_operators_dir']
covariances_dir = d['covariances_dir']
pspy_utils.create_directory(covariances_dir)

sv2arrs2chans = pspipe_list.get_survey_array_channel_map(d)

lmax = 8500 # FIXME
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

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
for sv1 in sv2arrs2chans:
    for ar1 in sv2arrs2chans[sv1]:
        for chan1 in sv2arrs2chans[sv1][ar1]:
            sv_ar_chans.append((sv1, ar1, chan1)) 
            for split1 in range(len(d[f'maps_{sv1}_{ar1}_{chan1}'])):
                for pol1 in ('T', 'E', 'B'):                    
                    coadd_info = (sv1, ar1, chan1, pol1)
                    if coadd_info not in coadd_infos:
                        coadd_infos.append(coadd_info)
                    else:
                        pass # coadd_infos are not unique because of splits

canonized_combos = {}

for sv_ar_chan1, sv_ar_chan2, sv_ar_chan3, sv_ar_chan4 in product(sv_ar_chans, repeat=4):

    # canonize the coadded fields
    sv_ar_chani, sv_ar_chanj, sv_ar_chanp, sv_ar_chanq = psc.canonize_disconnected_4pt(
        sv_ar_chan1, sv_ar_chan2, sv_ar_chan3, sv_ar_chan4, sv_ar_chans
    )

    if (sv_ar_chani, sv_ar_chanj, sv_ar_chanp, sv_ar_chanq) not in canonized_combos:
        canonized_combos[(sv_ar_chani, sv_ar_chanj, sv_ar_chanp, sv_ar_chanq)] = [(sv_ar_chan1, sv_ar_chan2, sv_ar_chan3, sv_ar_chan4)]
    else:
        canonized_combos[(sv_ar_chani, sv_ar_chanj, sv_ar_chanp, sv_ar_chanq)].append((sv_ar_chan1, sv_ar_chan2, sv_ar_chan3, sv_ar_chan4))

np.save(f'{covariances_dir}/canonized_split_averaged_binned_spec_cov_combos.npy', canonized_combos)

start, stop = 0, len(canonized_combos)
if len(sys.argv) == 4:
    log.info(f'computing only the covariance blocks: ' + 
             f'{int(sys.argv[2])}:{int(sys.argv[3])} of {len(canonized_combos)}')
    start, stop = int(sys.argv[2]), int(sys.argv[3])

# # main loop, we will stitch all pols together here
# # iterate over all pairs/orders of channels
for i in range(start, stop):
    (sv_ar_chani, sv_ar_chanj, sv_ar_chanp, sv_ar_chanq) = list(canonized_combos.keys())[i]

    spec_cov_fn = f"{covariances_dir}/analytic_cov_{'_'.join(sv_ar_chani)}x{'_'.join(sv_ar_chanj)}"
    spec_cov_fn += f"_{'_'.join(sv_ar_chanp)}x{'_'.join(sv_ar_chanq)}.npy"

    if os.path.isfile(spec_cov_fn):
        log.info(f'{spec_cov_fn} exists, skipping')
    else:
        log.info(f'Generating {spec_cov_fn}')

        pseudo_cov = np.zeros((len(spectra) * (lmax - 2), len(spectra) * (lmax - 2)), dtype=np.float64)

        for ridx, (poli, polj) in enumerate(spectra):
            for cidx, (polp, polq) in enumerate(spectra):
                coadd_infoi = (*sv_ar_chani, poli)
                coadd_infoj = (*sv_ar_chanj, polj)
                coadd_infop = (*sv_ar_chanp, polp)
                coadd_infoq = (*sv_ar_chanq, polq)

                coadd_infoi, coadd_infoj, coadd_infop, coadd_infoq = psc.canonize_disconnected_4pt(
                    coadd_infoi, coadd_infoj, coadd_infop, coadd_infoq, coadd_infos
                    )

                pseudo_cov_fn = f"{covariances_dir}/pseudo_cov_{'_'.join(coadd_infoi)}x{'_'.join(coadd_infoj)}"
                pseudo_cov_fn += f"_{'_'.join(coadd_infop)}x{'_'.join(coadd_infoq)}.npy"

                log.info(f'Loading {pseudo_cov_fn}')
                pseudo_cov[ridx*(lmax - 2):(ridx+1)*(lmax - 2), cidx*(lmax - 2):(cidx+1)*(lmax - 2)] = \
                    np.load(pseudo_cov_fn)[2:lmax, 2:lmax]

        Fij_fn = f"{pspipe_ops_dir}/Finv_Pbl_Minv_{'_'.join(sv_ar_chani)}x{'_'.join(sv_ar_chanj)}.npy"
        log.info(f'Loading {Fij_fn}')
        Fij = np.load(Fij_fn)

        Fpq_fn = f"{pspipe_ops_dir}/Finv_Pbl_Minv_{'_'.join(sv_ar_chanp)}x{'_'.join(sv_ar_chanq)}.npy"
        log.info(f'Loading {Fpq_fn}')
        Fpq = np.load(Fpq_fn)

        spec_cov = Fij @ pseudo_cov @ Fpq.T
        np.save(spec_cov_fn, spec_cov)