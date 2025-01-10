import importlib
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import sacc
from pspy import pspy_utils, so_dict, so_mcm


def get_ell_covar(e1a, f1a, p1a, e1b, f1b, p1b, e2a, f2a, p2a, e2b, f2b, p2b, covar_in, n_ells, id_cov_order):

    nx_1 = e1a + "_" + f1a + "x" + e1b + "_" + f1b
    nx_2 = e2a + "_" + f2a + "x" + e2b + "_" + f2b
    px_1 = p1a + p1b
    px_2 = p2a + p2b

    # We fake the covariance for these combinations
    if px_1 in ["TB", "EB", "BT", "BE", "BB"]:
        if px_2 == px_1:
            return np.identity(n_ells)
        else:
            return np.zeros([n_ells, n_ells])
    elif px_2 in ["TB", "EB", "BT", "BE", "BB"]:
        return np.zeros([n_ells, n_ells])

    i1 = id_cov_order[nx_1 + "_" + px_1]
    i2 = id_cov_order[nx_2 + "_" + px_2]

    return covar_in[i1][ :, i2,:]

def get_bbl(ea, fa, pa, eb, fb, pb):

    spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

    prefix = "%s/%s_%sx%s_%s" % (mcm_dir, ea, fa, eb, fb)
    mbb_inv, Bbl = so_mcm.read_coupling(prefix=prefix, spin_pairs=spin_pairs)

    Bbl_TT = Bbl["spin0xspin0"]
    Bbl_TE = Bbl["spin0xspin2"]
    Bbl_EE = Bbl["spin2xspin2"][:Bbl_TE.shape[0],:Bbl_TE.shape[1]]

    px = pa + pb
    if px in ["EE", "EB", "BE", "BB"]:
        return Bbl_EE
    elif px in ["TE", "TB", "ET", "BT"]:
        return Bbl_TE
    else:
        return Bbl_TT


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

experiments = d["experiments"]
type = d["type"]
iStart = d["iStart"]
iStop = d["iStop"]+1

mcm_dir = "mcms"
cov_dir = "covariances"
spec_dir = "spectra"
sacc_dir = "like_products_sacc"

pspy_utils.create_directory(sacc_dir)

pols = ["T", "E", "B"]
map_types = {"T": "0", "E": "e", "B": "b"}

def get_x_iterator():
    for id_ea, ea in enumerate(experiments):
        freqs_a = d["freqs_%s" % ea]
        for id_fa, fa in enumerate(freqs_a):
            for id_eb, eb in enumerate(experiments):
                freqs_b = d["freqs_%s" % eb]
                for id_fb, fb in enumerate(freqs_b):
                    if  (id_ea == id_eb) & (id_fa >id_fb) : continue
                    if  (id_ea > id_eb) : continue
                    for ipa, pa in enumerate(pols):
                        if (ea == eb) & (fa == fb):
                            polsb = pols[ipa:]
                        else:
                            polsb = pols
                        for pb in polsb:
                            yield (ea, fa, eb, fb, pa, pb)

spec_name_list = []
for id_ea, ea in enumerate(experiments):
    freqs_a = d["freqs_%s" % ea]
    for id_fa, fa in enumerate(freqs_a):
        for id_eb, eb in enumerate(experiments):
            freqs_b = d["freqs_%s" % eb]
            for id_fb, fb in enumerate(freqs_b):
                if  (id_ea == id_eb) & (id_fa >id_fb) : continue
                if  (id_ea > id_eb) : continue
                spec_name = "%s_%sx%s_%s" % (ea, fa, eb, fb)
                spec_name_list += [spec_name]

spectra_covar_in = ["TT", "TE", "ET", "EE"]
cov_order = []
for spec in spectra_covar_in:
    for name in spec_name_list:
        cov_order += ["%s_%s" % (name,spec)]

covar_in = np.load("%s/full_analytic_cov.npy" % cov_dir)
id_cov_order = {n:i for i, n in enumerate(cov_order)}
len_x_cov = len(cov_order)
n_ells = len(covar_in) // len_x_cov
covar_in = covar_in.reshape([len_x_cov, n_ells, len_x_cov, n_ells])



nmaps = 0
for id_ea, ea in enumerate(experiments):
    freqs_a = d["freqs_%s" % ea]
    for id_fa, fa in enumerate(freqs_a):
        nmaps += 3

n_x = (nmaps * (nmaps + 1)) // 2
n_cls = n_ells * n_x

cov_full = np.zeros([n_x, n_x, n_ells, n_ells])
for ix1, (e1a, f1a, e1b, f1b, p1a, p1b) in enumerate(get_x_iterator()):
    for ix2, (e2a, f2a, e2b, f2b, p2a, p2b) in enumerate(get_x_iterator()):
        cv = get_ell_covar(e1a, f1a, p1a, e1b, f1b, p1b, e2a, f2a, p2a, e2b, f2b, p2b, covar_in, n_ells, id_cov_order)
        cov_full[ix1, ix2, :, :] = cv
cov_full = np.transpose(cov_full, axes=[0,2,1,3]).reshape([n_cls, n_cls])



for isim in range(iStart,iStop):
    print(isim)

    spec_sacc =  sacc.Sacc()
    if isim == 0:
        cov_sacc =  sacc.Sacc()

    sim_suffix = "%05d"%isim

    for exp in experiments:
        freqs = d["freqs_%s" % exp]
        for f in freqs:
            # dummies file: not in used
            data_bandpasses = {"nu":[f], "b_nu":[1.0]}
            data_beams = {"l":np.arange(10000), "bl":np.ones(10000)}

            spec_sacc.add_tracer("NuMap", "%s_%s_s0" % (exp, f),
                                 quantity="cmb_temperature", spin=0,
                                 nu=data_bandpasses["nu"],
                                 bandpass=data_bandpasses["nu"],
                                 ell=data_beams["l"],
                                 beam=data_beams["bl"])
            spec_sacc.add_tracer("NuMap", "%s_%s_s2" % (exp, f),
                                 quantity="cmb_polarization", spin=2,
                                 nu=data_bandpasses["nu"],
                                 bandpass=data_bandpasses["nu"],
                                 ell=data_beams["l"],
                                 beam=data_beams["bl"])
            if isim == 0:
                cov_sacc.add_tracer("NuMap", "%s_%s_s0" % (exp, f),
                                    quantity="cmb_temperature", spin=0,
                                    nu=data_bandpasses["nu"],
                                    bandpass=data_bandpasses["nu"],
                                    ell=data_beams["l"],
                                    beam=data_beams["bl"])
                cov_sacc.add_tracer("NuMap", "%s_%s_s2" % (exp, f),
                                    quantity="cmb_polarization", spin=2,
                                    nu=data_bandpasses["nu"],
                                    bandpass=data_bandpasses["nu"],
                                    ell=data_beams["l"],
                                    beam=data_beams["bl"])

    data = {}
    for spec_name in spec_name_list:
        na, nb = spec_name.split("x")
        data[na,nb] = {}
        spec = np.loadtxt("%s/%s_%s_cross_%s.dat" % (spec_dir, type, spec_name, sim_suffix), unpack=True)
        ps = {"lbin": spec[0],
              "TT": spec[1],
              "TE": spec[2],
              "TB": spec[3],
              "ET": spec[4],
              "BT": spec[5],
              "EE": spec[6],
              "EB": spec[7],
              "BE": spec[8],
              "BB": spec[9]}
        data[na,nb] = ps

    for i_x, (ea, fa, eb, fb, pa, pb) in enumerate(get_x_iterator()):

        if pa == "T":
            ta_name = "%s_%s_s0" % (ea, fa)
        else:
            ta_name = "%s_%s_s2" % (ea, fa)

        if pb == "T":
            tb_name = "%s_%s_s0" % (eb, fb)
        else:
            tb_name = "%s_%s_s2" % (eb, fb)

        if pb == "T":
            cl_type = "cl_" + map_types[pb] + map_types[pa]
        else:
            cl_type = "cl_" + map_types[pa] + map_types[pb]

        lbin = data["%s_%s" % (ea, fa), "%s_%s" % (eb, fb)]["lbin"]
        cb = data["%s_%s" % (ea, fa), "%s_%s" % (eb, fb)][pa + pb]

        spec_sacc.add_ell_cl(cl_type, ta_name, tb_name, lbin, cb)

        if isim == 0:
            bbl = get_bbl(ea, fa, pa, eb, fb, pb)
            ls_w = np.arange(2, bbl.shape[-1] + 2)
            wins = sacc.BandpowerWindow(ls_w, bbl.T)
            cov_sacc.add_ell_cl(cl_type, ta_name, tb_name,
                                lbin, cb, window=wins)

    if isim == 0:
        # Add metadata
        cov_sacc.metadata["author"] = d.get("author", "SO Collaboration PS Task Force")
        cov_sacc.metadata["date"] = d.get("date", datetime.today().strftime("%Y-%m-%d %H:%M:%S"))
        modules = ["camb", "mflike", "numpy", "pixell", "pspy", "pspipe", "sacc"]
        cov_sacc.metadata["modules"] = str(modules)
        for m in modules:
            cov_sacc.metadata[f"{m}_version"] = importlib.import_module(m).__version__
        # Store dict file as strings
        for k, v in d.items():
            cov_sacc.metadata[k] = str(v)
        cov_sacc.add_covariance(cov_full)
        cov_sacc.save_fits("%s/data_sacc_w_covar_and_Bbl.fits" % sacc_dir, overwrite=True)

    spec_sacc.save_fits("%s/data_sacc_%s.fits" % (sacc_dir, sim_suffix), overwrite=True)
