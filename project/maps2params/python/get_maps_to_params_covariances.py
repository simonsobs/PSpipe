from pspy import pspy_utils, so_dict, so_map, so_mpi, so_mcm, so_spectra, so_cov
import numpy as np
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

window_dir = "windows"
mcm_dir = "mcms"
cov_dir = "covariances"
specDir = "spectra"

experiments = d["experiments"]
clfile = d["clfile"]
lmax = d["lmax"]
type = d["type"]
niter = d["niter"]
binning_file = d["binning_file"]
lcut = d["lcut"]
include_fg = d["include_fg"]
fg_dir = d["fg_dir"]
fg_components = d["fg_components"]

pspy_utils.create_directory(cov_dir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
nbins = len(bin_hi)

lth, ps_th = pspy_utils.ps_lensed_theory_to_dict(clfile,output_type=type,lmax=lmax,start_at_zero=False)

ps_all, nl_all, ns = {}, {}, {}

spec_name = []

for exp in experiments:
    ns[exp] = d["nsplits_%s" % exp]

for id_exp1, exp1 in enumerate(experiments):
    freqs1 = d["freqs_%s" % exp1]
    for id_f1, f1 in enumerate(freqs1):
        for id_exp2, exp2 in enumerate(experiments):
            freqs2 = d["freqs_%s" % exp2]
            for id_f2, f2 in enumerate(freqs2):
                if  (id_exp1 == id_exp2) & (id_f1 >id_f2) : continue
                if  (id_exp1 > id_exp2) : continue

                l, bl1 = np.loadtxt("sim_data/beams/beam_%s_%s.dat" % (exp1, f1), unpack=True)
                l, bl2 = np.loadtxt("sim_data/beams/beam_%s_%s.dat" % (exp2, f2), unpack=True)
                bl1, bl2 = bl1[:lmax], bl2[:lmax]

                for spec in ["TT", "TE", "ET", "EE"]:
                    ps_all["%s_%s" % (exp1,f1), "%s_%s" % (exp2,f2), spec] = bl1 * bl2 * ps_th[spec]

                    if include_fg:
                        if spec == "TT":
                            flth_all = 0
                            for foreground in fg_components:
                                l, flth = np.loadtxt("%s/tt_%s_%sx%s.dat" % (fg_dir, foreground, f1, f2), unpack=True)
                                flth_all += flth[:lmax]
                            ps_all["%s_%s" % (exp1, f1), "%s_%s" %(exp2, f2), spec] = bl1 * bl2 * (ps_th[spec] + flth_all)
                
                    if exp1 == exp2:
                        
                        l, nl_t = np.loadtxt("sim_data/noise_ps/noise_t_%s_%sx%s_%s.dat" % (exp1, f1, exp2, f2), unpack=True)
                        l, nl_pol = np.loadtxt("sim_data/noise_ps/noise_pol_%s_%sx%s_%s.dat" %(exp1, f1, exp2, f2), unpack=True)

                        l, nl_t, nl_pol = l[:lmax], nl_t[:lmax], nl_pol[:lmax]
                        nl_t[:lcut], nl_pol[:lcut] = 0, 0
                        
                        if type == "Dl":
                            fac = l * (l + 1) / (2 * np.pi)
                        else:
                            fac = 1
            
                        if spec == "TT":
                            nl_all["%s_%s" % (exp1, f1), "%s_%s" % (exp2, f2), spec] = nl_t * fac * ns[exp1]
                        if spec == "EE":
                            nl_all["%s_%s" % (exp1, f1), "%s_%s" % (exp2, f2), spec] = nl_pol * fac * ns[exp1]
                        if spec == "TE" or spec == "ET":
                            nl_all["%s_%s" % (exp1, f1), "%s_%s" % (exp2, f2), spec] = nl_t * 0
                    else:
                        nl_all["%s_%s" % (exp1, f1), "%s_%s" % (exp2, f2), spec] = np.zeros(lmax)
    
                    ps_all["%s_%s" % (exp2, f2), "%s_%s" % (exp1, f1), spec] = ps_all["%s_%s" % (exp1, f1), "%s_%s" %(exp2, f2), spec]
                    nl_all["%s_%s" % (exp2, f2), "%s_%s" % (exp1, f1), spec] = nl_all["%s_%s" % (exp1, f1), "%s_%s" %(exp2, f2), spec]

                spec_name += ["%s_%sx%s_%s" % (exp1, f1, exp2, f2)]


na_list, nb_list, nc_list, nd_list = [], [], [], []
ncovs = 0

for sid1, spec1 in enumerate(spec_name):
    for sid2, spec2 in enumerate(spec_name):
        if sid1 > sid2: continue
        print (spec1,spec2)
        na, nb = spec1.split("x")
        nc, nd = spec2.split("x")
        na_list += [na]
        nb_list += [nb]
        nc_list += [nc]
        nd_list += [nd]
        ncovs += 1

nspecs=len(spec_name)

print("number of covariance matrices to compute : %s" % ncovs)
so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=0, imax=ncovs - 1)

for task in subtasks:
    task = int(task)

    na, nb, nc, nd = na_list[task], nb_list[task], nc_list[task], nd_list[task]

    win = {}
    win["Ta"] = so_map.read_map("%s/window_%s.fits" % (window_dir, na))
    win["Tb"] = so_map.read_map("%s/window_%s.fits" % (window_dir, nb))
    win["Tc"] = so_map.read_map("%s/window_%s.fits" % (window_dir, nc))
    win["Td"] = so_map.read_map("%s/window_%s.fits" % (window_dir, nd))
    win["Pa"] = so_map.read_map("%s/window_%s.fits" % (window_dir, na))
    win["Pb"] = so_map.read_map("%s/window_%s.fits" % (window_dir, nb))
    win["Pc"] = so_map.read_map("%s/window_%s.fits" % (window_dir, nc))
    win["Pd"] = so_map.read_map("%s/window_%s.fits" % (window_dir, nd))

    
    coupling = so_cov.cov_coupling_spin0and2_simple(win, lmax, niter=niter)
    
    analytic_cov = np.zeros((4*nbins, 4*nbins))

    # TaTbTcTd
    M_00 = coupling["TaTcTbTd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "TTTT")
    M_00 += coupling["TaTdTbTc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "TTTT")
    analytic_cov[0*nbins:1*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_00, binning_file, lmax)

    # TaEbTcEd
    M_11 = coupling["TaTcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "TTEE")
    M_11 += coupling["TaPdPbTc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "TEET")
    analytic_cov[1*nbins:2*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_11, binning_file, lmax)

    # EaTbEcTd
    M_22 = coupling["PaPcTbTd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "EETT")
    M_22 += coupling["PaTdTbPc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "ETTE")
    analytic_cov[2*nbins:3*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_22, binning_file, lmax)

    # EaEbEcEd
    M_33 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "EEEE")
    M_33 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "EEEE")
    analytic_cov[3*nbins:4*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_33, binning_file, lmax)

    # TaTbTcEd
    M_01 = coupling["TaTcTbPd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "TTTE")
    M_01 += coupling["TaPdTbTc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "TETT")
    analytic_cov[0*nbins:1*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_01, binning_file, lmax)

    # TaTbEcTd
    M_02 = coupling["TaPcTbTd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "TETT")
    M_02 += coupling["TaTdTbPc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "TTTE")
    analytic_cov[0*nbins:1*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_02, binning_file, lmax)

    # TaTbEcEd
    M_03 = coupling["TaPcTbPd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "TETE")
    M_03 += coupling["TaPdTbPc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "TETE")
    analytic_cov[0*nbins:1*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_03, binning_file, lmax)

    # TaEbEcTd
    M_12 = coupling["TaPcPbTd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "TEET")
    M_12 += coupling["TaTdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "TTEE")
    analytic_cov[1*nbins:2*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_12, binning_file, lmax)

    # TaEbEcEd
    M_13 = coupling["TaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "TEEE")
    M_13 += coupling["TaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "TEEE")
    analytic_cov[1*nbins:2*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_13, binning_file, lmax)

    # EaTbEcEd
    M_23 = coupling["PaPcTbPd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "EETE")
    M_23 += coupling["PaPdTbPc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "EETE")
    analytic_cov[2*nbins:3*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_23, binning_file, lmax)
    
    # TaEbTcTd
    M_10 = coupling["TaTcPbTd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "TTET")
    M_10 += coupling["TaTdPbTc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "TTET")
    analytic_cov[1*nbins:2*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_10, binning_file, lmax)
       
    # EaTbTcTd
    M_20 = coupling["PaTcTbTd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "ETTT")
    M_20 += coupling["PaTdTbTc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "ETTT")
    analytic_cov[2*nbins:3*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_20, binning_file, lmax)
       
    # EaEbTcTd
    M_30 = coupling["PaTcPbTd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "ETET")
    M_30 += coupling["PaTdPbTc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "ETET")
    analytic_cov[3*nbins:4*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_30, binning_file, lmax)
       
    # EaTbTcEd
    M_21 = coupling["PaTcTbPd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "ETTE")
    M_21 += coupling["PaPdTbTc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "EETT")
    analytic_cov[2*nbins:3*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_21, binning_file, lmax)
       
    # EaEbTcEd
    M_31 = coupling["PaTcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "ETEE")
    M_31 += coupling["PaPdPbTc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "EEET")
    analytic_cov[3*nbins:4*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_31, binning_file, lmax)
       
    # EaEbEcTd
    M_32 = coupling["PaPcPbTd"] * so_cov.chi(na, nc, nb, nd, ns, l, ps_all, nl_all, "EEET")
    M_32 += coupling["PaTdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, l, ps_all, nl_all, "ETEE")
    analytic_cov[3*nbins:4*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_32, binning_file, lmax)
    
    mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix="%s/%sx%s" % (mcm_dir, na, nb), spin_pairs=spin_pairs)
    mbb_inv_ab = so_cov.extract_TTTEEE_mbb(mbb_inv_ab)

    mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix="%s/%sx%s" % (mcm_dir, nc, nd), spin_pairs=spin_pairs)
    mbb_inv_cd = so_cov.extract_TTTEEE_mbb(mbb_inv_cd)
    
    #transpose = analytic_cov.copy().T
    #transpose[analytic_cov != 0] = 0
    #analytic_cov += transpose
    
    #analytic_cov = np.triu(analytic_cov) + np.tril(analytic_cov.T, -1)

    analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)

    np.save("%s/analytic_cov_%sx%s_%sx%s.npy" % (cov_dir, na, nb, nc, nd), analytic_cov)

    if d["get_mc_cov"]:
        iStart = d["iStart"]
        iStop = d["iStop"]

        ps_list_ab = []
        ps_list_cd = []
    
        for iii in range(iStart, iStop):
            spec_name_cross_ab = "%s_%sx%s_cross_%05d" % (type, na, nb, iii)
            spec_name_cross_cd = "%s_%sx%s_cross_%05d" % (type, nc, nd, iii)
            
            lb,ps_ab=so_spectra.read_ps(specDir + "/%s.dat" % spec_name_cross_ab, spectra=spectra)
            lb,ps_cd=so_spectra.read_ps(specDir + "/%s.dat" % spec_name_cross_cd, spectra=spectra)
        
            vec_ab = []
            vec_cd = []
            for spec in ["TT", "TE", "ET", "EE"]:
                vec_ab = np.append(vec_ab, ps_ab[spec])
                vec_cd = np.append(vec_cd, ps_cd[spec])
        
            ps_list_ab += [vec_ab]
            ps_list_cd += [vec_cd]
    
        cov_mc = 0
        for iii in range(iStart, iStop):
            cov_mc += np.outer(ps_list_ab[iii], ps_list_cd[iii])

        cov_mc = cov_mc / (iStop-iStart) - np.outer(np.mean(ps_list_ab, axis=0), np.mean(ps_list_cd, axis=0))

        np.save("%s/mc_cov_%sx%s_%sx%s.npy"%(cov_dir, na, nb, nc, nd), cov_mc)






