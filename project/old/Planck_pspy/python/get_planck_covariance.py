from pspy import pspy_utils, so_dict, so_map, so_mpi, so_mcm, so_spectra, so_cov
import numpy as np
import healpy as hp
import sys

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

windows_dir = "windows"
mcms_dir = "mcms"
spectra_dir = "spectra"
ps_model_dir = "noise_model"
cov_dir = "covariances"
bestfit_dir = "best_fits"


pspy_utils.create_directory(cov_dir)

type = d["type"]
freqs = d["freqs"]
binning_file = d["binning_file"]
lmax = d["lmax"]
niter = d["niter"]

exp = "Planck"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

bin_lo, bin_hi, bin_c, bin_size = pspy_utils.read_binning_file(binning_file, lmax)
nbins = len(bin_hi)

ps_all = {}
nl_all = {}
bl1,bl2 = {}, {}
spec_name = []

ns = {"Planck": 2}

pixwin = hp.pixwin(2048)[:lmax]

for c1,freq1 in enumerate(freqs):
    for c2,freq2 in enumerate(freqs):
        if c1>c2: continue
        
        
        l, bl1_hm1_T = np.loadtxt(d["beam_%s_hm1_T" % freq1], unpack=True)
        l, bl1_hm2_T = np.loadtxt(d["beam_%s_hm2_T" % freq1], unpack=True)
        l, bl1_hm1_pol = np.loadtxt(d["beam_%s_hm1_pol" % freq1], unpack=True)
        l, bl1_hm2_pol = np.loadtxt(d["beam_%s_hm2_pol" % freq1], unpack=True)


        l, bl2_hm1_T = np.loadtxt(d["beam_%s_hm1_T" % freq2], unpack=True)
        l, bl2_hm2_T = np.loadtxt(d["beam_%s_hm2_T" % freq2], unpack=True)
        l, bl2_hm1_pol = np.loadtxt(d["beam_%s_hm1_pol" % freq2], unpack=True)
        l, bl2_hm2_pol = np.loadtxt(d["beam_%s_hm2_pol" % freq2], unpack=True)


        bl1_hm1_T, bl1_hm2_T, bl2_hm1_T, bl2_hm2_T = bl1_hm1_T[2: lmax + 2], bl1_hm2_T[2: lmax + 2], bl2_hm1_T[2: lmax + 2], bl2_hm2_T[2: lmax + 2]
        bl1_hm1_pol, bl1_hm2_pol, bl2_hm1_pol, bl2_hm2_pol = bl1_hm1_pol[2: lmax + 2], bl1_hm2_pol[2: lmax + 2], bl2_hm1_pol[2: lmax + 2], bl2_hm2_pol[2: lmax + 2]

        bl1["TT"] = np.sqrt(bl1_hm1_T * bl1_hm2_T)
        bl2["TT"] = np.sqrt(bl2_hm1_T * bl2_hm2_T)

        bl1["EE"] = np.sqrt(bl1_hm1_pol * bl1_hm2_pol)
        bl2["EE"] = np.sqrt(bl2_hm1_pol * bl2_hm2_pol)

        bl1["TE"] = np.sqrt(bl1["EE"] * bl1["TT"])
        bl2["TE"] = np.sqrt(bl2["EE"] * bl2["TT"])
        
        bl1["ET"] = bl1["TE"]
        bl2["ET"] = bl2["TE"]

        if d["use_noise_from_sim"]:
            if d["use_ffp10"] == True:
                spec_name_noise = "mean_simffp10_%s_%sx%s_%s_noise" % (exp, freq1, exp, freq2)
            else:
                spec_name_noise = "mean_sim_%s_%sx%s_%s_noise" % (exp, freq1, exp, freq2)
        else:
            spec_name_noise = "mean_%s_%sx%s_%s_noise" % (exp, freq1, exp, freq2)

        
        l, Nl = so_spectra.read_ps(ps_model_dir + "/%s.dat" % spec_name_noise, spectra=spectra)
                
        for spec in ["TT", "TE", "ET", "EE"]:
            
            if spec == "ET":
                lth, ps_th = np.loadtxt("%s/best_fit_%sx%s_%s.dat"%(bestfit_dir, freq1, freq2, "TE"), unpack=True)
            else:
                lth, ps_th = np.loadtxt("%s/best_fit_%sx%s_%s.dat"%(bestfit_dir, freq1, freq2, spec), unpack=True)

            ps_th = ps_th[2: lmax + 2]
            
            
            ps_all["%s_%s" % (exp, freq1), "%s_%s" % (exp, freq2), spec] = bl1[spec] * bl2[spec] * pixwin**2 * ps_th
                    
            if freq1 == freq2:
                nl_all["%s_%s" % (exp, freq1), "%s_%s" % (exp, freq2), spec] = Nl[spec] * ns[exp] * pixwin**2
            else:
                nl_all["%s_%s" % (exp, freq1), "%s_%s" % (exp, freq2), spec] = np.zeros(lmax)
                    

            ps_all["%s_%s" % (exp, freq2), "%s_%s"%(exp, freq1), spec] = ps_all["%s_%s"%(exp, freq1), "%s_%s"%(exp, freq2), spec]
            nl_all["%s_%s" % (exp, freq2), "%s_%s"%(exp, freq1), spec] = nl_all["%s_%s"%(exp, freq1), "%s_%s"%(exp, freq2), spec]
                
        spec_name += ["%s_%sx%s_%s" % (exp, freq1, exp, freq2)]


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
    win["Ta"] = so_map.read_map("%s/window_T_%s-hm1.fits"%(windows_dir, na))
    win["Tb"] = so_map.read_map("%s/window_T_%s-hm2.fits"%(windows_dir, nb))
    win["Tc"] = so_map.read_map("%s/window_T_%s-hm1.fits"%(windows_dir, nc))
    win["Td"] = so_map.read_map("%s/window_T_%s-hm2.fits"%(windows_dir, nd))
    win["Pa"] = so_map.read_map("%s/window_P_%s-hm1.fits"%(windows_dir, na))
    win["Pb"] = so_map.read_map("%s/window_P_%s-hm2.fits"%(windows_dir, nb))
    win["Pc"] = so_map.read_map("%s/window_P_%s-hm1.fits"%(windows_dir, nc))
    win["Pd"] = so_map.read_map("%s/window_P_%s-hm2.fits"%(windows_dir, nd))

    coupling = so_cov.cov_coupling_spin0and2_simple(win, lmax, niter=niter, planck=True)
    analytic_cov = np.zeros((4*nbins, 4*nbins))

    # TaTbTcTd
    M_00 = coupling["TaTcTbTd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "TTTT")
    M_00 += coupling["TaTdTbTc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "TTTT")
    analytic_cov[0*nbins:1*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_00, binning_file, lmax)
    
    # TaEbTcEd
    M_11 = coupling["TaTcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "TTEE")
    M_11 += coupling["TaPdPbTc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "TEET")
    analytic_cov[1*nbins:2*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_11, binning_file, lmax)
    
    # EaTbEcTd
    M_22 = coupling["PaPcTbTd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EETT")
    M_22 += coupling["PaTdTbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "ETTE")
    analytic_cov[2*nbins:3*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_22, binning_file, lmax)
    
    # EaEbEcEd
    M_33 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EEEE")
    M_33 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EEEE")
    analytic_cov[3*nbins:4*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_33, binning_file, lmax)
    
    # TaTbTcEd
    M_01 = coupling["TaTcTbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "TTTE")
    M_01 += coupling["TaPdTbTc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "TETT")
    analytic_cov[0*nbins:1*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_01, binning_file, lmax)
    
    # TaTbEcTd
    M_02 = coupling["TaPcTbTd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "TETT")
    M_02 += coupling["TaTdTbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "TTTE")
    analytic_cov[0*nbins:1*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_02, binning_file, lmax)
    
    # TaTbEcEd
    M_03 = coupling["TaPcTbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "TETE")
    M_03 += coupling["TaPdTbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "TETE")
    analytic_cov[0*nbins:1*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_03, binning_file, lmax)
    
    # TaEbEcTd
    M_12 = coupling["TaPcPbTd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "TEET")
    M_12 += coupling["TaTdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "TTEE")
    analytic_cov[1*nbins:2*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_12, binning_file, lmax)
    
    # TaEbEcEd
    M_13 = coupling["TaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "TEEE")
    M_13 += coupling["TaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "TEEE")
    analytic_cov[1*nbins:2*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_13, binning_file, lmax)
    
    # EaTbEcEd
    M_23 = coupling["PaPcTbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EETE")
    M_23 += coupling["PaPdTbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EETE")
    analytic_cov[2*nbins:3*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_23, binning_file, lmax)
    
    # TaEbTcTd
    M_10 = coupling["TaTcPbTd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "TTET")
    M_10 += coupling["TaTdPbTc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "TTET")
    analytic_cov[1*nbins:2*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_10, binning_file, lmax)
    
    # EaTbTcTd
    M_20 = coupling["PaTcTbTd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "ETTT")
    M_20 += coupling["PaTdTbTc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "ETTT")
    analytic_cov[2*nbins:3*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_20, binning_file, lmax)
    
    # EaEbTcTd
    M_30 = coupling["PaTcPbTd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "ETET")
    M_30 += coupling["PaTdPbTc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "ETET")
    analytic_cov[3*nbins:4*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_30, binning_file, lmax)
    
    # EaTbTcEd
    M_21 = coupling["PaTcTbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "ETTE")
    M_21 += coupling["PaPdTbTc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EETT")
    analytic_cov[2*nbins:3*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_21, binning_file, lmax)
    
    # EaEbTcEd
    M_31 = coupling["PaTcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "ETEE")
    M_31 += coupling["PaPdPbTc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EEET")
    analytic_cov[3*nbins:4*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_31, binning_file, lmax)
    
    # EaEbEcTd
    M_32 = coupling["PaPcPbTd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EEET")
    M_32 += coupling["PaTdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "ETEE")
    analytic_cov[3*nbins:4*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_32, binning_file, lmax)
    
    mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix="%s/%sx%s-hm1xhm2" % (mcms_dir, na, nb), spin_pairs=spin_pairs)
    mbb_inv_ab = so_cov.extract_TTTEEE_mbb(mbb_inv_ab)
    
    mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix="%s/%sx%s-hm1xhm2" % (mcms_dir, nc, nd), spin_pairs=spin_pairs)
    mbb_inv_cd = so_cov.extract_TTTEEE_mbb(mbb_inv_cd)

    analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)
    
    np.save("%s/analytic_cov_%sx%s_%sx%s.npy" % (cov_dir, na, nb, nc, nd), analytic_cov)



