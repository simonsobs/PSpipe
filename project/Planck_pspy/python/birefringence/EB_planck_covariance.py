"""
This script is used to compute all EE-EB-BE-BB covariance elements
"""


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
        
        
        l, bl1_hm1_pol = np.loadtxt(d["beam_%s_hm1_pol" % freq1], unpack=True)
        l, bl1_hm2_pol = np.loadtxt(d["beam_%s_hm2_pol" % freq1], unpack=True)
        l, bl2_hm1_pol = np.loadtxt(d["beam_%s_hm1_pol" % freq2], unpack=True)
        l, bl2_hm2_pol = np.loadtxt(d["beam_%s_hm2_pol" % freq2], unpack=True)


        bl1_hm1_pol, bl1_hm2_pol = bl1_hm1_pol[2: lmax + 2], bl1_hm2_pol[2: lmax + 2]
        bl2_hm1_pol, bl2_hm2_pol = bl2_hm1_pol[2: lmax + 2], bl2_hm2_pol[2: lmax + 2]

        for spec in ["EE", "EB", "BE", "BB"]:
            bl1[spec] = np.sqrt(bl1_hm1_pol * bl1_hm2_pol)
            bl2[spec] = np.sqrt(bl2_hm1_pol * bl2_hm2_pol)

        if d["use_noise_from_sim"]:
            if d["use_ffp10"] == True:
                spec_name_noise = "mean_simffp10_%s_%sx%s_%s_noise" % (exp, freq1, exp, freq2)
            else:
                spec_name_noise = "mean_sim_%s_%sx%s_%s_noise" % (exp, freq1, exp, freq2)
        else:
            spec_name_noise = "mean_%s_%sx%s_%s_noise" % (exp, freq1, exp, freq2)

        
        l, Nl = so_spectra.read_ps(ps_model_dir + "/%s.dat" % spec_name_noise, spectra=spectra)
                
        for spec in ["EE", "EB", "BE", "BB"]:
            
            if spec == "BE":
                lth, ps_th = np.loadtxt("%s/best_fit_%sx%s_%s.dat"%(bestfit_dir, freq1, freq2, "EB"), unpack=True)
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

    # EaEbEcEd
    M_00 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EEEE")
    M_00 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EEEE")
    analytic_cov[0*nbins:1*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_00, binning_file, lmax)
    
    # EaBbEcBd
    M_11 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EEBB")
    M_11 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EBBE")
    analytic_cov[1*nbins:2*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_11, binning_file, lmax)
    
    # BaEbBcEd
    M_22 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "BBEE")
    M_22 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "BEEB")
    analytic_cov[2*nbins:3*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_22, binning_file, lmax)
    
    # BaBbBcBd
    M_33 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "BBBB")
    M_33 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "BBBB")
    analytic_cov[3*nbins:4*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_33, binning_file, lmax)
    
    # EaEbEcBd
    M_01 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EEEB")
    M_01 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EBEE")
    analytic_cov[0*nbins:1*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_01, binning_file, lmax)
    
    # EaEbBcEd
    M_02 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EBEE")
    M_02 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EEEB")
    analytic_cov[0*nbins:1*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_02, binning_file, lmax)
    
    # EaEbBcBd
    M_03 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EBEB")
    M_03 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EBEB")
    analytic_cov[0*nbins:1*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_03, binning_file, lmax)
    
    # EaBbBcEd
    M_12 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EBBE")
    M_12 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EEBB")
    analytic_cov[1*nbins:2*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_12, binning_file, lmax)
    
    # EaBbBcBd
    M_13 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EBBB")
    M_13 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EBBB")
    analytic_cov[1*nbins:2*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_13, binning_file, lmax)
    
    # BaEbBcBd
    M_23 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "BBEB")
    M_23 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "BBEB")
    analytic_cov[2*nbins:3*nbins, 3*nbins:4*nbins] = so_cov.bin_mat(M_23, binning_file, lmax)
    
    # EaBbEcEd
    M_10 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "EEBE")
    M_10 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "EEBE")
    analytic_cov[1*nbins:2*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_10, binning_file, lmax)
    
    # BaEbEcEd
    M_20 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "BEEE")
    M_20 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "BEEE")
    analytic_cov[2*nbins:3*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_20, binning_file, lmax)
    
    # BaBbEcEd
    M_30 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "BEBE")
    M_30 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "BEBE")
    analytic_cov[3*nbins:4*nbins, 0*nbins:1*nbins] = so_cov.bin_mat(M_30, binning_file, lmax)
    
    # BaEbEcBd
    M_21 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "BEEB")
    M_21 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "BBEE")
    analytic_cov[2*nbins:3*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_21, binning_file, lmax)
    
    # BaBbEcBd
    M_31 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "BEBB")
    M_31 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "BBBE")
    analytic_cov[3*nbins:4*nbins, 1*nbins:2*nbins] = so_cov.bin_mat(M_31, binning_file, lmax)
    
    # BaBbBcEd
    M_32 = coupling["PaPcPbPd"] * so_cov.chi(na, nc, nb, nd, ns, ps_all, nl_all, "BBBE")
    M_32 += coupling["PaPdPbPc"] * so_cov.chi(na, nd, nb, nc, ns, ps_all, nl_all, "BEBB")
    analytic_cov[3*nbins:4*nbins, 2*nbins:3*nbins] = so_cov.bin_mat(M_32, binning_file, lmax)
    
    mbb_inv_ab, Bbl_ab = so_mcm.read_coupling(prefix="%s/%sx%s-hm1xhm2" % (mcms_dir, na, nb), spin_pairs=spin_pairs)
    mbb_inv_ab = so_cov.extract_EEEBBB_mbb(mbb_inv_ab)
    
    mbb_inv_cd, Bbl_cd = so_mcm.read_coupling(prefix="%s/%sx%s-hm1xhm2" % (mcms_dir, nc, nd), spin_pairs=spin_pairs)
    mbb_inv_cd = so_cov.extract_EEEBBB_mbb(mbb_inv_cd)

    analytic_cov = np.dot(np.dot(mbb_inv_ab, analytic_cov), mbb_inv_cd.T)
    
    np.save("%s/analytic_cov_%sx%s_%sx%s_EB.npy" % (cov_dir, na, nb, nc, nd), analytic_cov)



