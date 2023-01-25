import numpy as np
import pylab as plt
from pspy import so_spectra
from scipy import interpolate

def interp(l, l_s, fs):
    temp  = interpolate.interp1d(l_s, fs, fill_value="extrapolate")
    f = temp(l)
    return f

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

arrays = ["pa6_f090", "pa5_f090"]
lmax_beam = 6000
nsims  = 200

gamma, gamma_s = {}, {}
for ar in arrays:

    beam = np.loadtxt(f"beams_adri/coadd_{ar}_night_beam_tform_jitter_cmb.txt")
    leakage_beam_TE = np.loadtxt(f"20220501_beams_leakage/coadd_{ar}_night_leakage_tform_1.txt")
    leakage_beam_TB = np.loadtxt(f"20220501_beams_leakage/coadd_{ar}_night_leakage_tform_2.txt")

    l, bl = beam[:,0], beam[:,1]
    l, bl_TE = leakage_beam_TE[:,0], leakage_beam_TE[:,1]
    l, bl_TB = leakage_beam_TB[:,0], leakage_beam_TB[:,1]

    l, bl, bl_TE, bl_TB = l[:lmax_beam], bl[:lmax_beam], bl_TE[:lmax_beam], bl_TB[:lmax_beam]

    gamma[ar, "TE"] = bl_TE / bl
    gamma[ar, "TB"] = bl_TB / bl

    l_s, bl_s, bl_TE_s, bl_TB_s =  np.loadtxt(f"beams_leakage_sigurd/uranus_{ar}_night_uranus_map0300_beam_tform.txt", unpack=True)
    bl_s_interp = interp(l, l_s, bl_s)
    bl_TE_s_interp = interp(l, l_s, bl_TE_s)
    bl_TB_s_interp = interp(l, l_s, bl_TB_s)

    gamma_s[ar, "TE"] = bl_TE_s_interp / bl_s_interp
    gamma_s[ar, "TB"] = bl_TB_s_interp / bl_s_interp

    plt.figure(figsize=(12, 8))

    id = np.where(l_s<lmax_beam)
    plt.plot(l, bl_s_interp, color="blue", label="Sig")
    plt.plot(l_s[id], bl_s[id], "o", color="blue")
    plt.plot(l, bl / bl[0], color="red", label="Adri")
    plt.ylabel(r"$B_{\ell}$", fontsize=20)
    plt.xlabel(r"$\ell$", fontsize=20)
    plt.legend(fontsize=16)
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(l, gamma_s[ar, "TE"], color="blue", label="Sig")
    plt.plot(l_s[id], bl_TE_s[id]/bl_s[id], "o", color="blue")
    plt.plot(l, gamma[ar, "TE"], color="red", label="Adri")
    plt.ylabel(r"$\gamma^{\rm TE}$", fontsize=20)
    plt.xlabel(r"$\ell$", fontsize=20)
    plt.legend(fontsize=16)
    plt.show()
    
    plt.figure(figsize=(12, 8))
    plt.plot(l, gamma_s[ar, "TB"], color="blue", label="Sig")
    plt.plot(l_s[id], bl_TB_s[id]/bl_s[id], "o", color="blue")
    plt.plot(l, gamma[ar, "TB"], color="red", label="Adri")
    plt.ylabel(r"$\gamma^{\rm TB}$", fontsize=20)
    plt.xlabel(r"$\ell$", fontsize=20)
    plt.legend(fontsize=16)
    plt.show()



ar0, ar1 = arrays
lb, Db0 = so_spectra.read_ps(f"spectra/Dl_dr6_{ar0}xdr6_{ar0}_cross.dat", spectra=spectra)
lb, Db1 = so_spectra.read_ps(f"spectra/Dl_dr6_{ar1}xdr6_{ar1}_cross.dat", spectra=spectra)

data = np.loadtxt("cosmo2017_10K_acc3_lensedCls.dat")
l, cl_TT = data[:,0], data[:,1]
l, cl_TT = l[:lmax_beam], cl_TT[:lmax_beam]


for field in ["TE", "TB"]:

    null =  Db1[field] - Db0[field]
    null_list = []
    for iii in range(nsims):
        lb, Db0_sim = so_spectra.read_ps(f"sim_spectra/Dl_dr6_{ar0}xdr6_{ar0}_cross_%05d.dat" % iii, spectra=spectra)
        lb, Db1_sim = so_spectra.read_ps(f"sim_spectra/Dl_dr6_{ar1}xdr6_{ar1}_cross_%05d.dat" % iii, spectra=spectra)
        null_list += [Db1_sim[field] - Db0_sim[field]]

    mean = np.mean(null_list, axis=0)
    std = np.std(null_list, axis=0)

    plt.figure(figsize=(16, 8))
    plt.errorbar(lb, null * lb, std * lb, fmt=".", label=f"null {ar1} - {ar0}")
    plt.plot(l, l * (gamma[ar1, field] - gamma[ar0, field]) * cl_TT, label="leakage std map maker (Adri)")
    plt.plot(l, l * (gamma_s[ar1, field] - gamma_s[ar0, field]) * cl_TT, label="leakage max like map maker (Sigurd)")
    plt.legend(fontsize=18)
    plt.savefig(f"leakage_{ar1}-{ar0}_{field}.png", bbox_inches="tight")
    plt.clf()
    plt.close()
