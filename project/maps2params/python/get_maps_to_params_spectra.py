"""
This script generates the simulations and compute their power spectra.
The simulations are generated on the fly and the spectra are written to disk.
"""
from pspy import pspy_utils, so_dict, so_map, so_mpi, sph_tools, so_mcm, so_spectra
import numpy as np
import sys
from pixell import curvedsky, powspec
import maps_to_params_utils
import time

d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

experiments = d["experiments"]
lmax = d["lmax"]
niter = d["niter"]
type = d["type"]
binning_file = d["binning_file"]
lcut = d["lcut"]
write_all_spectra = d["write_splits_spectra"]
include_fg = d["include_fg"]
fg_dir = d["fg_dir"]
nuis_params = d["nuisance_params"]
want_seed = d["want_seed"]

fg_components = d["fg_components"]
fg_components["tt"].remove("tSZ_and_CIB")
for comp in ["tSZ", "cibc", "tSZxCIB"]:
    fg_components["tt"].append(comp)

map_dir = d["mbs_dir"]
use_mbs = d["use_mbs"]

freq2chan = {27: "LF1",
             39: "LF2",
             93: "MFF1",
             145: "MFF2",
             225: "UHF1",
             280: "UHF2"}

window_dir = "windows"
mcm_dir = "mcms"
noise_data_dir = "sim_data/noise_ps"
specDir = "spectra"      

lmax_simu = lmax

pspy_utils.create_directory(specDir)

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]

all_freqs = [freq for exp in experiments for freq in d["freqs_%s" % exp]]
ncomp = 3
ps_cmb = powspec.read_spectrum(d["clfile"])[:ncomp, :ncomp]

if include_fg == True:
    l, ps_fg = maps_to_params_utils.get_foreground_matrix(fg_dir, fg_components, all_freqs, lmax_simu+1)

so_mpi.init(True)
subtasks = so_mpi.taskrange(imin=d["iStart"], imax=d["iStop"])

for iii in subtasks:
    #First we will generate our simulations and take their harmonics transforms
    if want_seed == True:
        print(iii)
        np.random.seed(iii.astype(int))
    t0 = time.time()

    if use_mbs:
        alms = np.zeros((3, (lmax_simu + 1) * (lmax_simu + 2) // 2), dtype = "complex128")
    else:
        alms = curvedsky.rand_alm(ps_cmb, lmax=lmax_simu)

    if include_fg == True:
        fglms = curvedsky.rand_alm(ps_fg, lmax=lmax_simu)

    master_alms = {}
    fcount = 0

    for exp in experiments:
        freqs = d["freqs_%s" % exp]
        nsplits = d["nsplits_%s" % exp]

        if d["pixel_%s" % exp] == "CAR":
            template = so_map.car_template(ncomp,
                                           d["ra0_%s" % exp],
                                           d["ra1_%s" % exp],
                                           d["dec0_%s" % exp],
                                           d["dec1_%s" % exp],
                                           d["res_%s" % exp])
        else:
            template = so_map.healpix_template(ncomp, nside=d["nside_%s" % exp])

        l, nl_array_t, nl_array_pol = maps_to_params_utils.get_noise_matrix_spin0and2(noise_data_dir,
                                                                                      exp,
                                                                                      freqs,
                                                                                      lmax_simu+1,
                                                                                      nsplits,
                                                                                      lcut=lcut)

        nlms = maps_to_params_utils.generate_noise_alms(nl_array_t,
                                                        lmax_simu,
                                                        nsplits,
                                                        ncomp,
                                                        nl_array_pol=nl_array_pol)

        for fid ,freq in enumerate(freqs):
            window = so_map.read_map("%s/window_%s_%s.fits" % (window_dir, exp, freq))
            window_tuple = (window, window)
            l, bl = np.loadtxt("sim_data/beams/beam_%s_%s.dat" %(exp, freq), unpack=True)

            alms_beamed = alms.copy()

            if include_fg == True:
                # include fg for the temperature alms
                # pol not implemented yet
                alms_beamed[0] += fglms[2*fcount]
                alms_beamed[1] += fglms[2*fcount+1]

            alms_beamed = maps_to_params_utils.multiply_alms(alms_beamed, bl, ncomp)
            
            #here we calibrate the T, E alms and rotate the E, B alms of each frequency, for the LAT experiment
            if exp == "LAT":
                cal_alms = maps_to_params_utils.calibrate_alm(alms_beamed, freq, **nuis_params)
                print(f"Calibrating the {exp} alm by calT_{freq} = {nuis_params[f'calT_{freq}']} and calE_{freq} = {nuis_params[f'calE_{freq}']}")

                sys_alms = maps_to_params_utils.rotate_alm_polang(cal_alms, freq, **nuis_params)
                print(f"Rotating the {exp} alm by alpha_{freq} = {nuis_params[f'alpha_{freq}']} deg")

            else: 
                sys_alms = alms_beamed

            fcount += 1

            if use_mbs:
                map = so_map.read_map(
                    "%s/%04d/simonsobs_cmb_uKCMB_la%s_nside4096_%04d.fits" % (
                    map_dir, iii, freq2chan[freq], iii))

            for k in range(nsplits):
                noisy_alms = sys_alms.copy()

                noisy_alms[0] +=  nlms["T",k][fid]
                noisy_alms[1] +=  nlms["E",k][fid]
                noisy_alms[2] +=  nlms["B",k][fid]

                split = sph_tools.alm2map(noisy_alms, template)

                if use_mbs:
                    split.data += map.data

                # Now that we have generated a split of data of experiment exp
                # and frequency freq, we take its harmonic transform
                split = maps_to_params_utils.remove_mean(split, window_tuple, ncomp)
                master_alms[exp, freq, k] = sph_tools.get_alms(split, window_tuple, niter, lmax)

    # We now form auto and cross power spectra from the alms
    ps_dict = {}

    for id_exp1, exp1 in enumerate(experiments):
        freqs1 = d["freqs_%s" % exp1]
        nsplits1 = d["nsplits_%s" % exp1]

        for id_f1, f1 in enumerate(freqs1):

            for id_exp2, exp2 in enumerate(experiments):
                freqs2 = d["freqs_%s" % exp2]
                nsplits2 = d["nsplits_%s" % exp2]

                for id_f2, f2 in enumerate(freqs2):

                    if  (id_exp1 == id_exp2) & (id_f1 > id_f2) : continue
                    if  (id_exp1 > id_exp2) : continue

                    for spec in spectra:
                        ps_dict[spec, "auto"] = []
                        ps_dict[spec, "cross"] = []

                    for s1 in range(nsplits1):
                        for s2 in range(nsplits2):
                            mbb_inv, Bbl = so_mcm.read_coupling(prefix="%s/%s_%sx%s_%s" % (mcm_dir, exp1, f1, exp2, f2),
                                                                spin_pairs=spin_pairs)

                            l, ps_master = so_spectra.get_spectra(master_alms[exp1, f1, s1],
                                                                  master_alms[exp2, f2, s2],
                                                                  spectra=spectra)

                            spec_name="%s_%s_%s_%dx%s_%s_%d_%05d" % (type, exp1, f1, s1, exp2, f2, s2, iii)

                            lb, ps = so_spectra.bin_spectra(l,
                                                            ps_master,
                                                            binning_file,
                                                            lmax,
                                                            type=type,
                                                            mbb_inv=mbb_inv,
                                                            spectra=spectra)

                            if write_all_spectra:
                                so_spectra.write_ps(specDir + "/%s.dat" % spec_name, lb, ps, type, spectra=spectra)

                            for count, spec in enumerate(spectra):
                                if (s1 == s2) & (exp1 == exp2):
                                    if count == 0:
                                        print("auto %s_%s split%d X %s_%s split%d" % (exp1, f1, s1, exp2, f2, s2))
                                    ps_dict[spec, "auto"] += [ps[spec]]
                                else:
                                    if count == 0:
                                        print("cross %s_%s split%d X %s_%s split%d" % (exp1, f1, s1, exp2, f2, s2))
                                    ps_dict[spec, "cross"] += [ps[spec]]

                    ps_dict_auto_mean = {}
                    ps_dict_cross_mean = {}
                    ps_dict_noise_mean = {}

                    for spec in spectra:
                        ps_dict_cross_mean[spec] = np.mean(ps_dict[spec, "cross"], axis=0)
                        spec_name_cross = "%s_%s_%sx%s_%s_cross_%05d" % (type, exp1, f1, exp2, f2, iii)

                        if exp1 == exp2:
                            ps_dict_auto_mean[spec] = np.mean(ps_dict[spec, "auto"], axis=0)
                            spec_name_auto = "%s_%s_%sx%s_%s_auto_%05d" % (type, exp1, f1, exp2, f2, iii)
                            ps_dict_noise_mean[spec] = (ps_dict_auto_mean[spec] - ps_dict_cross_mean[spec]) / d["nsplits_%s" % exp]
                            spec_name_noise = "%s_%s_%sx%s_%s_noise_%05d" % (type, exp1, f1, exp2, f2, iii)

                    so_spectra.write_ps(specDir + "/%s.dat" % spec_name_cross, lb, ps_dict_cross_mean, type, spectra=spectra)

                    if exp1 == exp2:
                        so_spectra.write_ps(specDir+"/%s.dat" % spec_name_auto, lb, ps_dict_auto_mean, type, spectra=spectra)
                        so_spectra.write_ps(specDir+"/%s.dat" % spec_name_noise, lb, ps_dict_noise_mean, type, spectra=spectra)


    print("sim number %05d done in %.02f s" % (iii, time.time()-t0))
