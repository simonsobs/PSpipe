import numpy as np
import pylab as plt
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils
from pixell import curvedsky, powspec
from pspipe_utils import simulation, pspipe_list, best_fits
import os
import pickle

my_seed = 0
np.random.seed(my_seed)

# we choose to be general, so 2 seasons and 3 arrays
# the map template for the test is a 40 x 40 sq degree CAR patch with 3 arcmin resolution
plot_test = True
rtol = 1e-05 # relative tolerance with respect to the reference data set
atol = 1e-08 # absolute tolerance with respect to the reference data set
surveys = ["sv1", "sv2"]
arrays = {}
arrays["sv1"] = ["pa1", "pa2"]
arrays["sv2"]= ["pa3"]
ra0, ra1, dec0, dec1 = -20, 20, -20, 20
res = 3
lmax = int(500 * (6 / res)) # this is adhoc but should work fine
n_ar = 3
nu_eff = {}
nu_eff["sv1", "pa1"] = 90
nu_eff["sv1", "pa2"] = 150
nu_eff["sv2", "pa3"] = 220
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
rms_uKarcmin = {}
rms_uKarcmin["sv1", "pa1xpa1"] = 20
rms_uKarcmin["sv1", "pa2xpa2"] = 40
rms_uKarcmin["sv1", "pa1xpa2"] = 0
rms_uKarcmin["sv1", "pa2xpa1"] = 0
rms_uKarcmin["sv2", "pa3xpa3"] = 60
beam_fwhm = {}
beam_fwhm[90] = 10
beam_fwhm[150] = 6
beam_fwhm[220] = 4.
sim_alm_dtype = "complex64"
n_splits = {}
n_splits["sv1"] = 2
n_splits["sv2"] = 2
ncomp = 3
spin_pairs = ["spin0xspin0", "spin0xspin2", "spin2xspin0", "spin2xspin2"]
noise_model_dir = "noise_model"
bestfit_dir = "best_fits"
window_dir = "windows"

# The first step is to write a "test" dict file with all relevant arguments

f = open("global_test.dict", "w")
f.write("surveys = %s \n" % surveys)
f.write("deconvolve_pixwin = False \n")
f.write("binning_file = 'test_data/binning_test.dat' \n")
f.write("niter = 0 \n")
f.write("remove_mean = False \n")
f.write("binned_mcm = True \n")
f.write("lmax = %d  \n" % lmax)
f.write("type = 'Dl'  \n")
f.write("write_splits_spectra = True \n")
f.write("multistep_path = './' \n")
f.write("use_toeplitz_mcm  = False \n")
f.write("use_toeplitz_cov  = True \n")
f.write("apply_kspace_filter  = True \n")
f.write("kspace_tf_path  = 'analytical' \n")

f.write("cosmo_params = {'cosmomc_theta':0.0104085, 'logA': 3.044, 'ombh2': 0.02237, 'omch2': 0.1200, 'ns': 0.9649, 'Alens': 1.0, 'tau': 0.0544} \n")
f.write("fg_norm = {'nu_0': 150.0, 'ell_0': 3000, 'T_CMB': 2.725}  \n")
f.write("fg_components = {'tt': ['tSZ_and_CIB', 'cibp', 'kSZ', 'radio', 'dust'], 'te': ['radio', 'dust'], 'ee': ['radio', 'dust'], 'bb': ['radio', 'dust'], 'tb': ['radio', 'dust'], 'eb': []} \n")
f.write("fg_params = {'a_tSZ': 3.30, 'a_kSZ': 1.60, 'a_p': 6.90, 'beta_p': 2.08, 'a_c': 4.90, 'beta_c': 2.20, 'a_s': 3.10, 'a_gtt': 2.79, 'a_gte': 0.36, 'a_gtb': 0, 'a_gee': 0.13, 'a_gbb': 0, 'a_psee': 0.05, 'a_psbb': 0, 'a_pste': 0, 'a_pstb': 0, 'xi': 0.1, 'T_d': 9.60}  \n")

f.write("iStart = 0 \n")
f.write("iStop = 10 \n")
f.write("sim_alm_dtype = '%s' \n" % sim_alm_dtype)

f.write(" \n")

count = 0
for sv in surveys:
    f.write("############# \n")
    f.write("arrays_%s = %s \n"  % (sv, arrays[sv]))
    f.write("k_filter_%s = {'type':'binary_cross','vk_mask':[-50, 50], 'hk_mask':[-50, 50], 'weighted':False} \n" % sv)
    f.write("deconvolve_map_maker_tf_%s = False \n" % sv)
    f.write("src_free_maps_%s = False \n" % sv)
    f.write(" \n")

    for ar in arrays[sv]:
        f.write("####\n")
        f.write("mm_tf_%s_%s  = 'test_data/tf_unity.dat' \n" % (sv, ar))
        f.write("maps_%s_%s  = ['test_data/maps_test_%s_%s_0.fits', 'test_data/maps_test_%s_%s_1.fits'] \n" % (sv, ar, sv, ar, sv, ar))
        f.write("cal_%s_%s  = 1 \n" % (sv, ar))
        f.write("pol_eff_%s_%s  = 1 \n" % (sv, ar))
        f.write("nu_eff_%s_%s  = %d \n" % (sv, ar, nu_eff[sv, ar]))
        f.write("beam_%s_%s  = 'test_data/beam_test_%s_%s.dat' \n" % (sv, ar, sv, ar))
        f.write("window_T_%s_%s =  'windows/window_test_%s_%s.fits' \n" % (sv, ar, sv, ar))
        f.write("window_pol_%s_%s =  'windows/window_test_%s_%s.fits' \n" % (sv, ar, sv, ar))
        f.write(" \n")

        count += 1

f.close()

############
# Now let's generate the fake data set
############

test_dir = "test_data"
pspy_utils.create_directory(test_dir)

############ let's generate the test binning file ############
pspy_utils.create_binning_file(bin_size=100, n_bins=300, file_name=f"{test_dir}/binning_test.dat")

############ let's generate the beams ############

count = 0
for sv in surveys:
    for ar in arrays[sv]:
        l, bl = pspy_utils.beam_from_fwhm(beam_fwhm[nu_eff[sv, ar]], lmax + 200)
        l = np.append(np.array([0, 1]), l)
        bl = np.append(np.array([1, 1]), bl)
        np.savetxt(f"test_data/beam_test_{sv}_{ar}.dat", np.transpose([l, bl]))
        count += 1

############ let's generate the window function ############

pspy_utils.create_directory(window_dir)

binary = so_map.car_template(1, ra0, ra1, dec0, dec1, res)
binary.data[:] = 0
binary.data[1:-1, 1:-1] = 1
res_deg = res / 60
window = so_window.create_apodization(binary, apo_type="C1", apo_radius_degree=10*res_deg)

for sv in surveys:
    for ar in arrays[sv]:
        pts_src_mask = so_map.simulate_source_mask(binary, n_holes=80, hole_radius_arcmin=3*res)
        pts_src_mask = so_window.create_apodization(pts_src_mask, apo_type="C1", apo_radius_degree=10*res_deg)
        pts_src_mask.data *= window.data
        pts_src_mask.write_map(f"{window_dir}/window_test_{sv}_{ar}.fits")
        binary.write_map(f"{window_dir}/binary_{sv}_{ar}.fits")


############ let's simulate some fake noise curve ############

pspy_utils.create_directory(noise_model_dir)

for sv in surveys:
    for id_ar1, ar1 in enumerate(arrays[sv]):
        for id_ar2, ar2 in enumerate(arrays[sv]):

            l, nl = pspy_utils.get_nlth_dict(rms_uKarcmin[sv, f"{ar1}x{ar2}"], "Dl", lmax, spectra=spectra)
            spec_name_noise_mean = f"mean_{ar1}x{ar2}_{sv}_noise"
            so_spectra.write_ps(f"{noise_model_dir}/{spec_name_noise_mean}.dat", l, nl, "Dl", spectra=spectra)


############ let's generate the best fits ############

os.system("python get_best_fit_mflike.py global_test.dict")


############ let's generate some simulations ############


f_name_cmb = bestfit_dir + "/cmb.dat"
f_name_noise = noise_model_dir + "/mean_{}x{}_{}_noise.dat"
f_name_fg = bestfit_dir + "/fg_{}x{}.dat"

ps_mat = simulation.cmb_matrix_from_file(f_name_cmb, lmax, spectra)

freq_list = []
for sv in surveys:
    for ar in arrays[sv]:
        freq_list += [nu_eff[sv, ar]]
# remove doublons
freq_list = list(dict.fromkeys(freq_list))


l, fg_mat = simulation.foreground_matrix_from_files(f_name_fg, freq_list, lmax, spectra)
noise_mat = {}
for sv in surveys:
    l, noise_mat[sv] = simulation.noise_matrix_from_files(f_name_noise,
                                                          sv,
                                                          arrays[sv],
                                                          lmax,
                                                          n_splits[sv],
                                                          spectra)

alms_cmb = curvedsky.rand_alm(ps_mat, lmax=lmax, dtype="complex64")
fglms = simulation.generate_fg_alms(fg_mat, freq_list, lmax)

binary = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
for sv in surveys:
    signal_alms = {}
    for ar in arrays[sv]:
        signal_alms[ar] = alms_cmb + fglms[nu_eff[sv, ar]]
        l, bl = pspy_utils.read_beam_file(f"test_data/beam_test_{sv}_{ar}.dat")
        signal_alms[ar] = curvedsky.almxfl(signal_alms[ar], bl)
    for k in range(n_splits[sv]):
        noise_alms = simulation.generate_noise_alms(noise_mat[sv], arrays[sv], lmax)
        for ar in arrays[sv]:
            split = sph_tools.alm2map(signal_alms[ar] + noise_alms[ar], binary)
            split.write_map(f"{test_dir}/maps_test_{sv}_{ar}_{k}.fits")


# for now you need to manually copy the script in the test directory
os.system("python get_mcm_and_bbl_mpi.py global_test.dict")
os.system("python get_alms.py global_test.dict")
os.system("python get_spectra_from_alms.py global_test.dict")
os.system("python get_best_fit_mflike.py global_test.dict")
os.system("python get_noise_model.py global_test.dict")
os.system("python fast_cov_get_sq_windows_alms.py global_test.dict")
os.system("python fast_cov_get_covariance.py global_test.dict")
os.system("python get_beam_covariance.py global_test.dict")
os.system(f"python mc_get_kspace_tf_spectra.py global_test.dict {my_seed}")
os.system("python mc_tf_analysis.py global_test.dict")


# now we compare the products produced with your scripts to the reference data

spec_name = []
for id_sv1, sv1 in enumerate(surveys):
    for id_ar1, ar1 in enumerate(arrays[sv1]):
        for id_sv2, sv2 in enumerate(surveys):
            for id_ar2, ar2 in enumerate(arrays[sv2]):
                # This ensures that we do not repeat redundant computations
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                
                spec_name += [f"{sv1}_{ar1}x{sv2}_{ar2}" ]



if plot_test == True:
    plot_dir = "test_plot"
    pspy_utils.create_directory("test_plot")

print("")
print("comparison with reference data set")
print("")

ref_data = pickle.load(open("ref_data/trial_data.pkl", "rb"))

ntest = 0
ntest_success = 0

for sid1, spec1 in enumerate(spec_name):

    for spin in spin_pairs:
        mcm = np.load(f"mcms/{spec1}_mode_coupling_inv_{spin}.npy")

        check = np.isclose(mcm, ref_data["mcm", spec1, spin], rtol=rtol, atol=atol, equal_nan=False)
        if check.all():
            print(f"mcm {spin}", spec1, u'\u2713')
            ntest_success += 1
        else:
            print(f"mcm {spin}", spec1, check)
        ntest += 1

    my_l, my_ps = so_spectra.read_ps(f"spectra/Dl_{spec1}_cross.dat", spectra=spectra)
    ps_ref = ref_data["spectra", spec1]
    
    
    for field in spectra:
        check = np.isclose(my_ps[field], ps_ref[field], rtol=rtol, atol=atol, equal_nan=False)
        if check.all():
            print("spectra", spec1, field, u'\u2713')
            ntest_success += 1
        else:
            print("spectra", spec1, field, check)
        ntest += 1

        if plot_test == True:
            plt.figure(figsize=(12,12))
            plt.subplot(2,1,1)
            plt.plot(my_l, my_ps[field], ".", label=f"my ps {spec1}")
            plt.plot(my_l, ps_ref[field], label=f"reference {spec1}")
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(my_l, my_ps[field]/ps_ref[field], label=" my ps/ps ref")
            plt.legend()
            plt.savefig(f"{plot_dir}/{spec1}_{field}.png", bbox_inches="tight")
            plt.clf()
            plt.close()
            

    for sid2, spec2 in enumerate(spec_name):
        if sid1 > sid2: continue

        my_analytic_cov = np.load(f"covariances/analytic_cov_{spec1}_{spec2}.npy")
        analytic_cov_ref = ref_data["analytic_cov", spec1, spec2]
        
        check = np.isclose(my_analytic_cov, analytic_cov_ref, rtol=rtol, atol=atol, equal_nan=False)

        if check.all():
            print("covariances", spec1, spec2, u'\u2713')
            ntest_success += 1
        else:
            print("covariances", spec1, spec2, check)
        ntest += 1

        if plot_test == True:
            plt.figure(figsize=(12,12))
            plt.subplot(2,1,1)
            plt.semilogy()
            plt.plot(my_analytic_cov.diagonal(), ".", label=f"my cov {spec1} {spec2}")
            plt.plot(analytic_cov_ref.diagonal(), label="reference {spec1} {spec2}")
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(my_analytic_cov.diagonal()/analytic_cov_ref.diagonal(), label="my cov/cov ref")
            plt.legend()
            plt.savefig(f"{plot_dir}/analytic_cov_diag_{spec1}_{spec2}.png", bbox_inches="tight")
            plt.clf()
            plt.close()

print("")
print("Summary of the tests")
print("")
print(f"{ntest_success} tests succesful for {ntest} tests total")
