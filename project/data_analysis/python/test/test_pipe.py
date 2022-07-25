import numpy as np
import pylab as plt
from pspy import so_map, so_window, so_mcm, sph_tools, so_spectra, pspy_utils
from pixell import curvedsky, powspec
import data_analysis_utils
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
nu_eff = [90, 150, 220]
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
rms_uKarcmin = {}
rms_uKarcmin["sv1", "pa1xpa1"] = 20
rms_uKarcmin["sv1", "pa2xpa2"] = 40
rms_uKarcmin["sv1", "pa1xpa2"] = 0
rms_uKarcmin["sv2", "pa3xpa3"] = 60
beam_fwhm = {}
beam_fwhm[90] = 10
beam_fwhm[150] = 6
beam_fwhm[220] = 4.
sim_alm_dtype = "complex64"
nsplits = 2
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
f.write("use_toeplitz_mcm  = False \n")
f.write("use_toeplitz_cov  = True \n")

f.write("cosmo_params = {'cosmomc_theta':0.0104085, 'logA': 3.044, 'ombh2': 0.02237, 'omch2': 0.1200, 'ns': 0.9649, 'Alens': 1.0, 'tau': 0.0544} \n")
f.write("fg_norm = {'nu_0': 150.0, 'ell_0': 3000, 'T_CMB': 2.725}  \n")
f.write("fg_components = {'tt': ['tSZ_and_CIB', 'cibp', 'kSZ', 'radio', 'dust'], 'te': ['radio', 'dust'], 'ee': ['radio', 'dust']} \n")
f.write("fg_params = {'a_tSZ': 3.30, 'a_kSZ': 1.60, 'a_p': 6.90, 'beta_p': 2.08, 'a_c': 4.90, 'beta_c': 2.20, 'a_s': 3.10, 'a_gtt': 2.79, 'a_gte': 0.36, 'a_gee': 0.13, 'a_psee': 0.05, 'a_pste': 0, 'xi': 0.1, 'T_d': 9.60}  \n")

f.write("iStart = 0 \n")
f.write("iStop = 2 \n")
f.write("sim_alm_dtype = '%s' \n" % sim_alm_dtype)

f.write(" \n")

count = 0
for sv in surveys:
    f.write("############# \n")
    f.write("arrays_%s = %s \n"  % (sv, arrays[sv]))
    f.write("k_filter_%s = {'apply':True, 'type':'binary_cross','vk_mask':[-50, 50], 'hk_mask':[-50, 50], 'weighted':False, 'tf': 'analytic'} \n" % sv)
    f.write("deconvolve_map_maker_tf_%s = False \n" % sv)
    f.write("src_free_maps_%s = False \n" % sv)
    f.write(" \n")

    for ar in arrays[sv]:
        f.write("####\n")
        f.write("mm_tf_%s_%s  = 'test_data/tf_unity.dat' \n" % (sv, ar))
        f.write("maps_%s_%s  = ['test_data/maps_test_%s_%s_0.fits', 'test_data/maps_test_%s_%s_1.fits'] \n" % (sv, ar, sv, ar, sv, ar))
        f.write("cal_%s_%s  = 1 \n" % (sv, ar))
        f.write("pol_eff_%s_%s  = 1 \n" % (sv, ar))
        f.write("nu_eff_%s_%s  = %d \n" % (sv, ar, nu_eff[count]))
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
pspy_utils.create_binning_file(bin_size=100, n_bins=300, file_name="%s/binning_test.dat" % test_dir)

############ let's generate the beams ############

count = 0
for sv in surveys:
    for ar in arrays[sv]:
        l, bl = pspy_utils.beam_from_fwhm(beam_fwhm[nu_eff[count]], lmax + 200)
        np.savetxt("test_data/beam_test_%s_%s.dat" % (sv, ar), np.transpose([l, bl]))
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
        pts_src_mask.write_map("%s/window_test_%s_%s.fits" % (window_dir, sv, ar))
        binary.write_map("%s/binary_%s_%s.fits" % (window_dir, sv, ar))


############ let's simulate some fake noise curve ############

pspy_utils.create_directory(noise_model_dir)

for sv in surveys:
    for id_ar1, ar1 in enumerate(arrays[sv]):
        for id_ar2, ar2 in enumerate(arrays[sv]):
            if id_ar1 > id_ar2: continue

            l, nl = pspy_utils.get_nlth_dict(rms_uKarcmin[sv, "%sx%s" % (ar1, ar2)], "Dl", lmax, spectra=spectra)
            spec_name_noise_mean = "mean_%sx%s_%s_noise" % (ar1, ar2, sv)
            so_spectra.write_ps("%s/%s.dat" % (noise_model_dir, spec_name_noise_mean), l, nl, "Dl", spectra=spectra)


############ let's generate the best fits ############
os.system("python get_best_fit_mflike.py global_test.dict")


############ let's generate some simulations ############

ps_cmb = powspec.read_spectrum("%s/lcdm.dat" % bestfit_dir)[:ncomp, :ncomp]
l, ps_fg = data_analysis_utils.get_foreground_matrix(bestfit_dir, nu_eff, lmax)
alms = curvedsky.rand_alm(ps_cmb, lmax=lmax, dtype=sim_alm_dtype)
fglms = curvedsky.rand_alm(ps_fg, lmax=lmax, dtype=sim_alm_dtype)

binary = so_map.car_template(ncomp, ra0, ra1, dec0, dec1, res)
count = 0
for sv in surveys:
    array_list = arrays[sv]

    l, nl_array_t, nl_array_pol = data_analysis_utils.get_noise_matrix_spin0and2(noise_model_dir,
                                                                                 sv,
                                                                                 array_list,
                                                                                 lmax,
                                                                                 nsplits=nsplits)

    nlms = data_analysis_utils.generate_noise_alms(nl_array_t,
                                                   lmax,
                                                   n_splits=nsplits,
                                                   ncomp=ncomp,
                                                   nl_array_pol=nl_array_pol,
                                                   dtype=sim_alm_dtype)
    
    for ar_id, ar in enumerate(array_list):
        alms_beamed = alms.copy()
        alms_beamed[0] += fglms[count]
        l, bl = pspy_utils.read_beam_file("test_data/beam_test_%s_%s.dat" % (sv, ar))
        alms_beamed = curvedsky.almxfl(alms_beamed, bl)

        for k in range(nsplits):
            noisy_alms = alms_beamed.copy()
            noisy_alms[0] +=  nlms["T", k][ar_id]
            noisy_alms[1] +=  nlms["E", k][ar_id]
            noisy_alms[2] +=  nlms["B", k][ar_id]
            
            split = sph_tools.alm2map(noisy_alms, binary)
            split.write_map("%s/maps_test_%s_%s_%d.fits" % (test_dir, sv, ar, k))

        count += 1

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
                if  (id_sv1 == id_sv2) & (id_ar1 > id_ar2) : continue
                if  (id_sv1 > id_sv2) : continue
                spec_name += ["%s_%sx%s_%s" % (sv1, ar1, sv2, ar2)]

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
        mcm = np.load("mcms/%s_mode_coupling_inv_%s.npy" % (spec1, spin))

        check = np.isclose(mcm, ref_data["mcm", spec1, spin], rtol=rtol, atol=atol, equal_nan=False)
        if check.all():
            print("mcm %s" % spin, spec1, u'\u2713')
            ntest_success += 1
        else:
            print("mcm %s" % spin, spec1, check)
        ntest += 1

    my_l, my_ps = so_spectra.read_ps("spectra/Dl_%s_cross.dat" % spec1, spectra=spectra)
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
            plt.plot(my_l, my_ps[field], ".", label=" my ps %s" % spec1)
            plt.plot(my_l, ps_ref[field], label="reference %s" % spec1)
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(my_l, my_ps[field]/ps_ref[field], label=" my ps/ps ref")
            plt.legend()
            plt.savefig("%s/%s_%s.png" % (plot_dir, spec1, field), bbox_inches="tight")
            plt.clf()
            plt.close()
            
            
    for my_key1 in ["filter", "nofilter"]:
        for my_key2 in  ["standard", "noE", "noB"]:
            _, my_ps = so_spectra.read_ps(f"sim_spectra_for_tf/Dl_{spec1}_{my_key1}_{my_key2}_00000.dat", spectra=spectra)
            ps_ref = ref_data[f"sim_spectra_{my_key1}_{my_key2}", spec1]

            for field in spectra:
            
                check = np.isclose(my_ps[field], ps_ref[field], rtol=rtol, atol=atol, equal_nan=False)
                if check.all():
                    print(f"sim spectra {my_key1} {my_key2} {spec1} {field}", u'\u2713')
                    ntest_success += 1
                else:
                    print(f"sim spectra {my_key1} {my_key2} {spec1} {field}", check)
                ntest += 1


    for sid2, spec2 in enumerate(spec_name):
        if sid1 > sid2: continue

        my_analytic_cov = np.load("covariances/analytic_cov_%s_%s.npy" % (spec1, spec2))
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
            plt.plot(my_analytic_cov.diagonal(), ".", label=" my cov %s %s" % (spec1, spec2))
            plt.plot(analytic_cov_ref.diagonal(), label="reference %s %s" % (spec1, spec2))
            plt.legend()
            plt.subplot(2,1,2)
            plt.plot(my_analytic_cov.diagonal()/analytic_cov_ref.diagonal(), label=" my cov/cov ref")
            plt.legend()
            plt.savefig("%s/analytic_cov_diag_%s_%s.png" % (plot_dir, spec1, spec2), bbox_inches="tight")
            plt.clf()
            plt.close()

print("")
print("Summary of the tests")
print("")
print("%d tests succesful for %d tests total" % (ntest_success, ntest))
