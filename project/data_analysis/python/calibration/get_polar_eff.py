from pspy import pspy_utils, so_dict, so_consistency
import numpy as np
import pickle
import sys


d = so_dict.so_dict()
d.read_from_file(sys.argv[1])

specDir = "../test_spectra_poleff"
covDir = "../test_covariances_poleff"

specFile = "Dl_%sx%s_cross.dat"
covFile = "analytic_cov_%sx%s_%sx%s.npy"

_, _, lb, _ = pspy_utils.read_binning_file(d["binning_file"], d["lmax"])
nBins = len(lb)

# Define the projection pattern - i.e. which
# spectra combination will be used to compute
# the residuals
projPattern = np.array([1, 0, -1])

# Mode to use to get the
# polarization efficiency
usedMode = "EE"

# Create output dirs
outputDir = f"polar_efficiency_results_{usedMode}"
pspy_utils.create_directory(outputDir)

residualOutputDir = f"{outputDir}/residuals"
pspy_utils.create_directory(residualOutputDir)

chainsDir = f"{outputDir}/chains"
pspy_utils.create_directory(chainsDir)

# Define the multipole range used to obtain
# the polarization efficiencies
multipoleRange = {"dr6_pa4_f150": [850, 3000],
                  "dr6_pa4_f220": [850, 3000],
                  "dr6_pa5_f090": [850, 3000],
                  "dr6_pa5_f150": [850, 3000],
                  "dr6_pa6_f090": [850, 3000],
                  "dr6_pa6_f150": [850, 3000]}
pickle.dump(multipoleRange, open(f"{outputDir}/multipole_range.pkl", "wb"))

# Define the reference arrays
refArrays = {"dr6_pa4_f150": "dr6_pa6_f150",
             "dr6_pa4_f220": "dr6_pa4_f220",
             "dr6_pa5_f090": "dr6_pa6_f090",
             "dr6_pa5_f150": "dr6_pa6_f150",
             "dr6_pa6_f090": "dr6_pa6_f090",
             "dr6_pa6_f150": "dr6_pa6_f150"}
pickle.dump(refArrays, open(f"{outputDir}/reference_arrays.pkl", "wb"))

calibDict = {}
for ar in d["arrays_dr6"]:
    array = f"dr6_{ar}"
    if array == refArrays[array]: continue
    refArray = refArrays[array]

    lb, specVec, fullCov = so_consistency.get_spectraVec_and_fullCov(f"{specDir}/{specFile}",
                                                                     f"{covDir}/{covFile}",
                                                                     array, refArray, usedMode, nBins)

    # Save and plot residuals before calibration
    resSpectrum, resCov = so_consistency.get_residual_spectra_and_cov(specVec, fullCov, projPattern)
    np.savetxt(f"{residualOutputDir}/residual_{array}_before.dat", np.array([lb, resSpectrum]).T)
    np.savetxt(f"{residualOutputDir}/residual_cov_{array}.dat", resCov)

    lmin, lmax = multipoleRange[array]
    id = np.where((lb >= lmin) & (lb <= lmax))
    so_consistency.plot_residual(lb[id], resSpectrum[id], resCov[np.ix_(id[0], id[0])],
                                 usedMode, array, f"{residualOutputDir}/residual_{array}_before")

    calMean, calStd = so_consistency.get_calibration_amplitudes(specVec, fullCov,
                                                                projPattern, id,
                                                                f"{chainsDir}/{array}")
    calibDict[array] = [calMean, calStd]

    if usedMode == "EE":
        calibVec = np.array([calMean**2, calMean, 1])
    elif usedMode == "TE":
        calibVec = np.array([calMean, 1, 1])

    resSpectrum, resCov = so_consistency.get_residual_spectra_and_cov(specVec, fullCov,
                                                                      projPattern, calibVec = calibVec)
    np.savetxt(f"{residualOutputDir}/residual_{array}_after.dat", np.array([lb, resSpectrum]).T)
    so_consistency.plot_residual(lb[id], resSpectrum[id], resCov[np.ix_(id[0], id[0])],
                                 usedMode, array, f"{residualOutputDir}/residual_{array}_after")

pickle.dump(calibDict, open(f"{outputDir}/polareff_dict.pkl", "wb"))
