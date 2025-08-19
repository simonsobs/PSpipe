import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

# Our simulation setup is as follows. A given simulation involves 99 null tests,
# matching the number of pwv and inout null tests. This 99 is assumed to be
# composed of 33 independent test sets (a test set could be a separate array
# or polarization cross), each of which is composed of 3 pwv cross differences:
# pwv1pwv1 - pwv1pwv2, pwv1pwv2 - pwv2pwv2, pwv1pwv1 - pwv2pwv2. Each "spectrum"
# has 39 "ell bins.""
nsim = 20_000 
ntest = 33  
nbin = 39 

sim_ptes = np.zeros((nsim, ntest, 3))

for sidx in range(nsim): 
    if sidx % 100 == 0:
        print(sidx)
    
    for tidx in range(ntest): 

        # simulate the 3 pwv cross spectra as Gaussian r.v's
        p11 = np.random.randn(nbin) 
        p22 = np.random.randn(nbin) 
        p12 = np.random.randn(nbin)

        # form the 3 null test residuals. this step introduces correlations
        # between null tests
        res_a = p11 - p22 
        res_b = p12 - p22 
        res_c = p11 - p12 

        # get the chi2 of each null test, assuming diagonal over null tests
        # and ell bins. The variance of a null test residual in a given ell 
        # bin is 2.
        chi2_a = np.sum(res_a ** 2 / 2) 
        chi2_b = np.sum(res_b ** 2 / 2) 
        chi2_c = np.sum(res_c ** 2 / 2) 

        # Get the ptes of the 3 null tests
        pte_a = 1 - stats.chi2.cdf(chi2_a, nbin)  
        pte_b = 1 - stats.chi2.cdf(chi2_b, nbin) 
        pte_c = 1 - stats.chi2.cdf(chi2_c, nbin) 

        # For each sim, we have 99 null tests; they are correlated along
        # the last axis which holds the pwv residuals
        sim_ptes[sidx,tidx,0] = pte_a 
        sim_ptes[sidx,tidx,1] = pte_b  
        sim_ptes[sidx,tidx,2] = pte_c

# Evaluate the kstest of the 99 null tests for each of our sims
ks_tests = np.zeros(nsim) 
for sidx in range(nsim):
    ks_tests[sidx] = stats.kstest(
        sim_ptes[sidx].reshape(-1), 'uniform', alternative='two-sided', mode='exact').pvalue
    
# Estimate the CDF from samples.
x = np.sort(ks_tests)
y = np.arange(x.size) / x.size
cdf_interp = interp1d(x, y, fill_value="extrapolate") 

ks_test_pwv = 0.018
ks_test_inout = 0.039

print(cdf_interp(ks_test_pwv))
print(cdf_interp(ks_test_pwv))
