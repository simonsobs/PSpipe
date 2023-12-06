import numpy as np
from scipy import stats
from scipy.interpolate import interp1d

nbin = 39 
nsim = 20_000 
ntest = 33  
sim_ptes = np.zeros((nsim, ntest, 3))
for sidx in range(nsim): 
    if sidx % 100 == 0: print(sidx)
    for tidx in range(ntest): 
        p11 = np.random.randn(nbin)  
        p22 = np.random.randn(nbin)  
        p12 = np.random.randn(nbin) 
        res_a = p11 - p22 
        res_b = p12 - p22 
        res_c = p11 - p12 
        chi2_a = np.sum(res_a ** 2 / 2) 
        chi2_b = np.sum(res_b ** 2 / 2) 
        chi2_c = np.sum(res_c ** 2 / 2) 
        pte_a = 1 - stats.chi2.cdf(chi2_a, nbin)  
        pte_b = 1 - stats.chi2.cdf(chi2_b, nbin) 
        pte_c = 1 - stats.chi2.cdf(chi2_c, nbin) 
        sim_ptes[sidx,tidx,0] = pte_a 
        sim_ptes[sidx,tidx,1] = pte_b  
        sim_ptes[sidx,tidx,2] = pte_c

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

print(1 - cdf_interp(ks_test_pwv))
print(1 - cdf_interp(ks_test_pwv))
