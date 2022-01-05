from pspy import so_spectra, so_cov, pspy_utils
import pylab as plt, numpy as np
from cobaya.run import run



def sigurd_func(lb, a, lk, cst):
    oof = 1 / (1 + (lb/lk)**a)
    return oof/(oof+cst)
    
def thib_func(lb, eps_min, lmin, lmax):
    func = eps_min + (1-eps_min)*np.sin(np.pi/2*(lb-lmin)/(lmax-lmin))**2
    func[lb > lmax] = 1
    return func
    
def beta_func(lb, eps_min, lmax, beta):
    func = np.zeros(len(lb))
    id = np.where(lb < lmax)
    func[id] = eps_min + (1-eps_min) / (1 + (lb[id]/(lmax-lb[id]))**(-beta))
    func[lb>= lmax] = 1
    return func
    



for my_spec in ["dr6_pa4_f150", "dr6_pa4_f220", "dr6_pa5_f090", "dr6_pa5_f150",  "dr6_pa6_f090", "dr6_pa6_f150"]:
    
    lb, TF1, std_TF1 = np.loadtxt("TF_%s.dat" % my_spec, unpack=True)
    lb, TF2, std_TF2 = np.loadtxt("TF_%s_cross.dat" % my_spec, unpack=True)

    
    def compute_loglike(eps_min, lmax, beta):
        vec_res = TF2 - beta_func(lb, eps_min, lmax, beta)
        chi2 = np.sum(vec_res ** 2 / std_TF2 ** 2)
        return -0.5 * chi2
        
    print(compute_loglike(0.5, 500, 3))
    info = {
        "likelihood": {"my_like": compute_loglike},
        "params": {
            "eps_min": {"prior": {"min": 0, "max": 1}, "latex": r"\epsilon_{min}"},
            "lmax": {"prior": {"min": 100, "max": 1500}, "latex": r"\ell_{max}"},
            "beta": {"prior": {"min": 0, "max": 5}, "latex": r"\beta"},
        },
        "sampler": {
            "mcmc": {
                "max_tries": 10 ** 8,
                "Rminus1_stop": 0.001,
                "Rminus1_cl_stop": 0.008,
            }
        },
        "output": "chains/mcmc_%s" % my_spec,
        "force": True,
    }

    updated_info, sampler = run(info)
