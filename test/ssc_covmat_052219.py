from pixell import enmap, curvedsky, powspec
import healpy as hp
import numpy as np
import pickle, os
from scipy.interpolate import interp1d

### input parameters
lmax  = 5000
lmin  = 2

window_dir   = '/global/homes/d/dwhan89/projects/act/data/spartial_window_functions/mr3c_20181012_190203'
window_file  = os.path.join(window_dir, 'deep56_mr3c_20181012_190203_master_apo_w0.fits')
window = enmap.read_fits(window_file)

len_theo_file   = '/global/homes/d/dwhan89/cori/workspace/ps_py/data/cosmo2017_10K_acc3_lensedCls.dat'
unlen_theo_file = '/global/homes/d/dwhan89/cori/workspace/ps_py/data/cosmo2017_10K_acc3_scalCls.dat'

ps_len  = powspec.read_spectrum(len_theo_file)
_, ps_pp = powspec.read_camb_scalar(unlen_theo_file)

ll   = np.arange(ps_len.shape[-1])
lp   = np.arange(ps_pp.shape[-1])

clkk = lp**2.*(lp+1.)**2/4.*ps_pp[0][0]

def calc_eff_geometry(window):
    loc  = np.where(window != 0)
    area = float(window[loc].size)/window.size*window.area()
    w2   = np.mean(window[loc]**2.)
    w4   = np.mean(window[loc]**4.)
    area_eff = area*(w2**2.)/w4
    fsky     = area/(4*np.pi)
    fsky_eff = area_eff/(4*np.pi)
    print fsky, fsky_eff
    return area_eff, fsky_eff

def take_delivaite(l, f, eps=0.1):
    f_interp = interp1d(l, f, bounds_error=False, fill_value=0.)
    df = f_interp(l+eps) - f_interp(l-eps)
    deriv = df/(2*eps)
    return deriv

def sum_over_wLM_sq(wLM):
    lmax  = hp.Alm.getlmax(wLM.shape[-1])
    wLMSQ = np.conj(wLM)*wLM  
    ret   = 2*np.sum(wLMSQ).real
    # m=0 is double counted 
    sidx = hp.Alm.getidx(lmax, 0, 0)
    eidx = hp.Alm.getidx(lmax, lmax, 0)
    ret  -= np.sum(wLMSQ[sidx:eidx]).real
    return ret

area_eff, fsky_eff =  calc_eff_geometry(window)
print(area_eff, fsky_eff)

# compute sigma kappa
wLM   = curvedsky.map2alm(window,lmax=lmax)
wLM   = hp.almxfl(wLM, np.sqrt(clkk))


sigma_kappa = np.sqrt(sum_over_wLM_sq(wLM))/area_eff
print sigma_kappa


### calculating numerical Cl derivatives 
l        = ll[:lmax+1]
deriv_tt = (take_delivaite(ll, ll**2.*ps_len[0][0])*ll)[:lmax+1]
deriv_te = (take_delivaite(ll, ll**2.*ps_len[0][1])*ll)[:lmax+1]
deriv_ee = (take_delivaite(ll, ll**2.*ps_len[1][1])*ll)[:lmax+1]

ssc_tttt = np.nan_to_num(np.outer(deriv_tt/l**2.,deriv_tt/l**2.))*sigma_kappa**2.
ssc_eeee = np.nan_to_num(np.outer(deriv_ee/l**2.,deriv_ee/l**2.))*sigma_kappa**2.
ssc_tete = np.nan_to_num(np.outer(deriv_te/l**2.,deriv_te/l**2.))*sigma_kappa**2.
ssc_ttee = np.nan_to_num(np.outer(deriv_tt/l**2.,deriv_ee/l**2.))*sigma_kappa**2.
ssc_ttte = np.nan_to_num(np.outer(deriv_tt/l**2.,deriv_te/l**2.))*sigma_kappa**2.
ssc_eete = np.nan_to_num(np.outer(deriv_ee/l**2.,deriv_te/l**2.))*sigma_kappa**2.

package = {'l':l, "TTTT":ssc_tttt}#, "EEEE":ssc_eeee, "TETE":ssc_tete, "TTEE":ssc_ttee, "TTTE":ssc_ttte, "EETE":ssc_eete} 

pickle.dump(package,open('/global/cscratch1/sd/dwhan89/temp/anlaytic_cov/ssc_cov.pkl','wb'))



