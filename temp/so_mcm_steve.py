"""
@brief: python routines for mode coupling calculation.
Should include an option for pure B mode estimation
"""
import healpy as hp, pylab as plt, numpy as np
from mmcm_v7 import mmcmatrix_v7 as mm


def P_bl(b, l, lbands):    # binning
    if 2<=lbands[b,0] and lbands[b,0]<=l and l<=lbands[b,1]:
        return 1.0 / (2.0*np.pi) * l*(l+1) / (lbands[b,1]-lbands[b,0]+1)
    else:
        return 0

def Q_lb(l, b, lbands):    # inverse binning
    if 2<=lbands[b,0] and lbands[b,0]<=l and l<=lbands[b,1]:
        return 2.0*np.pi / (l * (l+1))
    else:
        return 0

def binning_op(lbands):
    lmax = int(lbands[-1][-1])
    nbins = len(lbands)
    m_P_bl = np.zeros((nbins, lmax+1))
    m_Q_lb = np.zeros((lmax+1, nbins))
    for l in xrange(lmax+1):
        for b in xrange(nbins):
            m_P_bl[b,l] = P_bl(b,l,lbands)
            m_Q_lb[l,b] = Q_lb(l,b,lbands)
            M_P_bl = np.append(np.append(m_P_bl,np.zeros(m_P_bl.shape),axis = 1), np.append(np.zeros(m_P_bl.shape),m_P_bl,axis = 1), axis = 0)
            M_Q_lb = np.append(np.append(m_Q_lb,np.zeros(m_Q_lb.shape),axis = 1),np.append(np.zeros(m_Q_lb.shape),m_Q_lb,axis = 1), axis = 0)
    return m_P_bl, m_Q_lb, M_P_bl, M_Q_lb

def read_binfile(binning_file,lcut=100000):
    lbands = np.array(np.genfromtxt('%s'%binning_file)[:,:2],dtype='int')
    if len(np.where(lbands[:,1]>lcut)[0]) != 0:
        lbands = lbands[:len(lbands)-len(np.where(lbands[:,1]>lcut)[0])]
    if lbands[0,0] == 0:
        lbands[0,0] = 2
    return lbands


def mcm_and_bbl_TT_steve(wlm1, binning_file,lmax, wlm2=None,bl1=None,bl2=None,type='Dl'):
    
    lbands = read_binfile(binning_file, lcut=lmax)
    lmax = lbands[-1,1]
    ell = np.arange(lmax + 1)
    m_P_bl, m_Q_lb = binning_op(lbands)[:2]
    if bl1 is None:
        bl1=np.ones(lmax+1)

    if wlm2 is not None:
        if bl2 is None:
            bl2 = bl1
        w_tt = hp.alm2cl(wlm1, wlm2)
        m_tt = mm.mmcm_tt(w_tt, ell)
        for i in ell:
            m_tt[i] *= bl1 * bl2
    else:
        w_tt = hp.alm2cl(wlm1)
        w_tt=w_tt[:lmax+1]
        m_tt = mm.mmcm_tt(w_tt, ell)
        for i in ell:
            m_tt[i] *= bl1[:lmax+1]**2

    mbb, Bbl= np.dot(m_P_bl, np.dot(m_tt, m_Q_lb)), np.dot(np.linalg.inv(np.dot(m_P_bl, np.dot(m_tt, m_Q_lb))), np.dot(m_P_bl, m_tt))
    return mbb, Bbl

def mcm_and_bbl_EEBB_steve(wlm1, binning_file,lmax, wlm2=None,bl1=None,bl2=None,type='Dl'):
    lbands = read_binfile(binning_file, lcut=lmax)
    lmax = lbands[-1, 1]
    print lbands,lmax
    ell = np.arange(lmax + 1)
    M_P_bl, M_Q_lb = binning_op(lbands)[2:]
    
    if bl1 is None:
        bl1=np.ones(lmax+1)

    if wlm2 is not None:
        if bl2 is None:
            bl2 = bl1
        w_pp = hp.alm2cl(wlm1, wlm2)
        m_ee, m_eebb = mm.mmcm_eebb(w_pp, ell)
        for i in ell:
            m_ee[i] *= bl1 * bl2
            m_eebb[i] *= bl1 * bl2
                                
        M_eebb = np.append(np.append(m_ee,m_eebb,axis = 1),np.append(m_eebb,m_ee,axis = 1),axis = 0)
    else:
        w_pp = hp.alm2cl(wlm1)
        w_pp=w_pp[:lmax+1]
        m_ee, m_eebb = mm.mmcm_eebb(w_pp, ell)
        for i in ell:
            m_ee[i] *= bl1[:lmax+1]**2
            m_eebb[i] *= bl1[:lmax+1]**2
        M_eebb = np.append(np.append(m_ee,m_eebb,axis = 1),np.append(m_eebb,m_ee,axis = 1),axis = 0)

    mbb, Bbl= np.dot(M_P_bl, np.dot(M_eebb, M_Q_lb)), np.dot(np.linalg.inv(np.dot(M_P_bl, np.dot(M_eebb, M_Q_lb))), np.dot(M_P_bl, M_eebb))
    return mbb, Bbl


def mcm_and_bbl_TEB_steve(wlm1, binning_file,lmax, wlm2=None,bl1=None,bl2=None,type='Dl'):
    lbands = read_binfile(binning_file, lcut=lmax)
    lmax = lbands[-1, 1]
    ell = np.arange(lmax + 1)
    m_P_bl, m_Q_lb = binning_op(lbands)[:2]
    
    if bl1 is None:
        bl1=np.ones(lmax+1)

    if wlm2 is not None:
        if bl2 is None:
            bl2 = bl1
        w_tp = hp.alm2cl(wlm1, wlm2)
        m_te = mm.mmcm_te(w_tp, ell)
        for i in ell:
            m_te[i] *= bl1 * bl2
    else:
        w_tp = hp.alm2cl(wlm1)
        w_tp=w_tp[:lmax+1]

        m_te = mm.mmcm_te(w_tp, ell)
        for i in ell:
            m_te[i] *= bl1[:lmax+1]**2
    mbb, Bbl= np.dot(m_P_bl, np.dot(m_te, m_Q_lb)), np.dot(np.linalg.inv(np.dot(m_P_bl, np.dot(m_te, m_Q_lb))), np.dot(m_P_bl, m_te))
    return mbb, Bbl

def mcm_and_bbl_EB_steve(wlm1, binning_file,lmax, wlm2=None,bl1=None,bl2=None,type='Dl'):
    lbands = read_binfile(binning_file, lcut=lmax)
    lmax = lbands[-1, 1]
    ell = np.arange(lmax + 1)
    m_P_bl, m_Q_lb = binning_op(lbands)[:2]
    
    if bl1 is None:
        bl1=np.ones(lmax+1)

    
    if wlm2 is not None:
        if bl2 is None:
            bl2 = bl1
        w_pp = hp.alm2cl(wlm1, wlm2)
        m_eb = mm.mmcm_eb(w_pp, ell)
        for i in ell:
            m_eb[i] *= bl1 * bl2
    else:
        w_pp = hp.alm2cl(wlm1)
        w_pp=w_pp[:lmax+1]

        m_eb = mm.mmcm_eb(w_pp, ell)
        for i in ell:
            m_eb[i] *= bl1[:lmax+1]**2
    mbb, Bbl= np.dot(m_P_bl, np.dot(m_eb, m_Q_lb)), np.dot(np.linalg.inv(np.dot(m_P_bl, np.dot(m_eb, m_Q_lb))), np.dot(m_P_bl, m_eb))
    return mbb, Bbl



