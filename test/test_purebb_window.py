# pure BB spatial window follwoing Grain+ 2009
# will check a couple things and add comments soon -Steve 1/24/19

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from pixell import sharp
import copy
from pspy import so_map, so_window

def get_apodized_mask(nside,theta,fi,r,r_cut):
   # cos apodization
   npix = nside**2*12
   r *= np.pi/180.
   r_cut *= np.pi/180.  # start apodizing here
   center_vec = hp.ang2vec(theta*np.pi/180.,fi*np.pi/180.)
   ret = np.zeros(npix)
   pix = hp.query_disc(nside,center_vec,r)
   ret[pix] = 1
   for i in pix:
      if hp.rotator.angdist(center_vec,hp.pix2vec(nside,i)) >= r_cut:
         dist_to_center = hp.rotator.angdist(center_vec,hp.pix2vec(nside,i))
         ret[i] = np.cos(np.pi/(2*(r-r_cut))*(dist_to_center-r_cut))**2
   print 'fsky = %.6f'%(np.sum(ret)/float(npix))
   return ret

# =========== computing s1 and s2 windows for pure bb ============

nside = 128 
alm_order='rectangular'

theta = 90
fi = 0
r = 10.
r_cut = 8.

mask = get_apodized_mask(nside,theta,fi,r,r_cut)
lmax = 500

minfo = sharp.map_info_healpix(nside=nside)
ainfo = sharp.alm_info(lmax=lmax,layout=alm_order)
sht_rect = sharp.sht(minfo, ainfo)

def get_s1s2_win(sht_rect, w, lmax):
   mask = copy.copy(w)
   mask[mask!=0] = 1
   wlm0 = -sht_rect.map2alm(w).reshape(lmax+1,lmax+1).T
   wlm1_e = np.zeros([lmax+1,lmax+1], dtype='cfloat')
   wlm2_e = np.zeros([lmax+1,lmax+1], dtype='cfloat')

   ell = np.arange(lmax+1)
   for m in xrange(lmax + 1):
      wlm1_e[:, m] = np.sqrt((ell+1)*ell)*wlm0[:, m]
      wlm2_e[:, m] = np.sqrt((ell+2)*(ell+1)*ell*(ell-1))*(wlm0[:, m])

   wlm1_e[:1, :1] = 0
   wlm2_e[:2, :2] = 0
   wlm1_b = np.zeros_like(wlm1_e)
   wlm2_b = np.zeros_like(wlm2_e)
   w1 = sht_rect.alm2map([wlm1_e.T.reshape(-1),wlm1_b.reshape(-1)],spin=1)*np.array([mask, mask])
   w2 = sht_rect.alm2map([wlm2_e.T.reshape(-1),wlm2_b.reshape(-1)],spin=2)*np.array([mask, mask])
   return w1, w2

w1,w2 = get_s1s2_win(sht_rect,mask,lmax)
w1[w1==0] = hp.UNSEEN
w2[w2==0] = hp.UNSEEN
mask[mask==0] = hp.UNSEEN

m_mask = hp.mollview(mask,return_projected_map=True)
m_w10 = hp.mollview(w1[0],return_projected_map=True)
m_w11 = hp.mollview(w1[1],return_projected_map=True)
m_w20 = hp.mollview(w2[0],return_projected_map=True)
m_w21 = hp.mollview(w2[1],return_projected_map=True)
plt.clf()
plt.close('all')

plt.subplot(311)
plt.imshow(m_mask)
plt.title('spin 0')
plt.axis([360,440,160,240])

plt.subplot(323)
plt.imshow(m_w10)
plt.title('spin 1 Q')
plt.axis([360,440,160,240])

plt.subplot(324)
plt.imshow(m_w11)
plt.title('spin 1 U')
plt.axis([360,440,160,240])

plt.subplot(325)
plt.imshow(m_w20)
plt.title('spin 2 Q')
plt.axis([360,440,160,240])

plt.subplot(326)
plt.imshow(m_w21)
plt.title('spin 2 U')
plt.axis([360,440,160,240])
plt.savefig('so_purebb_window.png')
plt.clf()

