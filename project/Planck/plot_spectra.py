import numpy as np
import pylab as plt



name=['TT_100x100','TT_143x143','TT_143x217','TT_217x217']

plt.semilogy()
for n in name:
    print (n)
    lmin,lmax,l,cl,error=np.loadtxt('planck_spectra/spectra_' + n + '.dat',unpack=True)
    plt.errorbar(l,cl*l**2/(2*np.pi),error*l**2/(2*np.pi),fmt='.',label='%s'%n)
plt.legend()
plt.show()

name=['EE_100x100','EE_100x143','EE_100x217','EE_143x143','EE_143x217','EE_217x217']
for n in name:
    print (n)
    lmin,lmax,l,cl,error=np.loadtxt('planck_spectra/spectra_' + n + '.dat',unpack=True)
    plt.errorbar(l,cl*l**2/(2*np.pi),error*l**2/(2*np.pi),fmt='.',label='%s'%n)
plt.legend()
plt.show()

name=['TE_100x100','TE_100x143','TE_100x217','TE_143x143','TE_143x217','TE_217x217']
for n in name:
    print (n)
    lmin,lmax,l,cl,error=np.loadtxt('planck_spectra/spectra_' + n + '.dat',unpack=True)
    plt.errorbar(l,cl*l**2/(2*np.pi),error*l**2/(2*np.pi),fmt='.',label='%s'%n)
plt.legend()
plt.show()


name=['100hm1x100hm2','100hm1x143hm2','100hm1x217hm2','143hm1x143hm2','143hm1x217hm2','217hm1x217hm2']

for n in name:
    print (n)
    l,bl=np.loadtxt('planck_beam/beam_' + n + '.dat',unpack=True)
    plt.plot(l,bl,label='%s'%n)
plt.legend()
plt.show()
