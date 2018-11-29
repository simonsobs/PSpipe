from pspy import so_map,sph_tools
import so_mcm,so_mcm_steve
import healpy as hp, pylab as plt, numpy as np
import matplotlib.cm as cm


l_th,cl_TT,cl_EE,cl_BB,cl_TE=np.loadtxt('../data/bode_almost_wmap5_lmax_1e4_lensedCls_startAt2.dat',unpack=True)

#Spin0 test
win_spin0=so_map.read_map('window.fits')
l,bl_spin0=np.loadtxt('../data/beam.txt',unpack=True)
lmax=976
win_spin0= sph_tools.map2alm(win_spin0,niter=3,lmax=lmax)
binning_file='../data/BIN_ACTPOL_50_4_SC_low_ell_startAt2'

bin_lo,bin_hi,bin_c = plt.loadtxt(binning_file,unpack=True)
id = np.where(bin_hi <lmax)
bin_lo,bin_hi,bin_c=bin_lo[id],bin_hi[id],bin_c[id]


mbb, Bbl= so_mcm.mcm_and_bbl_spin0(win_spin0,binning_file,lmax, 'Dl', bl1=bl_spin0,input_alm=True)
mbb_steve_TT, Bbl_steve_TT= so_mcm_steve.mcm_and_bbl_TT_steve(win_spin0,binning_file,lmax,bl1=bl_spin0[:lmax+1],type='Dl')




cb_steve=np.dot(Bbl_steve_TT,cl_TT[:lmax]*2*np.pi/(l_th[:lmax]*(l_th[:lmax]+1)))
cb_th=np.dot(Bbl,cl_TT[:lmax])
Bbl=np.load('/Users/thibaut/Desktop/Project/so_ps_codes/test/pspy_test/mosaic/mcm_car/Bbl_T_000_143GHzx143GHz.npy')

cb_th2=np.dot(Bbl,cl_TT[:lmax])

plt.semilogy()
plt.plot(l_th,cl_TT)
plt.plot(bin_c,cb_th2,'o')
plt.plot(bin_c,cb_th)
plt.show()

sys.exit()

nbins= mbb.shape[0]

print nbins,Bbl.shape,Bbl_steve_TT.shape

colors = cm.rainbow(np.linspace(0, 1, nbins))
print colors.shape
for i,color in zip(np.arange(nbins),colors ):
    plt.plot(Bbl[i,:],color=color,apha=0.3)
    plt.plot(Bbl_steve_TT[i,:],'-.',color=color)
plt.show()

#Spin0 and Spin2 test
win_spin2=so_map.read_map('window_pol.fits')
win_spin2= sph_tools.map2alm(win_spin2,niter=3,lmax=lmax)
l,bl_spin2=np.loadtxt('../data/beam.txt',unpack=True)


win1_tuple=(win_spin0,win_spin2)
bl1_tuple=(bl_spin0,bl_spin2)

mbb, Bbl= so_mcm.mcm_and_bbl_spin0and2(win1_tuple,binning_file,lmax,bl1=bl1_tuple,type='Dl',input_alm=True)
mbb_array= so_mcm.dict_to_array(mbb)

plt.matshow( np.log(np.abs(mbb_array)))
plt.colorbar()
plt.show()
sys.exit()

mbb_steve_TT, Bbl_steve_TT= so_mcm_steve.mcm_and_bbl_TT_steve(win_spin0,binning_file,lmax,bl1=bl_spin0[:lmax+1],type='Dl')
mbb_steve_EEBB, Bbl_steve_EEBB= so_mcm_steve.mcm_and_bbl_EEBB_steve(win_spin0,binning_file,lmax,bl1=bl_spin0[:lmax+1],type='Dl')
mbb_steve_TEB, Bbl_steve_TEB= so_mcm_steve.mcm_and_bbl_TEB_steve(win_spin0,binning_file,lmax,bl1=bl_spin0[:lmax+1],type='Dl')
mbb_steve_EB, Bbl_steve_EB= so_mcm_steve.mcm_and_bbl_EB_steve(win_spin0,binning_file,lmax,bl1=bl_spin0[:lmax+1],type='Dl')


plt.matshow( (mbb_steve_TT-mbb['spin0xspin0'])/mbb_steve_TT)
plt.title('(steve(TT)-SO(TT))/steve(TT)', fontsize=22)
plt.colorbar()
plt.show()

nbins= mbb['spin0xspin0'].shape[0]
plt.matshow( (mbb_steve_EEBB[:nbins,:nbins] -mbb['spin2xspin2'] [:nbins,:nbins] )/mbb_steve_EEBB[:nbins,:nbins])
plt.title('(steve(EE->EE)-SO(EE->EE)/steve(EE->EE)', fontsize=22)
plt.colorbar()
plt.show()

plt.matshow( (mbb_steve_EEBB[nbins:2*nbins,:nbins] -mbb['spin2xspin2'] [3*nbins:4*nbins,:nbins] )/mbb_steve_EEBB[nbins:2*nbins,:nbins])
plt.title('(steve(EE->BB)-SO(EE->BB)/steve(EE->BB)', fontsize=22)
plt.colorbar()
plt.show()

plt.matshow( (mbb_steve_TEB -mbb['spin0xspin2'] )/mbb_steve_TEB)
plt.title('(steve(TE)-SO(TE)/steve(TE)', fontsize=22)
plt.colorbar()
plt.show()
