import pixell
from pixell import enmap
from pixell import curvedsky
import healpy
import numpy
import pylab
import pickle

### input parameters
X = numpy.loadtxt('planckFINAL_lensedCls.dat')
Y = numpy.loadtxt('planckFINAL_scalCls.dat')
maskName = 'deep56_s14_pa1_f150_mr3c_20181012_190203_w0_cl0.00nK_pt1.00_nt0.0_T.fits'#'deep56_mr3c_20181012_190203_master_apo_w0.fits'#
ellmaxSigmaK = 5000 ## ellmax for sigmaK: something much smaller than patch size
ellmaxCovmat = 5000 ## ellmax out to which the covmatrices are calculated
ellmin = 2 ## ellmin from which cov matrices are calculated
plots = True
addInternalLensing = True

### set up theory
TCMB = 2.726e6
ll = X[:,0]
l = ll
lyy = Y[:,0]

clkap = Y[:,4]/(4.*TCMB**2)*(lyy*(lyy+1.))**2./lyy**4.
clkap = numpy.insert(clkap,0,0.)
clkap = numpy.insert(clkap,0,0.)

lsqClOver2piBB = X[:,3]
clBB = lsqClOver2piBB*(2*numpy.pi)/(l*(l+1.0))
###clBB /=TCMB**2 #### note: commenting out these lines gives output of covariance matrices in uK^4 (instead of dim.-less)
lsqClOver2piEE = X[:,2]
clEE = lsqClOver2piEE*(2*numpy.pi)/(l*(l+1.0))
###clEE /=TCMB**2
lsqClOver2piTT = X[:,1]
clTT = lsqClOver2piTT*(2*numpy.pi)/(l*(l+1.0))
###clTT /=TCMB**2
lsqClOver2piTE = X[:,4]
clTE = lsqClOver2piTE*(2*numpy.pi)/(l*(l+1.0))
###clTE /=TCMB**2

### read in mask and calculate mean-kappa variance
print('step 1: calculating sigma_kappa^2')
imap = enmap.read_map(maskName)
imap /= imap.max() ### normalizing step
flm = pixell.curvedsky.map2alm(imap,lmax=ellmaxSigmaK)
flmsq = numpy.real(numpy.conjugate(flm)*flm)
value = numpy.real(numpy.sum(healpy.almxfl(flmsq,clkap))) ### equation 14 of 1810.09347 without area-squared division (applied in last step)
Area = imap.sum()/imap.size * imap.area()
reductionFactor = (numpy.mean(imap**2.)**2./numpy.mean(imap**4.))
fSkyFixed =imap.area()/4./3.14159*reductionFactor
#print(fSkyFixed, fSkyFixed/41253.,'fskyfixed', 'fskyfixeddebugged')
print(reductionFactor, 'reduction factor')
print(Area/4./3.14159*41253., '= area in square degs.', imap.area()/4./3.14159*41253.*reductionFactor,'=corrected area in sqdeg') ### commented out as only true if mask has unit value
sigmaKSq = value/Area**2.
print(sigmaKSq,'final sigma_kappa^2 for this mask')

### calculating covariance matrices
print('step 2: calculating covariance matrices')

### calculating numerical Cl derivatives #### n.b. later I think sign is flipped but irrelevant if always taking outer products
l2dTT = (numpy.roll(l**2*clTT,1)-l**2*clTT)
l2dTT[0] = 0.
l2dEE = (numpy.roll(l**2*clEE,1)-l**2*clEE)
l2dEE[0] = 0.
l2dTE = (numpy.roll(l**2*clTE,1)-l**2*clTE)
l2dTE[0] = 0.
l2dBB = (numpy.roll(l**2*clBB,1)-l**2*clBB)
l2dBB[0] = 0.

derivTT = l2dTT/l
derivTE = l2dTE/l
derivEE = l2dEE/l
derivBB = l2dBB/l

print(derivTT[100],'derivative')

if plots:### plot derivative test plots if required
    pylab.clf()
    pylab.loglog(l,numpy.abs(derivTT),label='TT')
    pylab.loglog(l,numpy.abs(derivTT),label='TE')
    pylab.loglog(l,numpy.abs(derivEE),label='EE')
    pylab.loglog(l,numpy.abs(derivBB),label='BB')
    pylab.savefig('newDerivCheck.png')

    #### test plot
    int = clTT.copy()
    for i in numpy.arange(9000):
        int[i] = numpy.sum(l2dTT[0:i])
    pylab.loglog(l,l**2*clTT)
    pylab.loglog(l,numpy.abs(l2dTT))
    pylab.loglog(l,numpy.abs(int))
    pylab.savefig('derivs.png')
    #### test plot
    pylab.clf()
    pylab.loglog(l,numpy.abs(derivTT))
    pylab.savefig('derivPlot.png')
    pylab.clf()

derivTT = derivTT[numpy.where(  (ellmin<l)&(l<(ellmaxCovmat+1)) )]
derivTE = derivTE[numpy.where(  (ellmin<l)&(l<(ellmaxCovmat+1)) )]
derivEE = derivEE[numpy.where(  (ellmin<l)&(l<(ellmaxCovmat+1)) )]
clTT = clTT[numpy.where(  (ellmin<l)&(l<(ellmaxCovmat+1)) )]
clTE = clTE[numpy.where(  (ellmin<l)&(l<(ellmaxCovmat+1)) )]
clEE = clEE[numpy.where( (ellmin<l)&(l<(ellmaxCovmat+1)) )]
l = l[numpy.where( (ellmin<l)&(l<(ellmaxCovmat+1)) )]


#### calculate SSC covariance matrices
covTTmatrix = numpy.outer(derivTT,derivTT)*sigmaKSq
covEEmatrix = numpy.outer(derivEE,derivEE)*sigmaKSq
covTEmatrix = numpy.outer(derivTE,derivTE)*sigmaKSq
covTTEEmatrix = numpy.outer(derivTT,derivEE)*sigmaKSq
covTTTEmatrix = numpy.outer(derivTT,derivTE)*sigmaKSq
covEETEmatrix = numpy.outer(derivEE,derivTE)*sigmaKSq

print('shape of matrix', covTTmatrix.shape)

covDict = {"ell":l, "TTTT":covTTmatrix, "EEEE":covEEmatrix, "TETE":covTEmatrix, "TTEE":covTTEEmatrix, "TTTE":covTTTEmatrix, "EETE":covEETEmatrix}

pickle.dump(covDict,open('covarianceMatricesSSC.pkl','wb'))


#### read in internal lensing covmat
if addInternalLensing:
    x = numpy.ones(4998)
    x[l>2500.] *= 0.
    xmat = numpy.outer(x,x)
    print('shape',xmat.shape)

    print('adding internal lensing')
    fArea= 1/fSkyFixed*xmat#/41253.###xmat zeros out ells above 2200 -- this is a hack
#    internalCov = pickle.load(open('offDiagCov.pkl'),encoding='utf-8')
    covTTmatrix += numpy.load('internalCovTTTT.npy')*fArea#internalCov['lensed']['cl_TT']['cl_TT']
    covEEmatrix += numpy.load('internalCovEEEE.npy')*fArea#internalCov['lensed']['cl_EE']['cl_EE']
    covTEmatrix += numpy.load('internalCovTETE.npy')*fArea#internalCov['lensed']['cl_TE']['cl_TE']
    covTTEEmatrix += numpy.load('internalCovEETT.npy').T*fArea#internalCov['lensed']['cl_EE']['cl_TT'].T
    covTTTEmatrix += numpy.load('internalCovTETT.npy').T*fArea#internalCov['lensed']['cl_TE']['cl_TT'].T
    covEETEmatrix += numpy.load('internalCovEETE.npy')*fArea#internalCov['lensed']['cl_EE']['cl_TE']

covDict = {"ell":l, "TTTT":covTTmatrix, "EEEE":covEEmatrix, "TETE":covTEmatrix, "TTEE":covTTEEmatrix, "TTTE":covTTTEmatrix, "EETE":covEETEmatrix}

pickle.dump(covDict,open('covarianceMatricesFull.pkl','wb'))


#### make plots!
fSky = Area/4./3.14159 
#### TT,TT
pylab.matshow(((covTTmatrix.T)),origin='lower')
pylab.savefig('covTTmatrix.png')
pylab.clf()

corrTTmatrix = covTTmatrix/numpy.sqrt(numpy.outer(2./(2.*l+1)*clTT**2./fSky,2./(2.*l+1)*clTT**2./fSky))
pylab.matshow(((corrTTmatrix.T)),origin='lower')
pylab.xlabel(r'$\ell_1$')
pylab.ylabel(r'$\ell_2$')
pylab.title('TT,TT corr.')
pylab.colorbar()
pylab.savefig('corrTTmatrix.png')
pylab.clf()

#### EE, EE
pylab.matshow(((covEEmatrix.T)),origin='lower')
pylab.savefig('covEEmatrix.png')
pylab.clf()

sigma = 1.
corrEEmatrix = covEEmatrix/numpy.sqrt(numpy.outer(2./(2.*l+1)*clEE**2./fSky,2./(2.*l+1)*clEE**2./fSky))
pylab.matshow(((corrEEmatrix.T)),origin='lower')
pylab.xlabel(r'$\ell_1$')
pylab.ylabel(r'$\ell_2$')
pylab.title('EE,EE corr.')
pylab.colorbar()
pylab.savefig('corrEEmatrix.png')
pylab.clf()


#### TE, TE
pylab.matshow(((covTEmatrix.T)),origin='lower')
pylab.savefig('covTEmatrix.png')
pylab.clf()

corrTEmatrix = covTEmatrix/numpy.sqrt(numpy.outer(1./(2.*l+1)*(clEE*clTT+clTE**2.)/fSky,1./(2.*l+1)*(clEE*clTT+clTE**2.)/fSky))
pylab.matshow(((corrTEmatrix.T)),origin='lower')
pylab.xlabel(r'$\ell_1$')
pylab.ylabel(r'$\ell_2$')
pylab.title('TE,TE corr.')
pylab.colorbar()
pylab.savefig('corrTEmatrix.png')
pylab.clf()

#### TT, EE
pylab.matshow(((covTTEEmatrix.T)),origin='lower')
pylab.savefig('covTTEEmatrix.png')
pylab.clf()

corrTTEEmatrix = covTTEEmatrix/numpy.sqrt(numpy.outer(2./(2.*l+1)*clTT**2./fSky,2./(2.*l+1)*clEE**2./fSky))
pylab.matshow(((corrTTEEmatrix.T)),origin='lower')
pylab.xlabel(r'$\ell_1$')
pylab.ylabel(r'$\ell_2$')
pylab.title('TT,EE corr.')
pylab.colorbar()
pylab.savefig('corrTTEEmatrix.png')
pylab.clf()

#### TT, TE
pylab.matshow(((covTTTEmatrix.T)),origin='lower')
pylab.savefig('covTTTEmatrix.png')
pylab.clf()

corrTTTEmatrix = covTTTEmatrix/numpy.sqrt(numpy.outer(2./(2.*l+1)*(clTT**2.)/fSky,1./(2.*l+1)*(clEE*clTT+clTE**2.)/fSky))
pylab.matshow(((corrTTTEmatrix.T)),origin='lower')
pylab.xlabel(r'$\ell_1$')
pylab.ylabel(r'$\ell_2$')
pylab.title('TT,TE corr.')
pylab.colorbar()
pylab.savefig('corrTTTEmatrix.png')
pylab.clf()

#### EE, TE
pylab.matshow(((covEETEmatrix.T)),origin='lower')
pylab.savefig('covEETEmatrix.png')
pylab.clf()

corrEETEmatrix = covEETEmatrix/numpy.sqrt(numpy.outer(2./(2.*l+1)*(clEE**2.)/fSky,1./(2.*l+1)*(clEE*clTT+clTE**2.)/fSky))
pylab.matshow(((corrEETEmatrix.T)),origin='lower')
pylab.xlabel(r'$\ell_1$')
pylab.ylabel(r'$\ell_2$')
pylab.title('EE,TE corr.')
pylab.colorbar()
pylab.savefig('corrEETEmatrix.png')
pylab.clf()
