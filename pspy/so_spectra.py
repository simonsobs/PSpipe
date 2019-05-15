"""
@brief: python routines for power spectra estimation and debiasing.
"""
from __future__ import absolute_import, print_function
import healpy as hp, numpy as np
from pspy import pspy_utils,so_mcm

def get_spectra(alm1,alm2=None,spectra=None):
    """
    @brief get the power spectrum of alm1 and alm2, we use healpy.alm2cl for doing this.
    for the spin0 and spin2 case it is a bit ugly as we have to deal with healpix convention.
    Our  convention for spectra is:  ['TT','TE','TB','ET','BT','EE','EB','BE','BB']
    or a corresponding convention for spin0 and spin2 fields (replacing T by gal and E by kappa for example).
    while healpix convention is to take alm1,alm2 and return ['TT','EE','BB','TE','EB','TB']
    @param alm1 the spherical harmonic transform of map1
    @param (optional) alm2 the spherical harmonic transform of map2
    @param (optional) spectra,  needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    @return l the multipole
    @return cl the 1d array power spectrum or cl_dict (for spin0 and spin2) a dictionnary of cl with entry spectra
    """

    if spectra is None:
        if alm2 is None:
            cls=hp.sphtfunc.alm2cl(alm1)
        else:
            cls=hp.sphtfunc.alm2cl(alm1,alm2)
        l=np.arange(len(cls))
        return l,cls
    else:
        cl_dict={}
        cls=hp.sphtfunc.alm2cl(alm1,alm2)
        l=np.arange(len(cls[0]))
        spectra_healpix=[spectra[0],spectra[5],spectra[8],spectra[1],spectra[6],spectra[2]]
        for c,f in enumerate(spectra_healpix):
            cl_dict[f]=cls[c]

        if alm2 is None:
            cl_dict[spectra[3]]= cl_dict[spectra[1]]
            cl_dict[spectra[7]]= cl_dict[spectra[6]]
            cl_dict[spectra[4]]= cl_dict[spectra[2]]
        else:
            #here we need to recompute cls inverting the order of the alm to get ET,BT and BE
            cls=hp.sphtfunc.alm2cl(alm2,alm1)
            spectra_healpix=[spectra[0],spectra[5],spectra[8],spectra[3],spectra[7],spectra[4]]
            for c,f in enumerate(spectra_healpix):
                cl_dict[f]=cls[c]


        return l,cl_dict

def bin_spectra(l,cl,binning_file,lmax,type,spectra=None,mbb_inv=None,mcm_inv=None):
    
    """
    @brief bin the power spectra
    @param l: the multipole
    @param cl: the power spectra to bin, can be a 1d array (spin0) or a dictionnary (spin0 and spin2)
    @param binning_file: a binning file with format bin low, bin high, bin mean
    @param lmax: the maximum multipole to consider
    @param type: the type of binning, either bin Cl or bin Dl
    @param (optional) spectra: needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    @param (optional) mbb_inv: the inverse of the mode coupling matrix to debiais the spectra
    @return bin_c: 1d array with the center of the bin
    @return binnedPower: 1d array with the binned power spectra
    """
    
    
    if mbb_inv is not None and mcm_inv is not None:
        print ('Error: you have to choose between binned or unbinned mcm')
        sys.exit()

    
    bin_lo,bin_hi,bin_c,bin_size= pspy_utils.read_binning_file(binning_file,lmax)
    n_bins=len(bin_hi)
    
    if type=='Dl':
        fac=(l*(l+1)/(2*np.pi))
    if type=='Cl':
        fac=l*0+1 

    if spectra is None:
        if mcm_inv is not None:
            cl=np.dot(mcm_inv,cl)
        binnedPower=np.zeros(len(bin_c))
        for ibin in range(n_bins):
            loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
            binnedPower[ibin] = (cl[loc]*fac[loc]).mean()
        if mbb_inv is None:
            return bin_c,binnedPower
        else:
            return bin_c,np.dot(mbb_inv,binnedPower)
    else:
        if mcm_inv is not None:
            unbin_vec=[]
            mcm_inv=so_mcm.coupling_dict_to_array(mcm_inv)
            for f in spectra:
                unbin_vec=np.append(unbin_vec,cl[f][2:])
            cl=vec2spec_dict(lmax-2,np.dot(mcm_inv[:-2,:-2],unbin_vec),spectra)

        vec=[]
        for f in spectra:
            binnedPower=np.zeros(len(bin_c))
            for ibin in range(n_bins):
                loc = np.where((l >= bin_lo[ibin]) & (l <= bin_hi[ibin]))
                binnedPower[ibin] = (cl[f][loc]*fac[loc]).mean()
            
            vec=np.append(vec,binnedPower)
        if mbb_inv is None:
            return bin_c,vec2spec_dict(n_bins,vec,spectra)
        else:
            mbb_inv=so_mcm.coupling_dict_to_array(mbb_inv)
            return bin_c,vec2spec_dict(n_bins,np.dot(mbb_inv,vec), spectra)

    return(binCent,binnedPower)

def vec2spec_dict(n_bins,vec,spectra):
    """
    @brief take a vector of power spectra and return a power spectra dictionnary
    vec should be of the form [spectra[0], spectra[1], ... ]
    """
    dict={}
    for c,f in enumerate(spectra):
        dict[f]=vec[c*n_bins:(c+1)*n_bins]
    return dict


def write_ps(file_name,l,ps,type,spectra=None):
    """
    @brief write down the power spectra
    @param l the angular multipole
    @param ps the power spectrum, if spectra is not None, expect a dictionary with entry spectra
    @param type: can be 'Cl' or 'Dl'
    @param (optional) spectra: needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    """

    if spectra is None:
        ps_list=[ps]
        ps_list[0:0]=[l]
        str='l'
        str+=' %s'%type
        ps_list=np.array(ps_list)
        np.savetxt(file_name, np.transpose(ps_list),header=str)
    else:
        ps_list=[ps[f] for f in spectra]
        ps_list[0:0]= [l]
        str='l'
        for l1 in spectra:
            str+=' %s_%s'%(type,l1)
        ps_list=np.array(ps_list)
        np.savetxt(file_name, np.transpose(ps_list),header=str)


def read_ps(file_name,spectra=None):
    """
    @brief read the power spectra
    @param file_name: the name of the datafile
    @param (optional) spectra: needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    @return l,ps
    """
    data=np.loadtxt(file_name)
    if spectra is None:
        return data[:,0],data[:,1]
    else:
        ps={}
        l=data[:,0]
        for c,f in enumerate(spectra):
            ps[f]=data[:,c+1]
        return(l,ps)



def write_ps_hdf5(file,spec_name,l,ps,spectra=None):
    """
    @brief write down the power spectra in hdf5
    @param file: the name of the hdf5 file
    @param spec_name: the name of the group in the hdf5 file
    @param l the angular multipole
    @param ps the power spectrum, if spectra is not None, expect a dictionary with entry spectra
    @param (optional) spectra: needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    """

    def array_from_dict(l,ps,spectra=None):
        array=[]
        array+=[l]
        if spectra==None:
            array+=[ps]
        else:
            for spec in spectra:
                array+=[ps[spec]]
        return(array)
    
    group=file.create_group(spec_name)
    array=array_from_dict(l,ps,spectra=spectra)
    group.create_dataset(name='data',data=array,dtype='float')


def read_ps_hdf5(file,spec_name,spectra=None):
    """
    @brief read the power spectra in a hdf5 file
    @param file: the name of the hdf5 file
    @param spec_name: the name of the group in the hdf5 file
    @param (optional) spectra: needed for spin0 and spin2 cross correlation, the arrangement of the spectra
    """
    
    spec=file[spec_name]
    data=np.array(spec['data']).T
    
    l=data[:,0]
    if spectra==None:
        ps=data[:,1]
    else:
        ps={}
        for count,spec in enumerate(spectra):
            ps[spec]=data[:,count+1]

    return l,ps


