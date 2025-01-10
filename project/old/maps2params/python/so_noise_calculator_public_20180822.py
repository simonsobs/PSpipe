from __future__ import print_function
import numpy as np

####################################################################
####################################################################
### LAT CALCULATOR ###
####################################################################
####################################################################
def Simons_Observatory_V3_LA_bands():
    ## returns the band centers in GHz for a CMB spectrum
    ## if your studies require color corrections ask and we can estimate these for you
    return(np.array([27.,39.,93.,145.,225.,280.]))

def Simons_Observatory_V3_LA_beams():
    ## returns the LAT beams in arcminutes
    beam_LAT_27  = 7.4
    beam_LAT_39  = 5.1
    beam_LAT_93  = 2.2
    beam_LAT_145 = 1.4
    beam_LAT_225 = 1.0
    beam_LAT_280 = 0.9
    return(np.array([beam_LAT_27,beam_LAT_39,beam_LAT_93,beam_LAT_145,beam_LAT_225,beam_LAT_280]))

def Simons_Observatory_V3_LA_noise(sensitivity_mode,f_sky,ell_min,ell_max,delta_ell,N_LF=1.,N_MF=4.,N_UHF=2., apply_beam_correction=True, apply_kludge_correction=True):
    ## returns noise curves in both temperature and polarization, including the impact of the beam, for the SO large aperture telescope
    # sensitivity_mode:
    #     1: baseline,
    #     2: goal
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the computation of N(ell)
    # delta_ell: the step size for computing N_ell
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## LARGE APERTURE
    # configuration
    # ensure valid parameter choices
    assert( sensitivity_mode == 1 or sensitivity_mode == 2)
    assert( f_sky > 0. and f_sky <= 1.)
    assert( ell_max <= 2e4 )
    assert( delta_ell >= 1 )
    # ensure total is 7
    if (N_LF + N_MF + N_UHF) != 7:
        print("WARNING! You requested:",N_LF + N_MF + N_UHF, " optics tubes while SO LAT design is for 7")
    NTubes_LF  = N_LF  #default = 1
    NTubes_MF  = N_MF  #default = 4.
    NTubes_UHF = N_UHF #default = 2.
    # sensitivity in uK*sqrt(s)
    # set noise to irrelevantly high value when NTubes=0
    # note that default noise levels are for 1-4-2 tube configuration
    if (NTubes_LF == 0.):
        S_LA_27 = 1.e9*np.ones(3)
        S_LA_39 = 1.e9*np.ones(3)
    else:
        S_LA_27  = np.array([1.e9,48.,35.]) * np.sqrt(1./NTubes_LF)  ## converting these to per tube sensitivities
        S_LA_39  = np.array([1.e9,24.,18.]) * np.sqrt(1./NTubes_LF)
    if (NTubes_MF == 0.):
        S_LA_93 = 1.e9*np.ones(3)
        S_LA_145 = 1.e9*np.ones(3)
    else:
        S_LA_93  = np.array([1.e9,5.4,3.9]) * np.sqrt(4./NTubes_MF)
        S_LA_145 = np.array([1.e9,6.7,4.2]) * np.sqrt(4./NTubes_MF)
    if (NTubes_UHF == 0.):
        S_LA_225 = 1.e9*np.ones(3)
        S_LA_280 = 1.e9*np.ones(3)
    else:
        S_LA_225 = np.array([1.e9,15.,10.]) * np.sqrt(2./NTubes_UHF)
        S_LA_280 = np.array([1.e9,36.,25.]) * np.sqrt(2./NTubes_UHF)
    # 1/f polarization noise -- see Sec. 2.2 of SO science goals paper
    f_knee_pol_LA_27 = 700.
    f_knee_pol_LA_39 = 700.
    f_knee_pol_LA_93 = 700.
    f_knee_pol_LA_145 = 700.
    f_knee_pol_LA_225 = 700.
    f_knee_pol_LA_280 = 700.
    alpha_pol = -1.4
    # atmospheric 1/f temperature noise -- see Sec. 2.2 of SO science goals paper
    C_27  =    200.
    C_39  =     77.
    C_93  =   1800.
    C_145 =  12000.
    C_225 =  68000.
    C_280 = 124000.
    alpha_temp = -3.5

    ####################################################################
    ## calculate the survey area and time
    survey_time = 5. #years
    t = survey_time * 365.25 * 24. * 3600.    ## convert years to seconds
    t = t * 0.2   ## retention after observing efficiency and cuts
    if apply_kludge_correction:
        t = t * 0.85  ## a kludge for the noise non-uniformity of the map edges
    A_SR = 4. * np.pi * f_sky  ## sky areas in steradians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    #print("sky area: ", A_deg, "degrees^2")

    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(ell_min,ell_max,delta_ell)

    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_LA_27[sensitivity_mode]  / np.sqrt(t)
    W_T_39  = S_LA_39[sensitivity_mode]  / np.sqrt(t)
    W_T_93  = S_LA_93[sensitivity_mode]  / np.sqrt(t)
    W_T_145 = S_LA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_LA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_LA_280[sensitivity_mode] / np.sqrt(t)

    ## calculate the map noise level (white) for the survey in uK_arcmin for temperature
    MN_T_27  = W_T_27  * np.sqrt(A_arcmin)
    MN_T_39  = W_T_39  * np.sqrt(A_arcmin)
    MN_T_93  = W_T_93  * np.sqrt(A_arcmin)
    MN_T_145 = W_T_145 * np.sqrt(A_arcmin)
    MN_T_225 = W_T_225 * np.sqrt(A_arcmin)
    MN_T_280 = W_T_280 * np.sqrt(A_arcmin)
    Map_white_noise_levels= np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225,MN_T_280])
    #print("white noise levels (T): ",Map_white_noise_levels ,"[uK-arcmin]")

    ## calculate the atmospheric contribution for T
    ## see Sec. 2.2 of SO science goals paper
    ell_pivot = 1000.
    # handle cases where there are zero tubes of some kind
    if (NTubes_LF == 0.):
        AN_T_27 = 0. #irrelevantly large noise already set above
        AN_T_39 = 0.
    else:
        AN_T_27  = C_27  * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_LF)
        AN_T_39  = C_39  * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_LF)
    if (NTubes_MF == 0.):
        AN_T_93 = 0.
        AN_T_145 = 0.
    else:
        AN_T_93  = C_93  * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_MF)
        AN_T_145 = C_145 * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_MF)
    if (NTubes_UHF == 0.):
        AN_T_225 = 0.
        AN_T_280 = 0.
    else:
        AN_T_225 = C_225 * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_UHF)
        AN_T_280 = C_280 * (ell/ell_pivot)**alpha_temp * A_SR / t / (2.*NTubes_UHF)
    # include cross-frequency correlations in the atmosphere
    # use correlation coefficient of r=0.9 within each dichroic pair and 0 otherwise
    r_atm = 0.9
    AN_T_27x39 = r_atm * np.sqrt(AN_T_27 * AN_T_39)
    AN_T_93x145 = r_atm * np.sqrt(AN_T_93 * AN_T_145)
    AN_T_225x280 = r_atm * np.sqrt(AN_T_225 * AN_T_280)

    ## calculate N(ell)
    N_ell_T_27   = (W_T_27**2. * A_SR) + AN_T_27
    N_ell_T_39   = (W_T_39**2. * A_SR) + AN_T_39
    N_ell_T_93   = (W_T_93**2. * A_SR) + AN_T_93
    N_ell_T_145  = (W_T_145**2. * A_SR) + AN_T_145
    N_ell_T_225  = (W_T_225**2. * A_SR) + AN_T_225
    N_ell_T_280  = (W_T_280**2. * A_SR) + AN_T_280
    # include cross-correlations due to atmospheric noise
    N_ell_T_27x39 = AN_T_27x39
    N_ell_T_93x145 = AN_T_93x145
    N_ell_T_225x280 = AN_T_225x280

    if apply_beam_correction:
        ## include the impact of the beam
        LA_beams = Simons_Observatory_V3_LA_beams() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
        ## LAT beams as a sigma expressed in radians
        N_ell_T_27  *= np.exp( ell*(ell+1)* LA_beams[0]**2. )
        N_ell_T_39  *= np.exp( ell*(ell+1)* LA_beams[1]**2. )
        N_ell_T_93  *= np.exp( ell*(ell+1)* LA_beams[2]**2. )
        N_ell_T_145 *= np.exp( ell*(ell+1)* LA_beams[3]**2. )
        N_ell_T_225 *= np.exp( ell*(ell+1)* LA_beams[4]**2. )
        N_ell_T_280 *= np.exp( ell*(ell+1)* LA_beams[5]**2. )
        N_ell_T_27x39 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[0]**2. + LA_beams[1]**2.) )
        N_ell_T_93x145 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[2]**2. + LA_beams[3]**2.) )
        N_ell_T_225x280 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[4]**2. + LA_beams[5]**2.) )

    ## make an array of noise curves for T
    # include cross-correlations due to atmospheric noise
    N_ell_T_LA = np.array([N_ell_T_27,N_ell_T_39,N_ell_T_93,N_ell_T_145,N_ell_T_225,N_ell_T_280,N_ell_T_27x39,N_ell_T_93x145,N_ell_T_225x280])

    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the atmospheric contribution for P
    AN_P_27  = (ell / f_knee_pol_LA_27 )**alpha_pol + 1.
    AN_P_39  = (ell / f_knee_pol_LA_39 )**alpha_pol + 1.
    AN_P_93  = (ell / f_knee_pol_LA_93 )**alpha_pol + 1.
    AN_P_145 = (ell / f_knee_pol_LA_145)**alpha_pol + 1.
    AN_P_225 = (ell / f_knee_pol_LA_225)**alpha_pol + 1.
    AN_P_280 = (ell / f_knee_pol_LA_280)**alpha_pol + 1.

    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2. * A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2. * A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2. * A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2. * A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2. * A_SR * AN_P_225
    N_ell_P_280  = (W_T_280 * np.sqrt(2))**2. * A_SR * AN_P_280
    # include cross-correlations due to atmospheric noise
    # different approach than for T -- need to subtract off the white noise part to get the purely atmospheric part
    # see Sec. 2.2 of the SO science goals paper
    N_ell_P_27_atm = (W_T_27  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_27 )**alpha_pol
    N_ell_P_39_atm = (W_T_39  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_39 )**alpha_pol
    N_ell_P_93_atm = (W_T_93  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_93 )**alpha_pol
    N_ell_P_145_atm = (W_T_145  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_145 )**alpha_pol
    N_ell_P_225_atm = (W_T_225  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_225 )**alpha_pol
    N_ell_P_280_atm = (W_T_280  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_280 )**alpha_pol
    N_ell_P_27x39 = r_atm * np.sqrt(N_ell_P_27_atm * N_ell_P_39_atm)
    N_ell_P_93x145 = r_atm * np.sqrt(N_ell_P_93_atm * N_ell_P_145_atm)
    N_ell_P_225x280 = r_atm * np.sqrt(N_ell_P_225_atm * N_ell_P_280_atm)

    if apply_beam_correction:
        ## include the impact of the beam
        N_ell_P_27  *= np.exp( ell*(ell+1)* LA_beams[0]**2 )
        N_ell_P_39  *= np.exp( ell*(ell+1)* LA_beams[1]**2 )
        N_ell_P_93  *= np.exp( ell*(ell+1)* LA_beams[2]**2 )
        N_ell_P_145 *= np.exp( ell*(ell+1)* LA_beams[3]**2 )
        N_ell_P_225 *= np.exp( ell*(ell+1)* LA_beams[4]**2 )
        N_ell_P_280 *= np.exp( ell*(ell+1)* LA_beams[5]**2 )
        N_ell_P_27x39 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[0]**2. + LA_beams[1]**2.) )
        N_ell_P_93x145 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[2]**2. + LA_beams[3]**2.) )
        N_ell_P_225x280 *= np.exp( (ell*(ell+1)/2.) * (LA_beams[4]**2. + LA_beams[5]**2.) )

    ## make an array of noise curves for P
    N_ell_P_LA = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_280,N_ell_P_27x39,N_ell_P_93x145,N_ell_P_225x280])

    ####################################################################
    return(ell, N_ell_T_LA, N_ell_P_LA, Map_white_noise_levels)

####################################################################
####################################################################
### SAT CALCULATOR ###
####################################################################
####################################################################
def Simons_Observatory_V3_SA_bands():
    ## returns the band centers in GHz for a CMB spectrum
    ## if your studies require color corrections ask and we can estimate these for you
    return(np.array([27.,39.,93.,145.,225.,280.]))

def Simons_Observatory_V3_SA_beams():
    ## returns the SAT beams in arcminutes
    beam_SAT_27 = 91.
    beam_SAT_39 = 63.
    beam_SAT_93 = 30.
    beam_SAT_145 = 17.
    beam_SAT_225 = 11.
    beam_SAT_280 = 9.
    return(np.array([beam_SAT_27,beam_SAT_39,beam_SAT_93,beam_SAT_145,beam_SAT_225,beam_SAT_280]))

def Simons_Observatory_V3_SA_noise(sensitivity_mode,one_over_f_mode,SAT_yrs_LF,f_sky,ell_min,ell_max,delta_ell,apply_beam_correction=True, apply_kludge_correction=True):
    ## returns noise curves in polarization only, including the impact of the beam, for the SO small aperture telescopes
    ## noise curves are polarization only
    # sensitivity_mode
    #     1: baseline,
    #     2: goal
    # one_over_f_mode
    #     0: pessimistic
    #     1: optimistic
    # SAT_yrs_LF: 0,1,2,3,4,5:  number of years where an LF is deployed on SAT
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the computation of N(ell)
    # delta_ell: the step size for computing N_ell
    ####################################################################
    ####################################################################
    ###                        Internal variables
    ## SMALL APERTURE
    # ensure valid parameter choices
    assert( sensitivity_mode == 1 or sensitivity_mode == 2)
    assert( one_over_f_mode == 0 or one_over_f_mode == 1)
    assert( SAT_yrs_LF <= 5) #N.B. SAT_yrs_LF can be negative
    assert( f_sky > 0. and f_sky <= 1.)
    assert( ell_max <= 2e4 )
    assert( delta_ell >= 1 )
    # configuration
    if (SAT_yrs_LF > 0):
        NTubes_LF  = SAT_yrs_LF/5. + 1e-6  ## regularized in case zero years is called
        NTubes_MF  = 2 - SAT_yrs_LF/5.
    else:
        NTubes_LF  = np.fabs(SAT_yrs_LF)/5. + 1e-6  ## regularized in case zero years is called
        NTubes_MF  = 2
    NTubes_UHF = 1.
    # sensitivity
    # N.B. divide-by-zero will occur if NTubes = 0
    # handle with assert() since it's highly unlikely we want any configurations without >= 1 of each tube type
    assert( NTubes_LF > 0. )
    assert( NTubes_MF > 0. )
    assert( NTubes_UHF > 0.)
    S_SA_27  = np.array([1.e9,21,15])    * np.sqrt(1./NTubes_LF)
    S_SA_39  = np.array([1.e9,13,10])    * np.sqrt(1./NTubes_LF)
    S_SA_93  = np.array([1.e9,3.4,2.4]) * np.sqrt(2./(NTubes_MF))
    S_SA_145 = np.array([1.e9,4.3,2.7]) * np.sqrt(2./(NTubes_MF))
    S_SA_225 = np.array([1.e9,8.6,5.7])  * np.sqrt(1./NTubes_UHF)
    S_SA_280 = np.array([1.e9,22,14])    * np.sqrt(1./NTubes_UHF)
    # 1/f polarization noise
    # see Sec. 2.2 of the SO science goals paper
    f_knee_pol_SA_27  = np.array([30.,15.])
    f_knee_pol_SA_39  = np.array([30.,15.])  ## from QUIET
    f_knee_pol_SA_93  = np.array([50.,25.])
    f_knee_pol_SA_145 = np.array([50.,25.])  ## from ABS, improvement possible by scanning faster
    f_knee_pol_SA_225 = np.array([70.,35.])
    f_knee_pol_SA_280 = np.array([100.,40.])
    alpha_pol =np.array([-2.4,-2.4,-2.5,-3,-3,-3])

    ####################################################################
    ## calculate the survey area and time
    t = 5* 365. * 24. * 3600    ## five years in seconds
    t = t * 0.2  ## retention after observing efficiency and cuts
    if apply_kludge_correction:
        t = t* 0.85  ## a kludge for the noise non-uniformity of the map edges
    A_SR = 4 * np.pi * f_sky  ## sky area in steradians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    A_arcmin = A_deg * 3600.
    #print("sky area: ", A_deg, "degrees^2")
    #print("Note that this code includes a factor of 1/0.85 increase in the noise power, corresponding to assumed mode loss due to map depth non-uniformity.")
    #print("If you have your own N_hits map that already includes such non-uniformity, you should increase the total integration time by a factor of 1/0.85 when generating noise realizations from the power spectra produced by this code, so that this factor is not mistakenly introduced twice.")

    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(ell_min,ell_max,delta_ell)

    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_SA_27[sensitivity_mode]  / np.sqrt(t)
    W_T_39  = S_SA_39[sensitivity_mode]  / np.sqrt(t)
    W_T_93  = S_SA_93[sensitivity_mode]  / np.sqrt(t)
    W_T_145 = S_SA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_SA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_SA_280[sensitivity_mode] / np.sqrt(t)

    ## calculate the map noise level (white) for the survey in uK_arcmin for temperature
    MN_T_27  = W_T_27  * np.sqrt(A_arcmin)
    MN_T_39  = W_T_39  * np.sqrt(A_arcmin)
    MN_T_93  = W_T_93  * np.sqrt(A_arcmin)
    MN_T_145 = W_T_145 * np.sqrt(A_arcmin)
    MN_T_225 = W_T_225 * np.sqrt(A_arcmin)
    MN_T_280 = W_T_280 * np.sqrt(A_arcmin)
    Map_white_noise_levels = np.array([MN_T_27,MN_T_39,MN_T_93,MN_T_145,MN_T_225,MN_T_280])
    #print("white noise levels (T): ",Map_white_noise_levels ,"[uK-arcmin]")

    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the atmospheric contribution for P
    ## see Sec. 2.2 of the SO science goals paper
    AN_P_27  = (ell / f_knee_pol_SA_27[one_over_f_mode] )**alpha_pol[0] + 1.
    AN_P_39  = (ell / f_knee_pol_SA_39[one_over_f_mode] )**alpha_pol[1] + 1.
    AN_P_93  = (ell / f_knee_pol_SA_93[one_over_f_mode] )**alpha_pol[2] + 1.
    AN_P_145 = (ell / f_knee_pol_SA_145[one_over_f_mode])**alpha_pol[3] + 1.
    AN_P_225 = (ell / f_knee_pol_SA_225[one_over_f_mode])**alpha_pol[4] + 1.
    AN_P_280 = (ell / f_knee_pol_SA_280[one_over_f_mode])**alpha_pol[5] + 1.

    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2.* A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2.* A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2.* A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2.* A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2.* A_SR * AN_P_225
    N_ell_P_280  = (W_T_280 * np.sqrt(2))**2.* A_SR * AN_P_280

    if apply_beam_correction:
        ## include the impact of the beam
        SA_beams = Simons_Observatory_V3_SA_beams() / np.sqrt(8. * np.log(2)) /60. * np.pi/180.
        ## SAT beams as a sigma expressed in radians
        N_ell_P_27  *= np.exp( ell*(ell+1)* SA_beams[0]**2. )
        N_ell_P_39  *= np.exp( ell*(ell+1)* SA_beams[1]**2. )
        N_ell_P_93  *= np.exp( ell*(ell+1)* SA_beams[2]**2. )
        N_ell_P_145 *= np.exp( ell*(ell+1)* SA_beams[3]**2. )
        N_ell_P_225 *= np.exp( ell*(ell+1)* SA_beams[4]**2. )
        N_ell_P_280 *= np.exp( ell*(ell+1)* SA_beams[5]**2. )

    ## make an array of noise curves for P
    N_ell_P_SA = np.array([N_ell_P_27,N_ell_P_39,N_ell_P_93,N_ell_P_145,N_ell_P_225,N_ell_P_280])

    ####################################################################
    return(ell,N_ell_P_SA,Map_white_noise_levels)
