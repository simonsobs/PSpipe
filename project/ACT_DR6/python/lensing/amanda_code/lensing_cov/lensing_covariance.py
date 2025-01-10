import os
from copy import deepcopy
import numpy as np

def load_derivs_wrt_lensing(root, cmb_spectra=['tt', 'te', 'ee', 'bb'], verbose=True):
    """
    Returns a dictionary with 2d arrays holding the derivatives of the 
    lensed CMB spectra with respect to the lensing potential spectrum.

    Parameters
    ----------
    root : str
        A string containing the `root` you passed to CLASS delens, and the 
        two-digit "version" number that CLASS adds to the `root`. For example, 
        if your CLASS ini file specifies `root = /home/output/test_`, and 
        you've ran CLASS a single time, you would pass
        `class_delens_root = '/home/output/test_00'`; if you ran CLASS a 
        second with the same `root`, the two-digit "version" will increase 
        to `01`, etc.
    cmb_spectra : list of str, default=['tt', 'te', 'ee', 'bb']
        A list of the kinds of CMB power spectra that will be used in the 
        covariance matrix calculation.
    verbose : bool, default=True
        Whether to print out which derivatives are being loaded.

    Returns
    -------
    dcmb_dcpp : dict of array of float
        A dictionary with keys corresponding to the elements of `cmb_spectra`,
        holding the 2d array of the derivative with respect to the lensing 
        potential spectrum, starting from ell = 0.
    """
    dcmb_dcpp = {}
    for s in cmb_spectra:
        if verbose: 
            print(f'loading derivative of lensed {s.upper()} with respect to the lensing potential')
        deriv = np.loadtxt(f'{root}_dCl{s.upper()}dCldd_lensed.dat')
        # the CLASS output starts at ell = 2; we start at ell = 0:
        dcmb_dcpp[s] = np.zeros((deriv.shape[0]+2, deriv.shape[1]+2))
        dcmb_dcpp[s][2:,2:] = deriv.copy()
        # CLASS outputs the derivatives with respect to C_L^dd = L(L+1) C_L^phiphi;
        # we want them to be with respect to just C_L^phiphi:
        deriv_ells = np.arange(deriv.shape[1]+2)
        for i, ell in enumerate(deriv_ells):
            if i > 1: # leave as zero for L = 0, 1
                dcmb_dcpp[s][:, i] *= (ell * (ell + 1))
    return dcmb_dcpp


def load_derivs_wrt_unlensed(root, cmb_spectra=['tt', 'te', 'ee', 'bb'], verbose=True):
    """
    Returns a dictionary with 2d arrays holding the derivatives of the 
    lensed CMB spectra with respect to the unlensed CMB spectra.

    Parameters
    ----------
    root : str
        A string containing the `root` you passed to CLASS delens, and the 
        two-digit "version" number that CLASS adds to the `root`. For example, 
        if your CLASS ini file specifies `root = /home/output/test_`, and 
        you've ran CLASS a single time, you would pass
        `class_delens_root = '/home/output/test_00'`; if you ran CLASS a 
        second with the same `root`, the two-digit "version" will increase 
        to `01`, etc.
    cmb_spectra : list of str, default=['tt', 'te', 'ee', 'bb']
        A list of the kinds of CMB power spectra that will be used in the 
        covariance matrix calculation.
    verbose : bool, default=True
        Whether to print out which derivatives are being loaded.

    Returns
    -------
    dlens_dunlens : dict of array of float
        A dictionary with keys corresponding to the elements of `cmb_spectra`,
        holding the 2d array of the derivative of the lensed CMB spectrum with 
        respect to the unlensed CMB spectrum, starting from ell = 0.
    """
    dlens_dunlens = {}
    msg = 'loading derivative of lensed {} with respect to unlensed {}'
    for s in cmb_spectra:
        if s == 'bb':
            if verbose:
                print(msg.format(s.upper(), 'EE'))
            deriv = np.loadtxt(f'{root}_dCl{s.upper()}dClEE_lensed.dat')
        else:
            if verbose:
                print(msg.format(s.upper(), s.upper()))
            deriv = np.loadtxt(f'{root}_dCl{s.upper()}dCl{s.upper()}_lensed.dat')
        # the CLASS output starts at ell = 2; we start at ell = 0:
        dlens_dunlens[s] = np.zeros((deriv.shape[0]+2, deriv.shape[1]+2))
        dlens_dunlens[s][2:,2:] = deriv.copy()
    return dlens_dunlens


def diag_noiseless_covariance(theo, lmax, spectra=['tt', 'te', 'ee', 'bb', 'kk']):
    """
    Returns a nested dictionary holding the diagonal elements of the covariance
    between the kinds of `spectra` (i.e. the cosmic variance part of the covariance).

    Parameters
    ----------
    theo : dict of array of float
        A dictionary holding the theory CMB spectra and lensing spectrum.
        The CMB spectra should be in units of uK^2 as C_l's (no ell-factors), 
        and have keys `'tt'`, `'te'`, `'ee'`, `'bb'`. The key for the CMB 
        lensing spectrum should be either `'pp'` (phi), `'kk'` (kappa),
        or `'dd'` (deflection). All spectra are expected to start at ell = 0
        and extend to at least ell = `lmax`.
    lmax : int
        The maximum multipole to use.
    spectra : list of str, default=['tt', 'te', 'ee', 'bb', 'kk']
        The list of spectra to use.

    Returns
    -------
    diags : nested dict of array of float
        A nested dictionary holding the diagonals of the covariance between the
        different spectra. For example, `diags['tt']['ee']` is a 1d array, with
        `lmax+1` elements, containing the covariance between C_l^TT and C_l^EE.

    Raises
    ------
    ValueError 
        If there is an unrecognized name in the `spectra` list.
    """
    ells = np.arange(lmax + 1)
    lfact = 1 / (2 * ells + 1)
    lfact[:2] = 1 # leaves elements for ell = 0, 1 as zero
    # create the nested dictionary and initialize everything to zero
    diags = {s1: {s2: np.zeros(len(ells)) for s2 in spectra} for s1 in spectra}
    # calculate diagonals of the diagonal covmat blocks (XY x XY):
    for s in spectra:
        # determine if X = Y:
        s1 = s[0]
        s2 = s[1]
        if s1 == s2:
            diags[s][s] = 2 * theo[s][:lmax+1]**2 * lfact
        elif s == 'te':
            diags[s][s] = (theo['te']**2 + theo['tt'] * theo['ee'])[:lmax+1] * lfact
        else:
            raise ValueError(f"Unknown element '{s}' in the list of spectra.")
    # calculate the non-zero diagonals of the off-diagonal blocks (XY x WZ):
    if ('tt' in spectra) and ('ee' in spectra):
        diags['tt']['ee'] = 2 * theo['te'][:lmax+1]**2 * lfact
        diags['ee']['tt'] = diags['tt']['ee']
    if ('tt' in spectra) and ('te' in spectra):
        diags['tt']['te'] = 2 * theo['tt'][:lmax+1] * theo['te'][:lmax+1] * lfact
        diags['te']['tt'] = diags['tt']['te']
    if ('ee' in spectra) and ('te' in spectra):
        diags['ee']['te'] = 2 * theo['ee'][:lmax+1] * theo['te'][:lmax+1] * lfact
        diags['te']['ee'] = diags['ee']['te']
    return diags


def phixCMB_to_kappaxCMB(cov_block, phi_ells):
    """
    Returns the covariance matrix for C_L^kk x C_l, where 
    C_L^kk = [L(L+1)}^2 C_L^phiphi / 4 and C_l is a CMB spectrum, given
    a covariance matrix for C_L^phiphi x C_l.

    Parameters
    ----------
    cov_block : array of float
        The 2d covariance matrix for C_L^phiphi, with each row corresponding 
        to a lensing multipole L.
    phi_ells : array of int
        A 1d array holding the lensing multipoles corresponding to the rows
        of the `cov_block`.

    Returns
    -------
    cov_block : array of float
        The same (i.e. NOT a copy) 2d covariance matrix but now for C_L^kk.
    """
    for i, L in enumerate(phi_ells):
        cov_block[i] = cov_block[i] * (L * (L + 1))**2 / 4
    return cov_block


class LensingCovariance:
    def __init__(self, theo, ell_ranges, class_delens_root, 
            cmb_spectra=['tt', 'te', 'ee', 'bb'], fsky=1.0, 
            use_derivs_wrt_lensing=True, use_derivs_wrt_unlensed=True,
            calc_lensingxcmb=True, cmb_bmat=None, lens_bmat=None, 
            bmat_lmin=2, bin_ranges=None, verbose=True, output_root=None):
        """
        Initialization to calculate the analytic lensing-induced terms of the covariance matrix.

        Parameters
        ----------
        theo : dict of array of float
            A dictionary holding the ** UNLENSED ** CMB theory spectra and 
            CMB lensing potential (phi) or convergence (kappa) spectrum. 
            The CMB spectra should be in units of uK^2 as C_l's (no ell-factors), 
            and have keys `'tt'`, `'te'`, `'ee'`, `'bb'`. The CMB lensing 
            potential spectrum C_L^phiphi should have a key `'pp'`, and/or you 
            can pass the CMB lensing convergence spectrum with a key `'kk'` 
            using the convention C_L^kk = [L(L+1)]^2 C_L^phiphi / 4. All spectra 
            are expected to start at ell = 0 and end at the same multipole.
        ell_ranges : dict of list of int
            A dictionary with keys `'tt'`, `'te'`, ..., `'kk'` holding a 
            list `[lmin, lmax]` of the multipole limits for each spectrum
            that will be used (i.e., the keys should be the same as the 
            elements in `cmb_spectra`, plus a key for `'kk'`). For example,
            `ell_ranges = {'tt': [500, 8000], 'kk': [40, 3000]}`.
        class_delens_root : str
            A string containing the `root` you passed to CLASS delens, and 
            the two-digit "version" number that CLASS adds to the `root`.
            For example, if your CLASS ini file sets `root = /home/output/test_`,
            and you've run CLASS a single time, you would pass
            `class_delens_root = '/home/output/test_00'`; if you ran CLASS 
            a second with the same `root`, the two-digit "version" will 
            increase to `01`, etc.
        cmb_spectra : list of str, default=['tt', 'te', 'ee', 'bb']
            A list of the kinds of CMB spectra to include in the calculation.
            The full list of options is used by default.
        fsky : float, default = 1.0
            The sky fraction. The covariance matrix blocks are multiplied by `1 / fsky`.
        use_derivs_wrt_lensing : bool, default=True
            Whether to include the term involving the derivatives of the CMB 
            spectra with respect to the lensing potential spectrum when 
            calculating the covariance between CMB spectra. (See the third line 
            of eq. 5 in arXiv:2111.15036).
        use_derivs_wrt_unlensed : bool, default=True
            Whether to include the term involving the derivatives of the lensed 
            CMB spectra with respect to the unlensed spectrum when calculating 
            the covariance between CMB spectra. (See the second line 
            of eq. 5 in arXiv:2111.15036).
        calc_lensingxcmb : bool, default=True
            Whether to calculate the covariance between the CMB spectra and the
            lensing convergence spectrum. This also requires the derivatives
            of the CMB spectra with respect to the lensing potential spectrum.
        cmb_bmat, lens_bmat : array of float, default=None
            Binning matrices with shape `(num_bins, num_ells)` for the CMB 
            and lensing spectra, respectively. If these are passed, you may
            request the covariance matrix blocks to be binned before they are
            returned. In this case you should also pass a `bmat_lmin` value
            (see below), and the `bin_ranges` if you want the binned matrices
            to be trimmed.
        bmat_lmin : int, default=2
            The minimum multipole in the binning matrix. For example, if
            `ells` is an array of multipoles beginning at zero, we multiply
            the binning matrix by `ells[bmat_lmin:]` to get the binned multipoles.
            Note that we assume the maximum multipole of the binning matrices
            is the maximum passed in the `ell_ranges` dictionary.
        bin_ranges : dict of list of int, default=None
            A dictionary with the same structure as the `ell_ranges`, providing
            the miminum and maximum bin number for the different kinds of 
            spectra as a list `[bmin, bmax]`. Note that we keep bins up to and 
            inckuding `bmax`. You may exclude keys for spectra 
            that do not need to be trimmed. You may also, for example, pass 
            `[bmin, None]` to only remove bins below `bmin`. If `bin_ranges=None`, 
            the full binned matrices are returned when binning is requested. 
            For example, if `bin_ranges = {'tt': [3, None]}`, we remove the
            first three bins for TT.
        verbose : bool, default=True
            Whether to print out some messages about the progress of the calculation.
        output_root : str, default=None
            If provided, any saved files will begin with the `output_root`.
            This should include the path and/or file name root, e.g.
            `output_root = 'output/test'`. If not provided, any saved files
            will be saved to the current directory. 
            NOTE that any existing files with the same `output_root` will be
            overwritten if you pass `save=True` when calculating the
            covariance matrix blocks.
        """
        self.verbose = verbose
        self.theo = deepcopy(theo)
        self.fsky = fsky
        self.cmb_spectra = cmb_spectra
        self.spectra = self.cmb_spectra + ['kk']
        
        self.ell_ranges = ell_ranges
        # lensing multipole range:
        if 'kk' in ell_ranges.keys():
            self.Lmin = self.ell_ranges['kk'][0]
            self.Lmax = self.ell_ranges['kk'][1] 
        elif 'pp' in ell_ranges.keys():
            self.Lmin = self.ell_ranges['pp'][0]
            self.Lmax = self.ell_ranges['pp'][1]
        else:
            errmsg = "You must pass the multipole ranges for the CMB lensing power spectrum in the `ell_ranges` dictionary with a key `'kk'` or `'pp'`."
            raise ValueError(errmsg)
        # use the maximum multipole in `ell_ranges` as an overall lmax value
        self.lmax = max([self.ell_ranges[s][1] for s in self.spectra])
        self.ells = np.arange(self.lmax + 1)

        self.cmb_bmat = cmb_bmat
        self.lens_bmat = lens_bmat
        self.bmat_lmin = bmat_lmin
        if bin_ranges is None:
            self.bin_ranges = {} # empty dict so we can still check for keys
        else:
            self.bin_ranges = bin_ranges

        self.class_delens_root = class_delens_root
        # which terms to include for CMB x CMB blocks:
        self.use_derivs_wrt_lensing = use_derivs_wrt_lensing
        self.use_derivs_wrt_unlensed = use_derivs_wrt_unlensed
        # whether to calculate the CMB x lensing blocks:
        self.calc_lensingxcmb = calc_lensingxcmb

        self.output_root = ''
        if output_root is not None:
            if not os.path.isdir(output_root):
                self.output_root = f'{output_root}_'
            else:
                self.output_root = output_root

        # load in the derivatives from CLASS delens:
        self.dcmb_dcpp = None
        self.dlens_dunlens = None
        if self.use_derivs_wrt_lensing or self.calc_lensingxcmb:
            self.get_dcmb_dcpp()
        if self.use_derivs_wrt_unlensed:
            self.get_dlens_dunlens()
        
        # we need the (noiseless) cosmic variance for C_L^phiphi, 
        #  and for the unlensed CMB if we are using the derivatives of the 
        #  lensed with respect to unlensed CMB spectra:
        if 'pp' not in self.theo.keys():
            if 'kk' not in self.theo.keys():
                errmsg = "Cannot find the theoretical CMB lensing spectrum in the `theo` dictionary. You must pass either C_L^phiphi with a key 'pp', or C_L^kappakappa with a key 'kk'."
                raise ValueError(errmsg)
            else:
                self.theo['pp'] = np.zeros(self.lmax + 1)
                Lfact = (self.ells * (self.ells + 1))**2 / 4
                self.theo['pp'][2:] = self.theo['kk'][2:] / Lfact[2:]
        ucov_spectra = self.cmb_spectra + ['pp']
        self.ucov_diags = diag_noiseless_covariance(self.theo, self.lmax, spectra=ucov_spectra)
        self.phi_cov = np.diag(self.ucov_diags['pp']['pp'][self.Lmin:self.Lmax+1])



    def get_dcmb_dcpp(self):
        """Ensures that we have the derivatives of the lensed CMB with respect 
        to the lensing potential."""
        if self.dcmb_dcpp is None:
            self.dcmb_dcpp = load_derivs_wrt_lensing(self.class_delens_root, cmb_spectra=self.cmb_spectra, verbose=self.verbose)


    def get_dlens_dunlens(self):
        """Ensures that we have the derivatives of the lensed CMB with respect 
        to the unlensed CMB."""
        if self.dlens_dunlens is None:
            self.dlens_dunlens = load_derivs_wrt_unlensed(self.class_delens_root, cmb_spectra=self.cmb_spectra, verbose=self.verbose)


    def bin_block(self, cov_block, xy, wz):
        """
        Bin the covariance between C_l^XY and C_l^WZ.

        Parameters
        ----------
        cov_block : array of float
            The two-dimensional covariance, starting at ell = 0.
        xy, wz : str
            The two kinds of spectra used. Each must be one of `'tt'`, 
            `'te'`, `'ee'`, `'bb'`, `'kk'`.

        Returns
        -------
        binned_block : array of float
            The binned covariance matrix block.

        Raises
        ------
        ValueError
            If the `cmb_bmat` (or `lens_bmat` when `xy` or `wz` is `'kk'`) was
            not passed when initializing the `LensingCovariance` class.
        """
        # make sure we have what we need:
        if self.cmb_bmat is None:
            errmsg = f"Unable to bin {xy} x {wz} without a binning matrix. You must pass the CMB binning matrix as `cmb_bmat` when initializing the `LensingCovariance` class."
            raise ValueError(errmsg)
        if ((xy == 'kk') or (wz == 'kk')) and (self.lens_bmat is None):
            errmsg = f"Unable to bin {xy} x {wz} without a binning matrix. You must pass the C_L^kk binning matrix as `lens_bmat` when initializing the `LensingCovariance` class."
            raise ValueError(errmsg)
        # get the binning matrices:
        bmat1 = self.lens_bmat if (xy == 'kk') else self.cmb_bmat
        bmat2 = self.lens_bmat if (wz == 'kk') else self.cmb_bmat
        # get the maximum multipole of each binning matrix and each dimension of the `cov_block`:
        bmat1_lmax = bmat1.shape[1] + self.bmat_lmin - 1
        bmat2_lmax = bmat2.shape[1] + self.bmat_lmin - 1
        block_lmax1 = cov_block.shape[0] - 1
        block_lmax2 = cov_block.shape[1] - 1
        # find the indices for the binning matrices and cov block, to make their shapes compatible:
        lmax1 = min([block_lmax1, bmat1_lmax])
        lmax2 = min([block_lmax2, bmat2_lmax])
        imax1 = lmax1 + 1 - self.bmat_lmin
        imax2 = lmax2 + 1 - self.bmat_lmin
        binned_block = bmat1[:,:imax1] @ cov_block[self.bmat_lmin:lmax1+1, self.bmat_lmin:lmax2+1] @ bmat2[:,:imax2].T
        # trim the binned block:
        if xy in self.bin_ranges.keys():
            if self.bin_ranges[xy] is not None:
                bmin = self.bin_ranges[xy][0]
                bmax = self.bin_ranges[xy][1]
                if bmin is None:
                    bmin = 0
                if bmax is None:
                    binned_block = binned_block[bmin:, :]
                else:
                    binned_block = binned_block[bmin:bmax+1, :]
        if wz in self.bin_ranges.keys():
            if self.bin_ranges[wz] is not None:
                bmin = self.bin_ranges[wz][0]
                bmax = self.bin_ranges[wz][1]
                if bmin is None:
                    bmin = 0
                if bmax is None:
                    binned_block = binned_block[:, bmin:]
                else:
                    binned_block = binned_block[:, bmin:bmax+1]
        return binned_block


    def cmbxcmb_block_from_derivs_wrt_lensing(self, xy, wz, binned=False, save=False):
        """
        Returns the term of the covariance between lensed CMB spectra 
        C_l1^XY and C_l2^WZ given by the third line of eq. 5 in arXiv:2111.15036,
        fsky^-1 * sum_L [d(C_l1^XY)/d(C_L^pp)] * [2(C_L^pp)^2/(2L+1)] * [d(C_l2^WZ)/d(C_L^pp)],
        where C_L^pp is the lensing potential power spectrum.

        Parameters
        ----------
        xy, wz : str
            The two kinds of lensed CMB spectra to use. Each must be one of
            `'tt'`, `'te'`, `'ee'`, `'bb'`.
        binned : bool, default=False
            Whether to bin the block before returning it.
        save: bool, default=False
            Whether to save the block. If `binned=True`, the binned block
            is saved as a `.txt` file. Otherwise, the unbinned block is 
            saved as a `.npy` file.

        Returns
        -------
        block : array of float
            The two-dimensional covariance between the two CMB spectra, 
            scaled by the sky fraction `fsky`. 
            If `binned = False`, the matrix is returned unbinned, starting 
            from ell = 0, with shape (lmax_XY+1, lmax_WZ+1), where lmax_XY 
            and lmax_WZ are the maximum multipoles for the XY and WZ spectra 
            that were passed in the `ell_ranges` dictionary.
            If `binned = True`, the binned matrix is returned with shape
            (nbin_XY, nbin_WZ). The number of bins for XY and WZ is 
            determined by the shape of the binning matrix, and on the
            `bin_ranges` if they were provided.
        """
        if self.verbose: print(f'calculating {xy} x {wz} from derivatives w/ respect to lensing potential')
        self.get_dcmb_dcpp() # make sure we have the derivatives
        lmin1 = self.ell_ranges[xy][0]
        lmax1 = self.ell_ranges[xy][1]
        derivs1 = self.dcmb_dcpp[xy][lmin1:lmax1+1, self.Lmin:self.Lmax+1]
        lmin2 = self.ell_ranges[wz][0]
        lmax2 = self.ell_ranges[wz][1]
        derivs2 = self.dcmb_dcpp[wz][lmin2:lmax2+1, self.Lmin:self.Lmax+1]
        block = np.zeros((lmax1+1, lmax2+1))
        block[lmin1:lmax1+1, lmin2:lmax2+1] += derivs1 @ self.phi_cov @ derivs2.T
        block /= self.fsky
        if binned:
            block = self.bin_block(block, xy, wz)
        if save:
            if binned:
                fname = f'{self.output_root}binned_{xy}x{wz}_phi_derivs.txt'
                if self.verbose: print(f'saving to {fname}')
                np.savetxt(fname, block)
            else:
                fname = f'{self.output_root}unbinned_{xy}x{wz}_phi_derivs.npy'
                if self.verbose: print(f'saving to {fname}')
                np.save(fname, block)
        return block 


    def cmbxcmb_block_from_derivs_wrt_unlensed(self, xy, wz, binned=False, save=False):
        """
        Returns the term of the covariance between lensed CMB spectra 
        C_l1^XY and C_l2^WZ given by the second line of eq. 5 in arXiv:2111.15036,
        fsky^-1 * sum_ell deriv_{l1,ell}^XY * ucov_ell(XY,WZ) * deriv_{l2,ell}^WZ,
        for l1 != l2, where deriv_{l,ell} = d(C_l^lensed)/d(C_ell^unlensed) 
        for XY or WZ, and ucov_ell(XY, WZ) is the diagonal covariance matrix 
        for the unlensed, noiseless CMB theory.

        Parameters
        ----------
        xy, wz : str
            The two kinds of lensed CMB spectra to use. Each must be one of
            `'tt'`, `'te'`, `'ee'`, `'bb'`.
        binned : bool, default=False
            Whether to bin the block before returning it.
        save: bool, default=False
            Whether to save the block. If `binned=True`, the binned block
            is saved as a `.txt` file. Otherwise, the unbinned block is 
            saved as a `.npy` file.

        Returns
        -------
        block : array of float
            The two-dimensional covariance between the two CMB spectra, 
            scaled by the sky fraction `fsky`. 
            If `binned = False`, the matrix is returned unbinned, starting 
            from ell = 0, with shape (lmax_XY+1, lmax_WZ+1), where lmax_XY 
            and lmax_WZ are the maximum multipoles for the XY and WZ spectra 
            that were passed in the `ell_ranges` dictionary.
            If `binned = True`, the binned matrix is returned with shape
            (nbin_XY, nbin_WZ). The number of bins for XY and WZ is 
            determined by the shape of the binning matrix, and on the
            `bin_ranges` if they were provided.
        """
        if self.verbose: print(f'calculating {xy} x {wz} from derivatives w/ respect to unlensed CMB')
        self.get_dlens_dunlens() # make sure we have the derivatives
        lmin1 = self.ell_ranges[xy][0]
        lmax1 = self.ell_ranges[xy][1]
        derivs1 = self.dlens_dunlens[xy][lmin1:lmax1+1, lmin1:lmax1+1]
        lmin2 = self.ell_ranges[wz][0]
        lmax2 = self.ell_ranges[wz][1]
        derivs2 = self.dlens_dunlens[wz][lmin2:lmax2+1, lmin2:lmax2+1]
        # get the unlensed cosmic variance part:
        ucov_s1 = 'ee' if (xy == 'bb') else xy
        ucov_s2 = 'ee' if (wz == 'bb') else wz
        ucov = np.diag(self.ucov_diags[ucov_s1][ucov_s2])[lmin1:lmax1+1, lmin2:lmax2+1]
        # calculate the term and keep only the off-diagonals:
        block = np.zeros((lmax1+1, lmax2+1))
        block[lmin1:lmax1+1, lmin2:lmax2+1] = derivs1 @ ucov @ derivs2.T
        np.fill_diagonal(block, 0)
        block /= self.fsky
        if binned:
            block = self.bin_block(block, xy, wz)
        if save:
            if binned:
                fname = f'{self.output_root}binned_{xy}x{wz}_unlens_derivs.txt'
                if self.verbose: print(f'saving to {fname}')
                np.savetxt(fname, block)
            else:
                fname = f'{self.output_root}unbinned_{xy}x{wz}_unlens_derivs.npy'
                if self.verbose: print(f'saving to {fname}')
                np.save(fname, block)
        return block 


    def lensingxcmb_block(self, xy, binned=False, save=False):
        """
        Returns the the covariance between the lensing convergence spectrum
        C_L^kk and the lensed CMB spectrum C_ell^XY. 

        Parameters
        ----------
        xy : str
            The kind of lensed CMB spectra to use. Must be one of `'tt'`, 
            `'te'`, `'ee'`, `'bb'`.
        binned : bool, default=False
            Whether to bin the block before returning it.
        save: bool, default=False
            Whether to save the block. If `binned=True`, the binned block
            is saved as a `.txt` file. Otherwise, the unbinned block is 
            saved as a `.npy` file.

        Returns
        -------
        block : array of float
            The two-dimensional covariance between C_L^kk and C_ell^XY
            scaled by the sky fraction `fsky`. 
            If `binned = False`, the matrix is returned unbinned, starting 
            from L, ell = 0, with shape (Lmax+1, lmax+1), where Lmax and 
            lmax are the maximum multipoles for C_L^kk and C_ell^XY that 
            were passed in the `ell_ranges` dictionary.
            If `binned = True`, the binned matrix is returned with shape
            (nbin_kk, nbin_XY). The number of bins for kk and XY is 
            determined by the shape of the binning matrices, and on the
            `bin_ranges` if they were provided.
        """
        if self.verbose: print(f'calculating kk x {xy} from derivatives w/ respect to lensing potential')
        self.get_dcmb_dcpp() # make sure we have the derivatives
        lmin = self.ell_ranges[xy][0]
        lmax = self.ell_ranges[xy][1]
        derivs = self.dcmb_dcpp[xy][lmin:lmax+1, self.Lmin:self.Lmax+1]
        phi_deriv = np.eye(self.Lmax+1)[self.Lmin:self.Lmax+1, self.Lmin:self.Lmax+1]
        phixcmb_block = phi_deriv @ self.phi_cov @ derivs.T
        kkxcmb_block = phixCMB_to_kappaxCMB(phixcmb_block, self.ells[self.Lmin:self.Lmax+1])
        block = np.zeros((self.Lmax+1, lmax+1))
        block[self.Lmin:self.Lmax+1, lmin:lmax+1] = kkxcmb_block
        block /= self.fsky
        if binned:
            block = self.bin_block(block, 'kk', xy)
        if save:
            if binned:
                fname = f'{self.output_root}binned_kkx{xy}.txt'
                if self.verbose: print(f'saving to {fname}')
                np.savetxt(fname, block)
            else:
                fname = f'{self.output_root}unbinned_kkx{xy}.npy'
                if self.verbose: print(f'saving to {fname}')
                np.save(fname, block)
        return block 



    def cov_blocks(self, binned=False, save=False):
        """
        Returns a nested dictionary containing the requested blocks of the 
        covariance matrix. The sum of the two terms of the CMB x CMB blocks
        is returned if both were requested.

        Parameters
        ----------
        binned : bool, default=False
            Whether to bin the blocks before returning them.
        save: bool, default=False
            Whether to save the blocks. If `binned=True`, the binned blocks
            are saved as `.txt` files. Otherwise, the unbinned blocks are 
            saved as `.npy` files. In both cases, only the diagonal blocks
            and upper half of the off-diagonal blocks are saved. 

        Returns
        -------
        blocks : dict of dict of array of float
            A nested dictionary holding the blocks of the covariance matrix.
            Both sets of keys are the elements of `cmb_spectra` (e.g. `'tt'`),
            plus a key `'kk'` for the lensing x CMB blocks if they were 
            requested. If `binned = False`, all blocks begin at ell = 0. 
            For example, `blocks['tt']['ee']` holds cov(C_l^TT, C_l^EE) with 
            shape (lmaxTT + 1, lmaxEE + 1). Otherwise, the shape of the binned 
            blocks depends on the shapes of the binning matrices and on the 
            `bin_ranges`, if they were provided.
        """
        cov_spectra = self.spectra if self.calc_lensingxcmb else self.cmb_spectra
        blocks = {s: {} for s in cov_spectra}
        for i, s1 in enumerate(self.cmb_spectra):
            # CMB x CMB:
            for s2 in self.cmb_spectra[i:]:
                if self.use_derivs_wrt_lensing:
                    blocks[s1][s2] = self.cmbxcmb_block_from_derivs_wrt_lensing(s1, s2, binned=binned, save=save)
                    if self.use_derivs_wrt_unlensed:
                        blocks[s1][s2] += self.cmbxcmb_block_from_derivs_wrt_unlensed(s1, s2, binned=binned, save=save)
                elif self.use_derivs_wrt_unlensed:
                    blocks[s1][s2] = self.cmbxcmb_block_from_derivs_wrt_unlensed(s1, s2, binned=binned, save=save)
                if (s2 != s1) and (s2 in blocks[s1].keys()):
                    blocks[s2][s1] = np.transpose(blocks[s1][s2])
            # lensing x CMB:
            if self.calc_lensingxcmb:
                blocks['kk'][s1] = self.lensingxcmb_block(s1, binned=binned, save=save)
                blocks[s1]['kk'] = np.transpose(blocks['kk'][s1])
        return blocks

