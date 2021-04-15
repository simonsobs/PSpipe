#
# ```@setup rawspectra
# # the example command line input for this script
# ARGS = ["global.toml", "P143hm1", "P143hm2"] 
# ``` 


# # Raw Spectra (rawspectra.jl)
# The first step in the pipeline is simply to compute the pseudo-spectrum between the 
# maps ``X`` and ``Y``. 
# We define the pseudo-spectrum ``\widetilde{C}_{\ell}`` as the 
# result of the estimator on spherical harmonic coefficients of the *masked* sky,
# ```math
# \widetilde{C}_{\ell} = \frac{1}{2\ell+1} \sum_m a^{i,X}_{\ell m} a^{j,Y}_{\ell m}.
# ```
# Since we mask the galaxy and point sources, this is a biased estimate of the underlying
# power spectrum. The mask couples modes together, and also removes power from parts of the 
# sky. This coupling is described by a linear operator ``\mathbf{M}``, the mode-coupling 
# matrix. 
# For more details on spectra and mode-coupling, please refer to the [documentation for 
# AngularPowerSpectra.jl](https://xzackli.github.io/AngularPowerSpectra.jl/dev/spectra/).
# If this matrix is known, then one can perform a linear solve to obtain an unbiased
# estimate of the underlying power spectrum ``C_{\ell}``,
# ```math
# \langle\widetilde{C}_{\ell}\rangle = 
# \mathbf{M}^{XY}(i,j)_{\ell_1 \ell_2} \langle {C}_{\ell} \rangle,
# ```
# 
# This script performs this linear solve, *without accounting for beams*. The noise spectra
# are estimated from the difference of auto- and cross-spectra, 
# The command-line syntax for using this component to compute mode-coupled spectra is 
# 
# ```@raw html
# <pre class="shell">
# <code class="language-shell hljs">$ julia rawspectra.jl global.toml [map1] [map2]</code></pre>
# ```
# `[map1]` and `[map2]` must be names of maps described in global.toml. 
#
#
# ## File Loading and Cleanup

# This 
# page shows the results of running the command
# ```@raw html
# <pre class="shell">
# <code class="language-shell hljs">$ julia rawspectra.jl global.toml P143hm1 P143hm2</code></pre>
# ```

# We start by loading the necessary packages.

using TOML
using Healpix
using AngularPowerSpectra

# The first step is just to unpack the command-line arguments, which consist of the 
# TOML config file and the map names, which we term channels 1 and 2.

configfile, ch1, ch2 = ARGS[1], ARGS[2], ARGS[3]
#
config = TOML.parsefile(configfile)

# Before we start, we define a utility function for masking maps.

"""mask a map in-place"""
function mask!(m::Map, mask)
    m .*= mask
end

"""mask an IQU map in-place with a maskT and a maskP"""
function mask!(m::PolarizedMap, maskT, maskP)
    m.i .*= maskT
    m.q .*= maskP
    m.u .*= maskP
end

# For both input channels, we need to do some pre-processing steps.
# 1. We first load in the ``I``, ``Q``, and ``U`` Stokes vectors, which are FITS 
#    columns 1, 2, and 3 in the Planck map files. These must be converted from nested to ring
#    ordered, to perform SHTs.
# 2. We read in the corresponding masks in temperature and polarization.
# 3. We zero the missing pixels in the maps, and also zero the corresponding pixels in the 
#    masks.
# 4. We convert from ``\mathrm{K}_{\mathrm{CMB}}`` to ``\mu \mathrm{K}_{\mathrm{CMB}}``.
# 5. Apply a small polarization amplitude adjustment, listed as `poleff` in the config.
# 6. We also remove some noisy pixels with ``\mathrm{\sigma}(Q) > 10^6\,\mu\mathrm{K}^2`` 
#    or ``\mathrm{\sigma}(U) > 10^6\,\mu\mathrm{K}^2``, or if they are negative. This 
#    removes a handful of pixels in 
#    the 2018 maps at 100 GHz which interfere with covariance estimation.
# 7. Estimate and subtract the pseudo-monopole and pseudo-dipole.


"""Return (polarized map, maskT, maskP) given a config and map identifier"""
function load_maps_and_masks(config, ch)
    ## read map file 
    mapfile = joinpath(config["dir"]["map"], config["map"][ch1])
    polmap = PolarizedMap{Float64, RingOrder}(
        nest2ring(readMapFromFITS(mapfile, 1, Float64)),  # I
        nest2ring(readMapFromFITS(mapfile, 2, Float64)),  # Q
        nest2ring(readMapFromFITS(mapfile, 3, Float64)))  # U

    ## read maskT and maskP
    maskfileT = joinpath(config["dir"]["map"], config["maskT"][ch])
    maskfileP = joinpath(config["dir"]["map"], config["maskP"][ch])

    ## read Q and U pixel variances, and convert to μK
    covQQ = nest2ring(readMapFromFITS(mapfile, 8, Float64)) .* 1e12
    covUU = nest2ring(readMapFromFITS(mapfile, 10, Float64)) .* 1e12

    ## go from KCMB to μKCMB, and apply polarization factor
    poleff = config["poleff"][ch]
    mask!(polmap, 1e6, 1e6 * poleff)  # apply 1e6 to (I) and 1e6 * poleff to (Q,U)

    ## identify missing pixels and also pixels with crazy variances
    missing_pix = (polmap.i .< -1.6e30)
    missing_pix .*= (covQQ .> 1e6) .| (covUU .> 1e6) .| (covQQ .< 0.0) .| (covUU .< 0.0)

    ## apply the missing pixels to the map and mask for T/P
    mask!(polmap, missing_pix, missing_pix)
    mask!(maskT, missing_pix)
    mask!(maskP, missing_pix)

    ## fit and remove monopole/dipole in I
    monopole, dipole = fitdipole(polmap.i, maskT)
    subtract_monopole_dipole!(polmap.i, monopole, dipole)
    
    return polmap, maskT, maskP
end

#
# # Computing Spectra and Saving
#
# Once you have cleaned up maps and masks, you can compute mode-coupling matrices. The 
# calculation is described in [AngularPowerSpectra - Mode Coupling](
# https://xzackli.github.io/AngularPowerSpectra.jl/dev/spectra/#Mode-Coupling-for-EE,-EB,-BB).
#

"""Construct a NamedTuple with T,E,B names for the alms."""
function name_alms(alms::Vector)
    return (T=alms[1], E=alms[2], B=alms[3])
end


"""Compute spectra from alms of masked maps and alms of the masks themselves."""
function compute_spectra(maskedmap₁vec::Vector{Alm}, maskT₁::Alm, maskP₁::Alm,
                         maskedmap₂vec::Vector{Alm}, maskT₂::Alm, maskP₂::Alm)
    ## add TEB names
    maskedmap₁ = name_alms(maskedmap₁vec)
    maskedmap₂ = name_alms(maskedmap₂vec)
    spectra = Dict()

    ## spectra that are independent
    for (X, Y) in ((:T,:T), (:T,:E), (:E,:T), (:E,:E))
        spec = Symbol(X, Y)  # join X and Y 

        ## select temp or pol mask
        maskX = (X == :T) ? maskT₁ : maskP₁
        maskY = (Y == :T) ? maskT₂ : maskP₂

        ## compute mcm
        M = mcm(spec, maskX, maskY)
        pCl = SpectralVector(alm2cl(maskedmap₁[X], maskedmap₂[Y]))
        Cl = M \ pCl
        spectra[spec] = Cl  # store the result
    end

    M_EE_BB, M_EB_BE = mcm((:EE_BB, :EB_BE), maskP₁, maskP₂)

    ## EE and BB have to be decoupled together
    pCl_EE = SpectralVector(alm2cl(maskedmap₁[:E], maskedmap₂[:E]))
    pCl_BB = SpectralVector(alm2cl(maskedmap₁[:B], maskedmap₂[:B]))
    ## apply the 2×2 block mode-coupling matrix to the stacked EE and BB spectra
    @spectra Cl_EE, Cl_BB = M_EE_BB \ [pCl_EE; pCl_BB]
    spectra[:EE] = Cl_EE
    spectra[:BB] = Cl_BB

    ## EB and BE have to be decoupled together
    pCl_EB = SpectralVector(alm2cl(maskedmap₁[:E], maskedmap₂[:B]))
    pCl_BE = SpectralVector(alm2cl(maskedmap₁[:B], maskedmap₂[:E]))
    ## apply the 2×2 block mode-coupling matrix to the stacked EB and BE spectra
    @spectra Cl_EB, Cl_BE = M_EB_BE \ [pCl_EB; pCl_BE]
    spectra[:EB] = Cl_EB
    spectra[:BE] = Cl_BE

    return spectra
end

#