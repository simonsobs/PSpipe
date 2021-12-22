# # [Utilities (util.jl)](@id util)
# Unlike every other file in the pipeline, this file is not intended to be run directly.
# Instead, include this in other files. These utilities provide an interface to the Planck
# data products, namely 
# 1. binning matrix
# 2. beam ``W_{\ell}^{XY} = B_{\ell}^X B_{\ell}^{Y}``
# 3. foreground model cross-spectra
# 4. ``\texttt{plic}`` reference covariance matrix and reference spectra, for comparison plots


using PowerSpectra
using DataFrames, CSV
using DelimitedFiles
using LinearAlgebra
using FITSIO

# ## Planck Binning 
# Planck bins the spectra at the very end, and applies an ``\ell (\ell+1)`` relative 
# weighting inside the bin. This utility function generates the binning operator 
# ``P_{b\ell}`` such that ``C_b = P_{b \ell} C_{\ell}``. It also returns the mean of the 
# left and right bin edges, which is what is used when plotting the Planck spectra.

"""
    util_planck_binning(binfile; lmax=6143)

Obtain the Planck binning scheme.

### Arguments:
- `binfile::String`: filename of the Planck binning, containing left/right bin edges

### Keywords
- `lmax::Int=6143`: maximum multipole for one dimension of the binning matrix

### Returns: 
- `Tuple{Matrix{Float64}, Vector{Float64}`: returns (binning matrix, bin centers)
"""
function util_planck_binning(binfile; lmax=6143)
    bin_df = DataFrame(CSV.File(binfile; 
        header=false, delim=" ", ignorerepeated=true))
    lb = (bin_df[:,1] .+ bin_df[:,2]) ./ 2
    P = binning_matrix(bin_df[:,1], bin_df[:,2], ℓ -> ℓ*(ℓ+1) / (2π); lmax=lmax)
    return P, lb[1:size(P,1)]
end


# ## Planck Beam 
# The Planck effective beams are azimuthally-averaged window functions induced by the 
# instrumental optics. This utility function reads the Planck beams from the RIMO, which 
# are of the form `TT_2_TT`, `TT_2_EE` etc. Conventionally, the Planck spectra are stored 
# with the diagonal of the beam-mixing matrix applied. 
#
# ```math
# C_{\ell} = W^{-1}_{\ell} \hat{C}_{\ell} 
# ```

"""
    util_planck_beam_Wl([T::Type=Float64], freq1, split1, freq2, split2, spec1_, spec2_; 
        lmax=4000, beamdir=nothing)

Returns the Planck beam transfer of [spec1]_to_[spec2], in Wl form.

### Arguments:
- `T::Type=Float64`: optional first parameter specifying numerical type
- `freq1::String`: frequency of first field 
- `split1::String`: split of first field (i.e. hm1)
- `freq2::String`: frequency of first field 
- `split2::String`: split of second field (i.e. hm2)
- `spec1_`: converted to string, source spectrum like TT 
- `spec2_`: converted to string, destination spectrum

### Keywords
- `lmax=4000`: maximum multipole
- `beamdir=nothing`: directory containing beam FITS files. if nothing, will fall back to 
        the PowerSpectra.jl beam files.

### Returns: 
- `SpectralVector`: the beam Wl, indexed 0:lmax
"""
function util_planck_beam_Wl(T::Type, freq1, split1, freq2, split2, spec1_, spec2_; 
                        lmax=4000, beamdir=nothing)
    if isnothing(beamdir)
        @warn "beam directory not specified. switching to PowerSpectra.jl fallback"
        beamdir = PowerSpectra.planck256_beamdir()
    end
    spec1 = String(spec1_)
    spec2 = String(spec2_)

    if parse(Int, freq1) > parse(Int, freq2)
        freq1, freq2 = freq2, freq1
        split1, split2 = split2, split1
    end
    if (freq1 == freq2) && ((split1 == "hm2") && (split2 == "hm1"))
        split1, split2 = split2, split1
    end

    fname = "Wl_R3.01_plikmask_$(freq1)$(split1)x$(freq2)$(split2).fits"
    f = FITS(joinpath(beamdir, "BeamWf_HFI_R3.01", fname))
    Wl = convert(Vector{T}, read(f[spec1], "$(spec1)_2_$(spec2)")[:,1])
    if lmax < 4000
        Wl = Wl[1:lmax+1]
    else
        Wl = vcat(Wl, last(Wl) * ones(T, lmax - 4000))
    end
    return SpectralVector(Wl)
end
util_planck_beam_Wl(T::Type, freq1, split1, freq2, split2, spec1; kwargs...) = 
    util_planck_beam_Wl(T, freq1, split1, freq2, split2, spec1, spec1; kwargs...)
util_planck_beam_Wl(freq1::String, split1, freq2, split2, spec1, spec2; kwargs...) = 
    util_planck_beam_Wl(Float64, freq1, split1, freq2, split2, spec1, spec2; kwargs...)


# ## Planck Likelihood Specifics
# The Planck likelihood uses a specific choice of spectra and multipole ranges for those 
# spectra. We provide some utility functions to retrieve a copy of the spectra order and 
# the multipole minimum and maximum for those spectra.

plic_order() = (
    (:TT,"100","100"), (:TT,"143","143"), (:TT,"143","217"), (:TT,"217","217"), 
    (:EE,"100","100"), (:EE,"100","143"), (:EE,"100","217"), (:EE,"143","143"), 
    (:EE,"143","217"), (:EE,"217","217"), 
    (:TE,"100","100"), (:TE,"100","143"), (:TE,"100","217"), (:TE,"143","143"), 
    (:TE,"143","217"), (:TE,"217","217")
)

const plic_ellranges = Dict(
    (:TT, "100", "100") => (30, 1197),
    (:TT, "143", "143") => (30, 1996),
    (:TT, "143", "217") => (30, 2508),
    (:TT, "217", "217") => (30, 2508),
    (:EE, "100", "100") => (30, 999),
    (:EE, "100", "143") => (30, 999),
    (:EE, "100", "217") => (505, 999),
    (:EE, "143", "143") => (30, 1996),
    (:EE, "143", "217") => (505, 1996),
    (:EE, "217", "217") => (505, 1996),

    (:TE, "100", "100") => (30, 999),
    (:TE, "100", "143") => (30, 999),
    (:TE, "100", "217") => (505, 999),
    (:TE, "143", "143") => (30, 1996),
    (:TE, "143", "217") => (505, 1996),
    (:TE, "217", "217") => (505, 1996),
    
    (:TT, "100", "143") => (30, 999),   # not used
    (:TT, "100", "217") => (505, 999),  # not used
)

function get_plic_ellrange(spec::Symbol, freq1, freq2)
    if spec ∈ (:TE, :ET)
        if parse(Float64, freq1) > parse(Float64, freq2)
            freq1, freq2 = freq2, freq1
        end
        return plic_ellranges[:TE, freq1, freq2]
    end
    return plic_ellranges[spec, freq1, freq2]
end


# ## Signal Spectra 
# The covariance matrix calculation and and signal simulations require an assumed signal 
# spectra. We use the same foreground spectra as used in the ``\text{plic}`` likelihood.
# This returns dictionaries for signal and theory ``C_{\ell}`` in ``\mu\mathrm{K}`` 
# between two frequencies. The data is stored in the `plicref` directory in the config.

function signal_and_theory(freq1, freq2, config::Dict)
    likelihood_data_dir = joinpath(config["scratch"], "plicref")
    th = read_commented_header(joinpath(likelihood_data_dir,"theory_cl.txt"))
    fg = read_commented_header(joinpath(likelihood_data_dir,
        "base_plikHM_TTTEEE_lowl_lowE_lensing.minimum.plik_foregrounds"))
        
    for (spec, f1, f2) in plic_order()
        lmin, lmax = get_plic_ellrange(spec, f1, f2)
        const_val = fg[lmax-1,"$(spec)$(f1)X$(f2)"]
        ## constant foreground level after lmax -- there are fitting artifacts otherwise
        fg[(lmax-1):end, "$(spec)$(f1)X$(f2)"] .= const_val
    end

    ## loop over spectra and also fill in the flipped name
    freqs = ("100", "143", "217")
    specs = ("TT", "TE", "ET", "EE")
    for f1 in freqs, f2 in freqs, spec in specs
        if "$(spec)$(f1)X$(f2)" ∉ names(fg)
            if "$(reverse(spec))$(f2)X$(f1)" ∈ names(fg)
                fg[!, "$(spec)$(f1)X$(f2)"] = fg[!, "$(reverse(spec))$(f2)X$(f1)"]
            else
                fg[!, "$(spec)$(f1)X$(f2)"] = zeros(nrow(fg))
            end
        end
    end

    ap(v) = vcat([0., 0.], v)
    ell_fac = fg[!, "l"] .* (fg[!, "l"] .+ 1) ./ (2π);
    signal_dict = Dict{String,Vector{Float64}}()
    theory_dict = Dict{String,Vector{Float64}}()

    for XY₀ in specs  ## XY₀ is the spectrum to store
        f₁, f₂ = parse(Int, freq1), parse(Int, freq2)
        if f₁ <= f₂
            XY = XY₀
        else  ## swap what we're looking for, as fg data only has those cross-spectra
            XY = XY₀[2] * XY₀[1]
            f₁, f₂ = f₂, f₁
        end
        if XY == "ET"
            theory_cl_XY = th[!, "TE"] ./ (th[!, "L"] .* (th[!, "L"] .+ 1) ./ (2π))
        else
            theory_cl_XY = th[!, XY] ./ (th[!, "L"] .* (th[!, "L"] .+ 1) ./ (2π))
        end
        fg_cl_XY = fg[!, "$(XY)$(f₁)X$(f₂)"] ./ (fg[!, "l"] .* (fg[!, "l"] .+ 1) ./ (2π))
        
        signal_dict[XY₀] =  ap(theory_cl_XY .+ fg_cl_XY[1:2507])
        theory_dict[XY₀] =  ap(theory_cl_XY)
    end
    return signal_dict, theory_dict
end


# ## Planck ``\texttt{plic}`` Reference
# In various parts of the pipeline, we want to compare our results to the official 2018 
# data release. These routines load them from disk. They're automatically downloaded to the 
# `plicref` directory specified in the configuration TOML. 


"""
    PlanckReferenceCov(plicrefpath::String)

Stores the spectra and covariances of the reference plic analysis.
"""
PlanckReferenceCov

struct PlanckReferenceCov{T}
    cov::Array{T,2}
    ells::Vector{Int}
    cls::Vector{T}
    keys::Vector{String}
    sub_indices::Vector{Int}
    key_index_dict::Dict{String,Int}
end

function PlanckReferenceCov(plicrefpath)
    ellspath = joinpath(plicrefpath, "vec_all_spectra.dat")
    clpath = joinpath(plicrefpath, "data_extracted.dat")
    covpath = joinpath(plicrefpath, "covmat.dat")
    keys = ["TT_100x100", "TT_143x143", "TT_143x217", "TT_217x217", "EE_100x100",
        "EE_100x143", "EE_100x217", "EE_143x143", "EE_143x217", "EE_217x217", "TE_100x100",
        "TE_100x143", "TE_100x217", "TE_143x143", "TE_143x217", "TE_217x217"]
    cov = inv(readdlm(covpath))
    ells = readdlm(ellspath)[:,1]
    cls = readdlm(clpath)[:,2]

    subarray_indices = collect(0:(size(cov,1)-2))[findall(diff(ells) .< 0) .+ 1] .+ 1
    sub_indices = [1, subarray_indices..., length(cls)+1]
    key_ind = 1:length(keys)
    key_index_dict = Dict(keys .=> key_ind)

    return PlanckReferenceCov{Float64}(cov, ells, cls, keys, sub_indices, key_index_dict)
end


"""
    get_subcov(pl::PlanckReferenceCov, spec1, spec2)

Extract the sub-covariance matrix corresponding to spec1 × spec2.

### Arguments:
- `pl::PlanckReferenceCov`: data structure storing reference covmat and spectra
- `spec1::String`: spectrum of form i.e. "TT_100x100"
- `spec2::String`: spectrum of form i.e. "TT_100x100"

### Returns: 
- `Matrix{Float64}`: subcovariance matrix
"""
function get_subcov(pl::PlanckReferenceCov, spec1, spec2)
    i = pl.key_index_dict[spec1]
    j = pl.key_index_dict[spec2]
    return pl.cov[
        pl.sub_indices[i] : (pl.sub_indices[i + 1] - 1),
        pl.sub_indices[j] : (pl.sub_indices[j + 1] - 1),
    ]
end

"""
    extract_spec_and_cov(pl::PlanckReferenceCov, spec1)

Extract the reference ells, cl, errorbar, and sub-covariance block for a spectrum × itself.

### Arguments:
- `pl::PlanckReferenceCov`: data structure storing reference covmat and spectra
- `spec1::String`: spectrum of form i.e. "TT_100x100"

### Returns: 
- `(ells, cl, err, this_subcov)`
"""
function extract_spec_and_cov(pl::PlanckReferenceCov, spec1)
    i = pl.key_index_dict[spec1]
    this_subcov = get_subcov(pl, spec1, spec1)
    ells = pl.ells[pl.sub_indices[i]:(pl.sub_indices[i + 1] - 1)]
    cl = pl.cls[pl.sub_indices[i]:(pl.sub_indices[i + 1] - 1)]
    err = sqrt.(diag(this_subcov))
    return ells, cl, err, this_subcov
end
