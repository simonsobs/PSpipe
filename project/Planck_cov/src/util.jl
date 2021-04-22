
# # Utilities 
# Unlike every other file in the pipeline, this file is not intended to be run directly.
# Instead, include this in other files. These utilities provide an interface to the Planck
# data products, namely 
# 1. binning matrix
# 2. beam ``W_{\ell}^{XY} = B_{\ell}^X B_{\ell}^{Y}``
# 3. ``\texttt{plic}`` reference covariance matrix and reference spectra, for comparison plots


using PowerSpectra
using DataFrames, CSV
using DelimitedFiles
using LinearAlgebra


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
    if (parse(Int, freq1) == parse(Int, freq2)) && ((split1 == "hm2") && (split1 == "hm1"))
        split1, split2 = split2, split1
    end

    fname = "Wl_R3.01_plikmask_$(freq1)$(split1)x$(freq2)$(split2).fits"
    f = PowerSpectra.FITS(joinpath(beamdir, "BeamWf_HFI_R3.01", fname))
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
