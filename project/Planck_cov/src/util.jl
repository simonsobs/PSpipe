
# # Utilities 
# Unlike every other file in the pipeline, this file is not intended to be run directly.
# Instead, include this in other files.

using PowerSpectra
using DataFrames, CSV
using DelimitedFiles
using LinearAlgebra

function util_planck_binning(binfile; lmax=6143)
    bin_df = DataFrame(CSV.File(binfile; 
        header=false, delim=" ", ignorerepeated=true))

    ## bin centers and binning matrix
    lb = (bin_df[:,1] .+ bin_df[:,2]) ./ 2
    P = binning_matrix(bin_df[:,1], bin_df[:,2], ℓ -> ℓ*(ℓ+1) / (2π); lmax=lmax)
    return P, lb
end


function util_planck_beam_bl(T::Type, freq1, split1, freq2, split2, spec1_, spec2_; 
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
    bl = convert(Vector{T}, read(f[spec1], "$(spec1)_2_$(spec2)")[:,1])
    if lmax < 4000
        bl = bl[1:lmax+1]
    else
        bl = vcat(bl, last(bl) * ones(T, lmax - 4000))
    end
    return SpectralVector(bl)
end
util_planck_beam_bl(T::Type, freq1, split1, freq2, split2, spec1; kwargs...) = 
    util_planck_beam_bl(T, freq1, split1, freq2, split2, spec1, spec1; kwargs...)
util_planck_beam_bl(freq1::String, split1, freq2, split2, spec1, spec2; kwargs...) = 
    util_planck_beam_bl(Float64, freq1, split1, freq2, split2, spec1, spec2; kwargs...)




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
    # subarray_indices = collect(0:size(
    cov = inv(readdlm(covpath))
    ells = readdlm(ellspath)[:,1]
    cls = readdlm(clpath)[:,2]

    subarray_indices = collect(0:(size(cov,1)-2))[findall(diff(ells) .< 0) .+ 1] .+ 1
    sub_indices = [1, subarray_indices..., length(cls)+1]
    key_ind = 1:length(keys)
    key_index_dict = Dict(keys .=> key_ind)

    return PlanckReferenceCov{Float64}(cov, ells, cls, keys, sub_indices, key_index_dict)
end

function get_subcov(pl::PlanckReferenceCov, spec1, spec2)
    i = pl.key_index_dict[spec1]
    j = pl.key_index_dict[spec2]
    return pl.cov[
        pl.sub_indices[i] : (pl.sub_indices[i + 1] - 1),
        pl.sub_indices[j] : (pl.sub_indices[j + 1] - 1),
    ]
end
function extract_spec_and_cov(pl::PlanckReferenceCov, spec1)
    i = pl.key_index_dict[spec1]
    this_subcov = get_subcov(pl, spec1, spec1)
    ells = pl.ells[pl.sub_indices[i]:(pl.sub_indices[i + 1] - 1)]
    cl = pl.cls[pl.sub_indices[i]:(pl.sub_indices[i + 1] - 1)]
    err = sqrt.(diag(this_subcov))
    return ells, cl, err, this_subcov
end
