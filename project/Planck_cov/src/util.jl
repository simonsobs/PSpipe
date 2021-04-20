
# # Utilities 
# Unlike every other file in the pipeline, this file is not intended to be run directly.
# Instead, include this in other files.

using PowerSpectra
using DataFrames, CSV


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
