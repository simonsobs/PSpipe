#
# ```@setup covmat
# # the example command line input for this script
# ARGS = ["example.toml", "100", "100", "100", "100", "T", "T", "T", "T", "1", "2", "1", "2", "--plot"] 
# ``` 

freqs = [ARGS[2], ARGS[3], ARGS[4], ARGS[5]]
specs = [ARGS[6], ARGS[7], ARGS[8], ARGS[9]]
splits = [ARGS[10], ARGS[11], ARGS[12], ARGS[13]]

using TOML
using PowerSpectra
using Healpix
using JLD2
using Plots
include("util.jl")


configfile = ARGS[1]
config = TOML.parsefile(configfile)
nside = config["general"]["nside"]
lmax = nside2lmax(nside)
lmax_planck = min(2508, lmax)
run_name = config["general"]["name"]

## determine if a channel is polarized
ispol(spec_str) = (spec_str == "T") ? 0 : 1 

function swap!(specs, freqs, splits, i, j)
    specs[i], specs[j] = specs[j], specs[i]
    freqs[i], freqs[j] = freqs[j], freqs[i]
    splits[i], splits[j] = splits[j], splits[i]
end

## canonical form: E as far left as possible
function canonical!(specs, freqs, splits)
    transposed = false 

    if sum(ispol.(specs[1:2])) > sum(ispol.(specs[3:4]))
        swap!(specs, freqs, splits, 1, 3)
        swap!(specs, freqs, splits, 2, 4)
        transposed = true
    end
    if specs[1] == "E" && specs[2] == "T"
        swap!(specs, freqs, splits, 1, 2)
    end
    if specs[3] == "E" && specs[4] == "T"
        swap!(specs, freqs, splits, 3, 4)
    end

    return transposed
end

transposed = canonical!(specs, freqs, splits)
@show specs freqs splits transposed

## convenience names since we looking for covariance between AB and CD spectra
specAB = Symbol(specs[1]*specs[2])
specCD = Symbol(specs[3]*specs[4])
@show specAB specCD

# 

## store info needed for covariance calculations
spectra = Dict{SpectrumName, SpectralVector{Float64, Vector{Float64}}}()
noiseratios = Dict{SpectrumName, SpectralVector{Float64, Vector{Float64}}}()
identity_spectrum = SpectralVector(ones(lmax+1));

#

function extend_signal(s::Vector{T}) where T
    es = SpectralVector(zeros(T, nside2lmax(nside)+1))
    es[0:(length(s)-1)] .= s
    return es
end

function get_correction(freq1, freq2, spec)
    spec_str = string(spec)
    if parse(Float64, freq1) > parse(Float64, freq2)
        freq1, freq2 = freq2, freq1
        spec_str = reverse(spec_str)
    end
    correction_path = joinpath(config["scratch"], "point_source_corrections")
    correction_file = joinpath(correction_path, "$(freq1)_$(freq2)_$(spec_str)_corr.dat")
    correction_gp = readdlm(correction_file)[:,1]
    correction_gp[1:3] .= 0.0
    return extend_signal(correction_gp)
end

function whitenoiselevel(config, freq::String, split::String)
    whitenoisefile = joinpath(config["scratch"], "whitenoise.dat")
    df = CSV.read(whitenoisefile, DataFrame)
    for row in 1:nrow(df)
        if (string(df[row,:freq]) == freq) && (string(df[row,:split]) == split)
            return df[row, :noiseT], df[row, :noiseP]
        end
    end
    throw(ArgumentError("freq and split combo missing from white noise file"))
end

#
@. camspec_model(ℓ, α) =  α[1] * (100. / ℓ)^α[2] + α[3] * (ℓ / 1000.)^α[4] / ( 1 + (ℓ / α[5])^α[6] )^α[7]

#

ells = collect(0:(lmax))
coefficientpath = joinpath(config["scratch"], "noise_model_coeffs")
beampath = joinpath(config["scratch"], "beams")

## loop over combinations and put stuff into the dictionaries
for (f1, s1) in zip(freqs, splits)
    for (f2, s2) in zip(freqs, splits)
        f1_name = "P$(f1)hm$(s1)"
        f2_name = "P$(f2)hm$(s2)"
        signal, theory_dict = signal_and_theory(f1, f2, config)
        
        WlTT = util_planck_beam_Wl(f1, "hm"*s1, f2, "hm"*s2, :TT, :TT; 
            lmax=lmax, beamdir=beampath)
        WlTE = util_planck_beam_Wl(f1, "hm"*s1, f2, "hm"*s2, :TE, :TE; 
            lmax=lmax, beamdir=beampath)
        WlEE = util_planck_beam_Wl(f1, "hm"*s1, f2, "hm"*s2, :EE, :EE; 
            lmax=lmax, beamdir=beampath)

        spectra[(:TT, f1_name, f2_name)] = extend_signal(signal["TT"]) .* WlTT .*
            sqrt.(1 .+ get_correction(f1, f2, "TT"))
        spectra[(:TE, f1_name, f2_name)] = extend_signal(signal["TE"]) .* WlTE .*
            sqrt.(1 .+ get_correction(f1, f2, "TE"))
        spectra[(:EE, f1_name, f2_name)] = extend_signal(signal["EE"]) .* WlEE .*
            sqrt.(1 .+ get_correction(f1, f2, "EE"))
        
        if f1_name == f2_name
            whitenoiseT, whitenoiseP = whitenoiselevel(config, f1, s1)
            coeffTT = readdlm(joinpath(coefficientpath, "$(run_name)_$(f1)_TT_hm$(s1).dat"))[:,1]
            coeffEE = readdlm(joinpath(coefficientpath, "$(run_name)_$(f1)_EE_hm$(s1).dat"))[:,1]
            nlTT = camspec_model(ells, coeffTT)
            nlEE = camspec_model(ells, coeffEE)
            ratioTT = SpectralVector(nlTT ./ whitenoiseT)
            ratioEE = SpectralVector(nlEE ./ whitenoiseP)
            ratioTT[0:1] .= 1.0
            ratioEE[0:1] .= 1.0 
            noiseratios[(:TT, f1_name, f2_name)] = sqrt.(ratioTT)
            noiseratios[(:EE, f1_name, f2_name)] = sqrt.(ratioEE)
        else
            noiseratios[(:TT, f1_name, f2_name)] = identity_spectrum
            noiseratios[(:EE, f1_name, f2_name)] = identity_spectrum
        end
    end
end

fnames = ["P$(freq)hm$(split)" for (freq, split) in zip(freqs, splits)]

if "--plot" ∈ ARGS
    plot(noiseratios[(:TT, fnames[1], fnames[1])].parent, label="$(freqs[1]) hm1 TT")
    plot!(noiseratios[(:EE, fnames[1], fnames[1])].parent, label="$(freqs[1]) hm1 EE", 
        ylim=(0.0,2), xlim=(0,200), size=(600,300))
end

#

## load information that the covmat needs about a field
function loadcovfield(freq, split)
    # read maskT, maskP, covariances
    mapid = "P$(freq)hm$(split)"
    maskfileT = joinpath(config["scratch"], "masks", "$(run_name)_$(mapid)_maskT.fits")
    maskfileP = joinpath(config["scratch"], "masks", "$(run_name)_$(mapid)_maskP.fits")
    mapfile = joinpath(config["scratch"], "maps", config["map"][mapid])
    maskT = readMapFromFITS(maskfileT, 1, Float64)
    maskP = readMapFromFITS(maskfileP, 1, Float64)
    covII = nest2ring(readMapFromFITS(mapfile, 5, Float64)) * 1e12
    covQQ = nest2ring(readMapFromFITS(mapfile, 8, Float64)) * 1e12
    covUU = nest2ring(readMapFromFITS(mapfile, 10, Float64)) * 1e12
    return CovField(mapid, maskT, maskP, PolarizedHealpixMap(covII, covQQ, covUU))
end

fields = [loadcovfield(freqs[i], splits[i]) for i in 1:4];

## compute mode-coupling matrices
mcm_type_AB = (specAB == :EE) ? :M⁺⁺ : specAB
mcm_type_CD = (specCD == :EE) ? :M⁺⁺ : specCD

@time M₁₂ = mcm(mcm_type_AB, fields[1], fields[2])
@time M₃₄ = mcm(mcm_type_CD, fields[3], fields[4]);

## compute mode-coupled covariance matrix
workspace = CovarianceWorkspace(fields...);
@time C = coupledcov(Symbol(specs[1]*specs[2]), Symbol(specs[3]*specs[4]), 
    workspace, spectra, noiseratios)


## this is the product of A and B pixel windows
function spec2pixwin(XY::Symbol, nside)
    pixwinT, pixwinP = pixwin(nside; pol=true)
    pixwinP[1:2] .= 1.0
    pX = (first(string(XY)) == 'T') ? pixwinT : pixwinP
    pY =  (last(string(XY)) == 'T') ? pixwinT : pixwinP
    return SpectralVector(pX .* pY)
end

pixwinAB = spec2pixwin(specAB, nside)
pixwinCD = spec2pixwin(specCD, nside)

## load binnings
binfile = joinpath(@__DIR__, "../", "input", "binused.dat")
P, lb = util_planck_binning(binfile; lmax=lmax)
beampath = joinpath(config["scratch"], "beams")

C₀ = decouple_covmat(C, M₁₂, M₃₄)
WlAB = util_planck_beam_Wl(freqs[1], "hm"*splits[1], freqs[2], "hm"*splits[2], specAB, specAB; 
    lmax=lmax, beamdir=beampath)
WlCD = util_planck_beam_Wl(freqs[3], "hm"*splits[3], freqs[4], "hm"*splits[4], specCD, specCD; 
    lmax=lmax, beamdir=beampath)

for l1 in 0:lmax
    for l2 in 0:lmax
        C₀[l1, l2] /= WlAB[l1] * WlCD[l2] * pixwinAB[l1] * pixwinCD[l2]
    end
end

Cbb = P * parent(C₀) * (P')

# 
# lminAB, lmaxAB = get_plic_ellrange(specAB, freqs[1], freqs[2])
# lminCD, lmaxCD = get_plic_ellrange(specCD, freqs[3], freqs[4])
# rangeAB = findfirst(lb .> lminAB):findlast(lb .< lmaxAB)
# rangeCD = findfirst(lb .> lminCD):findlast(lb .< lmaxCD)

# cov_result = Cbb[rangeAB, rangeCD]

##
if transposed
    Cbb = Array(transpose(Cbb))
end

spectrapath = joinpath(config["scratch"], "rawspectra")
covpath = joinpath(config["scratch"], "covmatblocks")
mkpath(covpath)

covmat_filename = joinpath(covpath, join(specs) * "_" *
    join(freqs) * "_" * join(splits) * ".jld2")
@save covmat_filename Cbb
