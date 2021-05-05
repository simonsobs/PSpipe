#
# ```@setup signalcorrections
# # the example command line input for this script
# ARGS = ["example.toml",   "143", "143", "TT", "--plot"]
# ``` 

# configfile, freq1, freq2, spec = ["example.toml", "143", "143", "TT"]

configfile, freq1, freq2, spec = ARGS

freqs = [freq1, freq2]
splits = ["1", "2"]

using Plots
using TOML
using UUIDs, JLD2, FileIO
using Statistics
include("../src/util.jl")

config = TOML.parsefile(configfile)
nside = config["general"]["nside"]
run_name = config["general"]["name"]
simpath = joinpath(config["dir"]["scratch"], "signalsims", 
    "$(freq1)_$(freq2)_$(spec)")
lmax = nside2lmax(nside)
lmax_planck = min(2508, lmax)

print("$(freqs[1]), $(freqs[2]), $(spec)\n")

function readsims(path)
    files = [f for f in readdir(simpath; join=false) if f != "summary.jld2"]
    test_spec_ = load(joinpath(path,files[1]), "cl")
    cl_array = zeros((length(test_spec_), length(files)))
    for (i,f) in enumerate(files)
        cl_array[:,i] .= parent(load(joinpath(path,f), "cl"))
    end
    return cl_array
end

            
@time cl_array = readsims(simpath)
cl_mean = mean(cl_array, dims=2)
cl_var = var(cl_array, dims=2)
@save joinpath(simpath, "summary.jld2") cl_mean cl_var

signal11, th = signal_and_theory(freq1, freq1, config)
signal12, th = signal_and_theory(freq1, freq2, config)
signal21, th = signal_and_theory(freq2, freq1, config)
signal22, th = signal_and_theory(freq2, freq2, config)

##
if "--plot" ∈ ARGS
    plot(cl_mean[1:lmax_planck+1] .* collect(0:lmax_planck).^2, 
        yerr=(sqrt.(cl_var[1:lmax_planck+1])  .* collect(0:lmax_planck).^2), alpha=0.5)
    plot!(signal12[spec] .* collect(0:length(signal11[spec])-1).^2, xlim=(0,2nside))
end

##
X, Y = Symbol(spec[1]), Symbol(spec[2])
run_name = config["general"]["name"]
masktype1 = (X == :T) ? "T" : "P"
masktype2 = (Y == :T) ? "T" : "P"
mapid1 = "P$(freq1)hm$(splits[1])"
mapid2 = "P$(freq2)hm$(splits[2])"

using Healpix
maskfileT₁ = joinpath(config["dir"]["mask"], "$(run_name)_$(mapid1)_maskT.fits")
maskfileP₁ = joinpath(config["dir"]["mask"], "$(run_name)_$(mapid1)_maskP.fits")
maskfileT₂ = joinpath(config["dir"]["mask"], "$(run_name)_$(mapid2)_maskT.fits")
maskfileP₂ = joinpath(config["dir"]["mask"], "$(run_name)_$(mapid2)_maskP.fits")

maskT₁ = readMapFromFITS(maskfileT₁, 1, Float64)
maskP₁ = readMapFromFITS(maskfileP₁, 1, Float64)
maskT₂ = readMapFromFITS(maskfileT₂, 1, Float64)
maskP₂ = readMapFromFITS(maskfileP₂, 1, Float64);

zero_var_component = HealpixMap{Float64, RingOrder}(zeros(nside2npix(nside)))
zero_var = PolarizedHealpixMap(zero_var_component, zero_var_component, zero_var_component);

m1_signal = CovField(mapid1, maskT₁, maskP₁, zero_var)
m2_signal = CovField(mapid1, maskT₂, maskP₂, zero_var)

# convert a Vector (starting from 0) into a SpectralVector of full length
function format_signal(v::Vector{T}, nside) where T
    result = SpectralVector(zeros(T, nside2lmax(nside)+1))
    last_ind = min(length(v), length(result)) - 1
    for ℓ in 0:last_ind
        result[ℓ] = v[ℓ + 1]
    end
    return result
end


spectra = Dict{SpectrumName, SpectralVector{Float64, Vector{Float64}}}(
    (:TT, mapid1, mapid1) => format_signal(signal11["TT"], nside),
    (:TT, mapid1, mapid2) => format_signal(signal12["TT"], nside),
    (:TT, mapid2, mapid1) => format_signal(signal21["TT"], nside),
    (:TT, mapid2, mapid2) => format_signal(signal22["TT"], nside),

    (:EE, mapid1, mapid1) => format_signal(signal11["EE"], nside),
    (:EE, mapid1, mapid2) => format_signal(signal12["EE"], nside),
    (:EE, mapid2, mapid1) => format_signal(signal21["EE"], nside),
    (:EE, mapid2, mapid2) => format_signal(signal22["EE"], nside),

    (:TE, mapid1, mapid1) => format_signal(signal11["TE"], nside),
    (:TE, mapid1, mapid2) => format_signal(signal12["TE"], nside),
    (:TE, mapid2, mapid1) => format_signal(signal21["TE"], nside),
    (:TE, mapid2, mapid2) => format_signal(signal22["TE"], nside),
);

workspace_signal = CovarianceWorkspace(m1_signal, m2_signal, m1_signal, m2_signal);

@time C = coupledcov(Symbol(spec), Symbol(spec), workspace_signal, spectra)

symspec = (spec == "EE") ? :M⁺⁺ : Symbol(spec)
@time 𝐌 = mcm(symspec, maskT₁, maskT₂)
C_decoupled = decouple_covmat(C, 𝐌, 𝐌)



correction = SpectralVector(cl_var[:,1] ./ diag(parent(C_decoupled)))

lmin_cut, lmax_cut = plic_ellranges()[(Symbol(spec), freq1, freq2)]
correction[lmax_cut:end] .= 1.0

using GaussianProcesses

fit_ells = (lmin_cut-20):(lmax_cut)
u = correction[fit_ells] .- 1
t = 1.0 * collect(fit_ells)

# Set-up mean and kernel
se = SE(0.0, 0.0)
m = MeanZero()
gp = GP(t,u,m,se)

using Optim
optimize!(gp)   # Optimise the hyperparameters
correction_gp = predict_y(gp,float.(0:(lmax_cut+20)))[1];
correction_gp[1:(lmin_cut-21)] .= 1.0

correction_path = joinpath(config["dir"]["scratch"], "point_source_corrections")
mkpath(correction_path)
correction_file = joinpath(correction_path, "$(freq1)_$(freq2)_$(spec)_corr.dat")
writedlm(correction_file, correction_gp)

