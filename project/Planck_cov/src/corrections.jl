# ```@setup corrections
# # the example command line input for this script
# ARGS = ["example.toml", "143", "143", "TT", "--plot"]
# ```
#
# # [Corrections for Point Sources (corrections.jl)](@id corrections)
# We need to estimate a correction to the covariance matrix due to insufficiently apodized 
# point source holes in the likelihood mask, and we had generated spectra in [`signalsim`](@ref signalsim).
# Now we will estimate a smooth correction from these spectra, by calculating the analytic covariance 
# and then using a Gaussian process to smoothly interpolate the deviation from 1 between the sample covariance 
# and the analytic covariance. 
#
# ```@raw html
# <pre class="shell">
# <code class="language-shell hljs">$ julia corrections.jl example.toml 143 143 TT</code></pre>
# ```
#
# # Configuration 
# As usual, we set up the various information we'll need later. 

configfile, freq1, freq2, spec = ARGS

freqs = [freq1, freq2]
splits = ["1", "2"]
print("$(freqs[1]), $(freqs[2]), $(spec)\n")

## We choose to never do ET, we'll just simulate TE but swap the maps.
transposed = (spec == "ET")
if transposed
    spec = "TE"
    splits = reverse(splits)
    freqs = reverse(freqs)
end

using Plots
using Healpix
using PowerSpectra
using TOML
using UUIDs, JLD2, FileIO
using Statistics
using GaussianProcesses

include("util.jl")

config = TOML.parsefile(configfile)
nside = config["general"]["nside"]
run_name = config["general"]["name"]
simpath = joinpath(config["scratch"], "signalsims", 
    "$(freq1)_$(freq2)_$(spec)")
lmax = nside2lmax(nside)
lmax_planck = min(2508, lmax)

# We'll load the existing simulated spectra from disk.

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

# Let's take a look at the results of the simulations compared to the signal.
if "--plot" ∈ ARGS
    plot(cl_mean[1:lmax_planck+1] .* collect(0:lmax_planck).^2, 
        yerr=(sqrt.(cl_var[1:lmax_planck+1])  .* collect(0:lmax_planck).^2), 
        label="sims", alpha=0.5)
    plot!(signal12[spec] .* collect(0:length(signal11[spec])-1).^2, xlim=(0,2nside),
        label="injected signal")
end

# Next, we'll estimate the analytic signal-only covariance matrix. This requires the masks,
# the signal spectrum, and some zero arrays for the pixel variances. 

X, Y = Symbol(spec[1]), Symbol(spec[2])
run_name = config["general"]["name"]
masktype1 = (X == :T) ? "T" : "P"
masktype2 = (Y == :T) ? "T" : "P"
mapid1 = "P$(freq1)hm$(splits[1])"
mapid2 = "P$(freq2)hm$(splits[2])"

maskfileT₁ = joinpath(config["scratch"], "masks", "$(run_name)_$(mapid1)_maskT.fits")
maskfileP₁ = joinpath(config["scratch"], "masks", "$(run_name)_$(mapid1)_maskP.fits")
maskfileT₂ = joinpath(config["scratch"], "masks", "$(run_name)_$(mapid2)_maskT.fits")
maskfileP₂ = joinpath(config["scratch"], "masks", "$(run_name)_$(mapid2)_maskP.fits")

maskT₁ = readMapFromFITS(maskfileT₁, 1, Float64)
maskP₁ = readMapFromFITS(maskfileP₁, 1, Float64)
maskT₂ = readMapFromFITS(maskfileT₂, 1, Float64)
maskP₂ = readMapFromFITS(maskfileP₂, 1, Float64);

zero_var_component = HealpixMap{Float64, RingOrder}(zeros(nside2npix(nside)))
zero_var = PolarizedHealpixMap(zero_var_component, zero_var_component, zero_var_component);

m1_signal = CovField(mapid1, maskT₁, maskP₁, zero_var)
m2_signal = CovField(mapid2, maskT₂, maskP₂, zero_var)


## convert a Vector (starting from 0) into a SpectralVector (i.e. a 0-indexed vector)
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
mask₁ = (spec[1] == 'T') ? maskT₁ : maskP₁
mask₂ = (spec[2] == 'T') ? maskT₂ : maskP₂
@time 𝐌 = mcm(symspec, mask₁, mask₂)
C_decoupled = decouple_covmat(C, 𝐌, 𝐌)

# Once a covariance matrix has been estimated, it needs to be corrected for mode-coupling 
# just like the spectra. 
# The covariance is possibly transposed, but we only care about the diagonal.

correction = SpectralVector(cl_var[:,1] ./ diag(parent(C_decoupled)))


# Now it's finally time to fit a smooth interpolation over the noisy correction we 
# just estimated. 

lmin_cut, lmax_cut = get_plic_ellrange(Symbol(spec), freq1, freq2)
correction[lmax_cut:end] .= 1.0

fit_ells = intersect((lmin_cut-20):(lmax_cut), 0:lmax)
u = correction[fit_ells] .- 1
t = 1.0 * collect(fit_ells)

# We use a really generic Gaussian kernel.

se = SE(0.0, 0.0)
m = MeanZero()
gp = GP(t,u,m,se)

using Optim
optimize!(gp)   # Optimise the hyperparameters
correction_gp = predict_y(gp, float.(0:(lmax_cut+20)))[1];
correction_gp[1:(lmin_cut-21)] .= 0.0

correction_path = joinpath(config["scratch"], "point_source_corrections")
mkpath(correction_path)

if transposed
    revspec = reverse(spec)
    correction_file = joinpath(correction_path, "$(freq1)_$(freq2)_$(revspec)_corr.dat")
else
    correction_file = joinpath(correction_path, "$(freq1)_$(freq2)_$(spec)_corr.dat")
end
writedlm(correction_file, correction_gp)

# This smooth curve will be applied to the diagonal of the covariance matrices later.
if "--plot" ∈ ARGS
    plot(correction_gp)
end
