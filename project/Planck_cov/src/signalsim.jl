#
# ```@setup signalsim
# # the example command line input for this script
# ARGS = ["example.toml",  "143", "143", "TT", "10", "--plot"]
# ``` 
#
# # [Signal Sims (signalsim.jl)](@id signalsim)
# This script runs signal-only simulations, and then computes spectra from those simulations 
# after applying and then correcting for the mode-coupling of the mask. The point-source holes 
# in the Planck likelihood masks are insufficiently apodized, which violates an assumption
# in the estimation of analytic covariance matrices. We wish to correct for these, so we 
# generate a bunch of simulations and compare the analytic and sample covariance.
#
# ```@raw html
# <pre class="shell">
# <code class="language-shell hljs">$ julia signalsim.jl example.toml 143 143 TT 10</code></pre>
# ```
#
# ## Configuration 
# We set up various information we'll need later. 
#
configfile, freq1, freq2, spec, nsims = ARGS

using TOML
using Plots
using Healpix
using JLD2, UUIDs  # for saving sim arrays
include("util.jl")

config = TOML.parsefile(configfile)
nside = config["general"]["nside"]
run_name = config["general"]["name"]
spectrapath = joinpath(config["scratch"], "rawspectra")
lmax = nside2lmax(nside)
lmax_planck = min(2508, lmax)
splits = "1", "2"  # planck never uses any other splits

# Next, we prepare the input signal used to generate these simulations.

signal11, th = signal_and_theory(freq1, freq1, config)
signal12, th = signal_and_theory(freq1, freq2, config)
signal22, th = signal_and_theory(freq2, freq2, config)
signal = Dict(("1", "1") => signal11, ("1", "2") => signal12, ("2", "2") => signal22)

# We'll store it in a nice array for access later, when simulating.

𝐂 = zeros(2, 2, lmax+1)
X, Y = spec
inds = 1:(lmax_planck+1)
𝐂[1,1,inds] .= signal[splits[1], splits[1]][X * X][inds]
𝐂[1,2,inds] .= signal[splits[1], splits[2]][X * Y][inds]
𝐂[2,1,inds] .= signal[splits[1], splits[2]][X * Y][inds]
𝐂[2,2,inds] .= signal[splits[2], splits[2]][Y * Y][inds];

# Next, we pre-allocate some structures that we'll re-use between simulation iterations.

m1 = PolarizedHealpixMap{Float64, RingOrder}(nside)
m2 = PolarizedHealpixMap{Float64, RingOrder}(nside)
a1 = [Alm(lmax, lmax) for i in 1:3]
a2 = [Alm(lmax, lmax) for i in 1:3]

X, Y = Symbol(spec[1]), Symbol(spec[2])
run_name = config["general"]["name"]
masktype1 = (X == :T) ? "T" : "P"
masktype2 = (Y == :T) ? "T" : "P"
mapid1 = "P$(freq1)hm$(splits[1])"
mapid2 = "P$(freq2)hm$(splits[2])"

# We'll apply the likelihood masks to the simulations.

maskfile1 = joinpath(config["scratch"], "masks", "$(run_name)_$(mapid1)_mask$(masktype1).fits")
maskfile2 = joinpath(config["scratch"], "masks", "$(run_name)_$(mapid2)_mask$(masktype2).fits")
mask1 = readMapFromFITS(maskfile1, 1, Float64)
mask2 = readMapFromFITS(maskfile2, 1, Float64)

# Next, we generate the mode-coupling matrix. 

if spec == "EE"
    @time M = mcm(:EE_BB, map2alm(mask1), map2alm(mask2))
else
    @time M = mcm(Symbol(X,Y), map2alm(mask1), map2alm(mask2))
end

# We write a function that does one iteration of this signal-only simulation and then 
# estimates spectra from it.

## map T,E,B => 1,2,3
channelindex(X) = findfirst(first(X), "TEB")

function sim_iteration(𝐂, m1, m2, a1, a2, M, spec::String)
    ## get indices of the spectrum
    c₁, c₂ = channelindex(spec[1]), channelindex(spec[2])

    ## zero out alms
    for i in 1:3
        fill!(a1[i].alm, 0.0)
        fill!(a2[i].alm, 0.0)
    end

    ## synthesize polarized spectrum into m1
    synalm!(𝐂, [a1[c₁], a2[c₂]])
    alm2map!(a1, m1)
    alm2map!(a2, m2)

    ## same signal, but different masks
    mask!(m1, mask1, mask1)
    mask!(m2, mask2, mask2)

    ## subtract monopole if TT
    if spec[1] == 'T'
        monopole, dipole = fitdipole(m1.i * mask1)
        subtract_monopole_dipole!(m1.i, monopole, dipole)
    end
    if spec[2] == 'T'
        monopole, dipole = fitdipole(m2.i * mask2)
        subtract_monopole_dipole!(m2.i, monopole, dipole)
    end

    ## apply pixel weights and then map2alm
    Healpix.applyFullWeights!(m1)
    Healpix.applyFullWeights!(m2)
    map2alm!(m1, a1; niter=0)
    map2alm!(m2, a2; niter=0)

    if spec == "EE"
        pCl_EE = SpectralVector(alm2cl(a1[c₁], a2[c₂]))
        pCl_BB = SpectralVector(zeros(length(pCl_EE)))
        @spectra Cl_EE, Cl_BB = M \ [pCl_EE; pCl_BB]
        return Cl_EE
    end

    ## otherwise easy mode coupling
    pCl_XY = SpectralVector(alm2cl(a1[c₁], a2[c₂]))
    return M \ pCl_XY
end

# Let's take a look at a simulated spectrum. Note the deviation at ``\ell > 2n_{\mathrm{side}}``. 
# This is due to pixelization, and is unimportant. For example, in the full Planck analysis we would
# expect to start seeing these problems at ``\ell \sim 2n_{\mathrm{side}} = 4096``.

if "--plot" in ARGS
    @time simTT = sim_iteration(𝐂, m1, m2, a1, a2, M, "TT")
    plot(simTT .* eachindex(simTT).^2, label="sim")
    plot!(signal12["TT"][1:(lmax_planck+1)] .* eachindex(0:lmax_planck).^2, 
        xlim=(0,lmax_planck), label="input")
end

# This script generates many simulations. We'll run this for `nsims` iterations and 
# then save the spectra to disk.

simpath = joinpath(config["scratch"], "signalsims", "$(freq1)_$(freq2)_$(spec)")
mkpath(simpath)

for sim_index in 1:parse(Int,nsims)
    @time cl = sim_iteration(𝐂, m1, m2, a1, a2, M, spec)
    @save "$(simpath)/$(uuid4()).jld2" cl=cl
end
