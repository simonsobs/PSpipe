#
# ```@setup signalsim
# # the example command line input for this script
# ARGS = ["example.toml",  "143", "143", "EE", "10", "--plot"]
# ``` 

configfile, freq1, freq2, spec, nsims = ARGS

# # Signal Sims (signalsim.jl)
# This script runs signal simulations.

using TOML
using Plots
using Healpix
using JLD2, UUIDs  # for saving sim arrays
include("util.jl")

config = TOML.parsefile(configfile)
nside = config["general"]["nside"]
run_name = config["general"]["name"]
spectrapath = joinpath(config["dir"]["scratch"], "rawspectra")
lmax = nside2lmax(nside)
lmax_planck = min(2508, lmax)
splits = "1", "2"  # planck never uses any other splits

# 
signal11, th = signal_and_theory(freq1, freq1, config)
signal12, th = signal_and_theory(freq1, freq2, config)
signal22, th = signal_and_theory(freq2, freq2, config)
signal = Dict(("1", "1") => signal11, ("1", "2") => signal12, ("2", "2") => signal22)

𝐂 = zeros(2, 2, lmax+1)
X, Y = spec
inds = 1:(lmax_planck+1)
𝐂[1,1,inds] .= signal[splits[1], splits[1]][X * X][inds]
𝐂[1,2,inds] .= signal[splits[1], splits[2]][X * Y][inds]
𝐂[2,1,inds] .= signal[splits[1], splits[2]][X * Y][inds]
𝐂[2,2,inds] .= signal[splits[2], splits[2]][Y * Y][inds];

# Next, we generate the mode-coupling matrix.
m1 = PolarizedMap{Float64, RingOrder}(nside)
m2 = PolarizedMap{Float64, RingOrder}(nside)
a1 = [Alm(lmax, lmax) for i in 1:3]
a2 = [Alm(lmax, lmax) for i in 1:3]

X, Y = Symbol(spec[1]), Symbol(spec[2])
run_name = config["general"]["name"]
masktype1 = (X == :T) ? "T" : "P"
masktype2 = (Y == :T) ? "T" : "P"
mapid1 = "P$(freq1)hm$(splits[1])"
mapid2 = "P$(freq2)hm$(splits[2])"

maskfile1 = joinpath(config["dir"]["mask"], "$(run_name)_$(mapid1)_mask$(masktype1).fits")
maskfile2 = joinpath(config["dir"]["mask"], "$(run_name)_$(mapid2)_mask$(masktype2).fits")
mask1 = readMapFromFITS(maskfile1, 1, Float64)
mask2 = readMapFromFITS(maskfile2, 1, Float64)

if spec == "EE"
    @time M = mcm(:EE_BB, map2alm(mask1), map2alm(mask2))
else
    @time M = mcm(Symbol(X,Y), map2alm(mask1), map2alm(mask2))
end

#

"""
    channelindex(s)

Convert string/char T,E,B => 1,2,3
"""
function channelindex(s)
    s = string(s)
    if s == "T"
        return 1
    elseif s == "E"
        return 2
    elseif s == "B"
        return 3
    end
    throw(ArgumentError("unknown spectrum"))
end

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
    if spec == "TT"
        monopole, dipole = fitdipole(m1.i * mask1)
        subtract_monopole_dipole!(m1.i, monopole, dipole)
        monopole, dipole = fitdipole(m2.i * mask2)
        subtract_monopole_dipole!(m2.i, monopole, dipole)
    end

    ## apply pixel weights and then map2alm
    Healpix.applyfullweights!(m1)
    Healpix.applyfullweights!(m2)
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

if "--plot" in ARGS
    @time simEE = sim_iteration(𝐂, m1, m2, a1, a2, M, "EE")
    plot(simEE .* eachindex(simEE).^2, label="sim")
    plot!(signal12["EE"] .* eachindex(0:lmax_planck).^2, xlim=(0,lmax_planck), label="input")
end

# This script generates many simulations.

simpath = joinpath(config["dir"]["scratch"], "signalsims", 
    "$(freq1)_$(freq2)_$(spec)")
mkpath(simpath)


for sim_index in 1:parse(Int,nsims)
    @time cl = sim_iteration(𝐂, m1, m2, a1, a2, M, spec)
    @save "$(simpath)/$(uuid4()).jld2" cl=cl
end
