# ```@setup binning
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 

configfile = ARGS[1]

## get modules and utility functions  
using Plots
using TOML
using Healpix
include("util.jl")

config = TOML.parsefile(configfile)
nside = config["general"]["nside"]
lmax = nside2lmax(nside)

pl = PlanckReferenceCov(joinpath(config["scratch"], "plicref"))

## read binning scheme
binfile = joinpath(@__DIR__, "../", "input", "binused.dat")
P, lb = util_planck_binning(binfile; lmax=lmax);

##
freq1, freq2 = "100", "100"
Wl = util_planck_beam_Wl(freq1, "hm1", freq2, "hm2", :EE, :EE; 
    lmax=lmax, beamdir=joinpath(config["scratch"], "beams"))
plot(Wl, yaxis=:log10, label="\$B_{\\ell}\$")

##
run_name = config["general"]["name"]
mapids = [k for k in keys(config["map"])]
spectrapath = joinpath(config["scratch"], "rawspectra")

mapid1 = "P$(freq1)hm1"
mapid2 = "P$(freq2)hm2"
spec = CSV.read(joinpath(spectrapath,"$(run_name)_$(mapid1)x$(mapid2).csv"), DataFrame)
plot(spec.ell, spec.ell.^2 .* spec.EE, label="$(run_name)_$(mapid1)x$(mapid2)",
    xlabel="multipole moment", ylabel="\$\\ell^2 C_{\\ell}^{EE}\$", xlim=(0,2nside))

##
lbref, cbref, err_ref, _ = extract_spec_and_cov(pl, "EE_$(freq1)x$(freq2)")

##
cl = SpectralVector(copy(spec.EE))
this_lmax = max(lastindex(cl), lastindex(Wl))
pixwinT, pixwinP = pixwin(nside; pol=true)
cl[0:this_lmax] ./= Wl[0:this_lmax] .* pixwinP[1:(this_lmax+1)].^2
cl[0:1] .= 0.0
cb = P * parent(cl)
planck_bin_choice = findfirst(lb .≥ lbref[1]):findlast(lb .≤ lbref[end])

nside_cut = 1:length(planck_bin_choice)
plot(lb[planck_bin_choice], (cb[planck_bin_choice] .- cbref[nside_cut]) ./ err_ref[nside_cut], 
    ylim=(-2,2), label=label="$(run_name)_$(mapid1)x$(mapid2)")
