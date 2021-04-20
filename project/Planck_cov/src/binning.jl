

# ```@setup binning
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 

configfile = ARGS[1]

## get modules and utility functions  
using Plots
using TOML
include("util.jl")

config = TOML.parsefile(configfile)
nside = config["general"]["nside"]
lmax = nside2lmax(nside)

## read binning scheme
binfile = joinpath(config["dir"]["pspipe_project"], "input", "binused.dat")
P, lb = util_planck_binning(binfile; lmax=lmax)

##
bl = util_planck_beam_bl("100", "hm1", "100", "hm2", :TT, :TT; 
    lmax=lmax, beamdir=config["dir"]["beam"])
plot(bl, yaxis=:log10, label="\$B_{\\ell}\$")
