
# ```@setup slurmgen
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 

configfile = ARGS[1]

# # 
# # [SLURM Commands for Spectra (slurmgen.jl)](@id slurmgen)
# This command generates SLURM commands that executes [rawspectra.jl](@ref rawspectra)
# on all the pairs of maps in the config.
## this file just prints out the SLURM commands required to compute the spectra
using TOML
config = TOML.parsefile(configfile)

# Let's generate the commands we need for likelihood spectra.
mapids = [k for k in keys(config["map"])]
cmd = "sbatch scripts/8core2hr.cmd"
for i in 1:length(mapids)
    for j in i:length(mapids)
        println("$(cmd) \"julia src/rawspectra.jl $(configfile) $(mapids[i]) $(mapids[j])\"")
    end
end

# I just paste these in and wait a few hours. The resulting spectra are deposited in the 
# `scratch` dir in the config.

# ## Fit Noise Model
# These SLURM commands fit the raw spectra with the camspec noise model.
## loop over freqs and noise channels
println()
for freq in ("100", "143", "217")
    for spec in ("TT", "EE")
        println("$(cmd) \"julia src/fitnoisemodel.jl $(configfile) $(freq) $(spec)\"")
    end
end


# ## Generating Signal-Only Simulations
# These SLURM commands correct the covariances for the insufficiently-apodized point source
# holes.
## loop over freqs and noise channels
println()
cmd = "sbatch scripts/8core2hr.cmd"
freqs = ("100", "143", "217")
for i in 1:3
    for j in i:3
        for spec in ("TT", "TE", "ET", "EE")
            freq1, freq2 = freqs[i], freqs[j]
            println("$(cmd) \"julia src/signalsim.jl $(configfile) $(freq1) $(freq2) $(spec) 5000\"")
        end
    end
end


# ## Fit Signal-Only Simulation Corrections
# These SLURM commands correct the covariances for the insufficiently-apodized point source
# holes.
## loop over freqs and noise channels
println()
freqs = ("100", "143", "217")
for i in 1:3
    for j in i:3
        for spec in ("TT", "TE", "ET", "EE")
            freq1, freq2 = freqs[i], freqs[j]
            println("$(cmd) \"julia src/corrections.jl $(configfile) $(freq1) $(freq2) $(spec)\"")
        end
    end
end

##
function unique_splits(X, Y, f1, f2)
    if (X == Y) && (f1 == f2)
        return [(1,2)]
    elseif (X==Y) && (f1 != f2)
        return [(1,2), (2,1)]
    else
        return [(1,1), (1,2), (2,1), (2,2)]
    end
end

using Test
using Base.Iterators

@test unique_splits("T", "T", "100", "100") == [(1,2)]
@test unique_splits("T", "E", "100", "100") == [(1,1), (1,2), (2,1), (2,2)]
@test unique_splits("T", "T", "100", "143") == [(1,2), (2,1)]
@test unique_splits("E", "E", "100", "143") == [(1,2), (2,1)]
@test unique_splits("T", "E", "100", "143") == [(1,1), (1,2), (2,1), (2,2)]

include("util.jl")
specs = (
    (:TT,"100","100"), (:TT,"143","143"), (:TT,"143","217"), (:TT,"217","217"), 
    (:EE,"100","100"), (:EE,"100","143"), (:EE,"100","217"), (:EE,"143","143"), 
    (:EE,"143","217"), (:EE,"217","217"), 
    (:TE,"100","100"), (:TE,"100","143"), (:TE,"100","217"), (:TE,"143","143"), 
    (:TE,"143","217"), (:TE,"217","217"),
    (:ET,"100","143"), (:ET,"100","217"), (:ET,"143","217")  # combined with TE in plic
)

constituents = []

for spec in specs
    AB, f1, f2 = spec
    A, B = string(AB)
    for (s1, s2) in unique_splits(A, B, f1, f2)
        push!(constituents, (A, B, f1, f2, s1, s2))
    end
end

run_name = config["general"]["name"]
cmd = "sbatch scripts/8core2hr.cmd"
scriptpath = joinpath(config["scratch"], "scripts")
mkpath(scriptpath)
open(joinpath(scriptpath, "$(run_name)_gen_covmats.sh"), "w") do f
    nspecs = length(constituents)

    ## generate the upper triangle
    for i1 in 1:nspecs
        for i2 in i1:nspecs
            A, B, f1, f2, s1, s2 = constituents[i1]
            C, D, f3, f4, s3, s4 = constituents[i2]
            write(f, "$(cmd) \"julia src/covmat.jl $(configfile) $f1 $f2 $f3 $f4 $A $B $C $D $s1 $s2 $s3 $s4\"\n")
        end
    end
end
