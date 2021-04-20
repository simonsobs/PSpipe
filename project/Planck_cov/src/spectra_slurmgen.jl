
# ```@setup spectra_slurmgen
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 


# # SLURM Commands for Spectra (spectra_slurmgen.jl)
# This command generates SLURM commands that executes [rawspectra.jl](@ref rawspectra)
# on all the pairs of maps in the config.

using TOML
using Plots

# The first step is just to unpack the command-line arguments, which consist of the 
# TOML config file and the map names, which we term channels 1 and 2.

configfile = ARGS[1]
config = TOML.parsefile(configfile)

# Let's generate the commands we need for likelihood spectra.
mapids = [k for k in keys(config["map"])]
run = "sbatch scripts/4core1hr.cmd"
for i in 1:length(mapids)
    for j in i:length(mapids)
        println("$(run) \"julia src/rawspectra.jl global.toml $(mapids[i]) $(mapids[j])\"")
    end
end

# I just paste these in and wait a few hours. The resulting spectra are deposited in the 
# `scratch` dir in the config.


# # Plotting Some Examples
# The resulting spectra are written to `[scratch]/rawspectra/`. The example spectra for 
# documentation rendering (what you're seeing now) have been precomputed and downloaded,
# instead of 

if config["general"]["plot"] == false
    Plots.plot(args...; kwargs...) = nothing
    Plots.plot!(args...; kwargs...) = nothing
end


