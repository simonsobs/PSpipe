
# ```@setup slurmgen
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 

configfile = ARGS[1]

# # SLURM Commands for Spectra (slurmgen.jl)
# This command generates SLURM commands that executes [rawspectra.jl](@ref rawspectra)
# on all the pairs of maps in the config.
## this file just prints out the SLURM commands required to compute the spectra
using TOML
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

# ## Fit Noise Model
# These SLURM commands fit the raw spectra with the camspec noise model.
## loop over freqs and noise channels
for freq in ("100", "143", "217")
    for spec in ("TT", "EE")
        println("$(run) \"julia src/fitnoisemodel.jl global.toml $(freq) $(spec)\"")
    end
end
