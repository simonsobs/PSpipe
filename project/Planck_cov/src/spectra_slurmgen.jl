
# ```@setup spectra_slurmgen
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 


# # SLURM Commands for Spectra (spectra_slurmgen.jl)
# This command generates SLURM commands that executes [rawspectra.jl](@ref rawspectra)
# on all the pairs of maps in the config.
## this file just prints out the SLURM commands required to compute the spectra
using TOML

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
# The resulting spectra are written to `[scratch]/rawspectra/`. When the documentation 
# is rendered via GitHub actions, the example spectra are precomputed and downloaded,
# cooking-show style. Let's plot a spectrum.

using CSV, DataFrames

run_name = config["general"]["name"]
mapids = [k for k in keys(config["map"])]
spectrapath = joinpath(config["dir"]["scratch"], "rawspectra")

## if allowed to plot, read a spectrum csv file and plot the EE spectrum
if config["general"]["plot"] 
    using Plots
    mapid1 = mapids[1]
    mapid2 = mapids[4]
    spec = DataFrame(CSV.File(joinpath(spectrapath,"$(run_name)_$(mapid1)x$(mapid2).csv")))
    plot(spec.ell, spec.ell.^2 .* spec.EE, label="$(run_name)_$(mapid1)x$(mapid2)",
        xlabel="multipole moment", ylabel="\$\\ell^2 C_{\\ell}^{EE}\$")
end
