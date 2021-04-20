
# ```@setup likelihoodspectra
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 

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

#

