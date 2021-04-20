#
# ```@setup setup
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 
#
# # Setup (setup.jl)
#
# This pipeline is written in Julia, so you will need a [Julia](https://julialang.org/) 
# installation in order to run the components. We recommend
# you use the precompiled binaries provided on the Julia website. Make sure to add the 
# Julia executable to your path, as described in the 
# [platform-specific instructions.](https://julialang.org/downloads/platform/)
#
# The command-line interface for this basic pipeline setup script is
# ```@raw html
# <pre class="shell">
# <code class="language-shell hljs">$ julia setup.jl global.toml</code></pre>
# ```
# * It displays the contents of the global TOML configuration file named *global.toml*.
# * This script downloads the Planck data to the specified directories in *global.toml*.
#
#
#
# ## Configuration 
# All of the pipeline scripts take a configuration TOML file as the first argument. 
# We now print out just the `[dir]` entry in the TOML, which is what you will need to 
# configure.
#

#src   This has the @example manually specified so that it runs. By default, nothing else 
#src   in this script should run. The #src after makes it run in the script too.

# ```@example setup
# using TOML
# configfile = ARGS[1]  # read in the first command line argument
# println("config filename: ", configfile, "\n")
# 
# # take a look at the config
# config = TOML.parsefile(configfile)
# TOML.print(Dict("dir"=>config["dir"]))  # print just the "dir" TOML entry
# ```

##src
using TOML                                                                  #src
configfile = ARGS[1]  # read in the first command line argument             #src
println("config filename: ", configfile, "\n")                              #src
config = TOML.parsefile(configfile)                                         #src
TOML.print(Dict("dir"=>config["dir"]))  # print just the "dir" TOML entry   #src

# * The `scratch` space is where intermediary files are deposited. 
# * Note that each map file has an identifier. This shortens the long names, but more 
#   importantly allows one to set up a custom system of names when we cross-correlate Planck
#   with other experiments.
# * In this case, we preface all Planck maps and masks with `P`, and include the frequency
#   and split.


# # Downloading the Planck 2018 Data

## set up download urls
using Downloads

if length(ARGS) == 0
    TARGET_DIR = pwd()
else
    TARGET_DIR = ARGS[1]
end

## set up directories
mapdest = config["dir"]["map"]
maskdest = config["dir"]["mask"]
beamdest = config["dir"]["beam"]
mkpath(mapdest)
mkpath(maskdest)
mkpath(beamdest)

function download_if_necessary(url, dest; verbose=true)
    if isfile(dest) == false
        verbose && println("Downloading ", dest)
        @time Downloads.download(url, dest)
    else
        verbose && println("Extant, skip ", dest)
    end
end

# 
## now read from config and then actually download
mapfiles = values(config["map"])
maskfiles = [values(config["maskT"])..., values(config["maskP"])...]

for f in mapfiles
    download_if_necessary(joinpath(config["url"]["maps"], f), joinpath(mapdest, f))
end

for f in maskfiles
    download_if_necessary(joinpath(config["url"]["masks"], f), joinpath(maskdest, f))
end

## just download beamfile to the target directory base.
beamfile = "HFI_RIMO_BEAMS_R3.01.tar.gz"
fullbeamfile = joinpath(beamdest, beamfile)
download_if_necessary(joinpath(config["url"]["beams"], beamfile), fullbeamfile)
run(`cp $(fullbeamfile) $(joinpath(beamdest, "tempbeamgz"))`)
run(`gzip -f -d $(joinpath(beamdest, beamfile))`)
run(`mv $(joinpath(beamdest, "tempbeamgz")) $(fullbeamfile)`)
beamfiletar = replace(beamfile, ".tar.gz"=>".tar")
run(`tar -xf $(joinpath(beamdest, beamfiletar)) --overwrite -C $(beamdest)`);
