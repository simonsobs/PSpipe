#
# ```@setup setup
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 
#
# # [Setup (setup.jl)](@id setup)
#
# This pipeline is written in Julia, so you will need a [Julia](https://julialang.org/) 
# installation in order to run the components. We recommend
# you use the precompiled binaries provided on the Julia website. Make sure to add the 
# Julia executable to your path, as described in the 
# [platform-specific instructions.](https://julialang.org/downloads/platform/)
#
#

# ## Package Installation
# We use the package manager in the Julia interpeter
# to install the latest versions of Healpix and PowerSpectra. This will be simpler in
# the future, when we tag a stable version of these packages for the General Registry. 
# For now, we add the latest versions of these packages from GitHub. Note that package 
# installation requires an internet connection, so unlike the other parts of the pipeline,
# `setup.jl` requires an internet connection. If you're on a cluster, that means you need 
# to run this file on the head node in order to install packages.
using Pkg  
Pkg.add.(["Healpix", "PowerSpectra", "CSV", "DataFrames", "TOML", 
  "BlackBoxOptim", "FileIO", "JLD2", "FITSIO",
  "DataInterpolations", "Optim", "GaussianProcesses"]);
#
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
# TOML.print(Dict("dir"=>config))  # print just the "dir" TOML entry
# ```

##src
using TOML                                                                  #src
configfile = ARGS[1]  # read in the first command line argument             #src
println("config filename: ", configfile, "\n")                              #src
config = TOML.parsefile(configfile)                                         #src
TOML.print(config)  # print just the "dir" TOML entry   #src

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
mapdest = joinpath(config["scratch"], "maps")
maskdest = joinpath(config["scratch"], "masks")
beamdest = joinpath(config["scratch"], "beams")
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

# 
## We also want to retrieve the plic templates. 
plicrefdir = joinpath(config["scratch"])
plicreffile = joinpath(config["scratch"], "plicref.tar.gz")
download_if_necessary(
    "https://github.com/xzackli/PSPipePlanckRender.jl/releases/download/0.1.2/plicref.tar.gz", 
    plicreffile)
run(`tar -xzvf $(plicreffile) --overwrite -C $(plicrefdir)`);


# For the sims, we will use the full pixel weights to efficiently compute the spherical harmonic
# transforms. The pixel weights are lazily downloaded in the Healpix package. We trigger
# that downloading here. Similarly we trigger some downloading of example data from PowerSpectra.jl 
# for the purpose of plotting.

using Healpix
Healpix.applyFullWeights!(HealpixMap{Float64, RingOrder}(2048))
using PowerSpectra
PowerSpectra.planck256_beamdir()
