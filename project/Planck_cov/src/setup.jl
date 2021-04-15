#
# ```@setup setup
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["global.toml"] 
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
# * This script installs the Healpix and AngularPowerSpectra packages.
# * It displays the contents of the global TOML configuration file named *global.toml*.
#
#
# ## Package Installation
# We use the package manager to 
# install the latest versions of Healpix and AngularPowerSpectra. This will be simpler in
# the future, when we tag a stable version of these packages for the General Registry. 
# For now, we add the latest versions of these packages from GitHub. Note that package 
# installation requires an internet connection, so unlike the other parts of the pipeline,
# `setup.jl` requires an internet connection. If you're on a cluster, that means you need 
# to run this file on the head node in order to install packages.
#
# ```julia
# using Pkg                 
# Pkg.add(PackageSpec(name="Healpix", rev="master")) 
# Pkg.add(PackageSpec(name="AngularPowerSpectra", rev="main"))
# ```

#src   This is written in both markdown above and #src below, to prevent it from running 
#src   when the page is rendered, but make it run when executed as a script. The #src 
#src   prevents a line from being run during rendering. 
#src   During doc rendering, the packages should already be installed!

using Pkg                                                          #src
Pkg.add(PackageSpec(name="Healpix", rev="master"))                 #src
Pkg.add(PackageSpec(name="AngularPowerSpectra", rev="master"))     #src

#
# ## Configuration 
# All of the pipeline scripts take a configuration TOML file as the first argument. 
#

using TOML
configfile = ARGS[1]  # read in the first command line argument
println("config filename: ", configfile, "\n")

## take a look at the config
config = TOML.parsefile(configfile)
TOML.print(config)

# * The `scratch` space is where intermediary files are deposited. 
# * Note that each map file has an identifier. This shortens the long names, but more 
#   importantly allows one to set up a custom system of names when we cross-correlate Planck
#   with other experiments.
# * In this case, we preface all Planck maps and masks with `P`, and include the frequency
#   and split.
