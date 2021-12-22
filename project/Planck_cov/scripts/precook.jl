# run with `julia scripts/precook.jl example.toml` to download precomputed spectra

using TOML
configfile = ARGS[1]
config = TOML.parsefile(configfile)


##
using Downloads 
using CSV, DataFrames

run_name = config["general"]["name"]
mapids = [k for k in keys(config["map"])]
spectrapath = joinpath(config["scratch"], "rawspectra")
mkpath(spectrapath)

function download_if_necessary(url, dest; verbose=true)
    if isfile(dest) == false
        verbose && println("Downloading ", dest)
        @time Downloads.download(url, dest)
    else
        verbose && println("Extant, skip ", dest)
    end
end

spec_url_root = "https://github.com/xzackli/PSPipePlanckRender.jl/releases/download/v0.1/"
spec_file = "rawspectra.tar.gz"
spec_url = spec_url_root * spec_file
spec_dest = joinpath(config["scratch"], spec_file)
download_if_necessary(spec_url, spec_dest)

run(`tar -xzvf $(spec_dest) --overwrite -C $(config["scratch"])`)
