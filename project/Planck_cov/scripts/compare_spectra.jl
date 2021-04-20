
using TOML
using CSV, DataFrames
using Plots

configfile = "global.toml"
config = TOML.parsefile(configfile)


run_name = config["general"]["name"]
mapids = [k for k in keys(config["map"])]
spectrapath = joinpath(config["dir"]["scratch"], "rawspectra")

mapid1 = mapids[1]
mapid2 = mapids[4]
spec = DataFrame(CSV.File(joinpath(spectrapath,"$(run_name)_$(mapid1)x$(mapid2).csv")))
plot(spec.ell, spec.ell.^2 .* spec.TT, label="$(run_name)_$(mapid1)x$(mapid2)",
    xlabel="multipole moment", ylabel="\$\\ell^2 C_{\\ell}^{TT}\$", xlim=(0,2000))


