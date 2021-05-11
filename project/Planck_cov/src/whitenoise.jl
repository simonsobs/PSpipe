#
# ```@setup whitenoise
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 
#
# # White Noise Levels (whitenoise.jl)
configfile = first(ARGS)


## setup data
using Plots
using PowerSpectra
using Healpix
using TOML
using CSV
using DataFrames
# using BlackBoxOptim
# include("util.jl")

config = TOML.parsefile(configfile)
nside = config["general"]["nside"]
run_name = config["general"]["name"]
spectrapath = joinpath(config["dir"]["scratch"], "rawspectra")
lmax = min(2508,nside2lmax(nside))
npix = nside2npix(nside)
Ωp = 4π / npix

## loop over freqs and noise channels
df = DataFrame(freq = String[], split = String[], noiseT = Float64[], noiseP = Float64[])

for freq ∈ ("100", "143", "217"),  split ∈ ("1", "2")
    mapid = "P$(freq)hm$(split)"
    maskfileT = joinpath(config["dir"]["mask"], "$(run_name)_$(mapid)_maskT.fits")
    maskfileP = joinpath(config["dir"]["mask"], "$(run_name)_$(mapid)_maskP.fits")
    mapfile = joinpath(config["dir"]["map"], config["map"][mapid])
    maskT = readMapFromFITS(maskfileT, 1, Float64)
    maskP = readMapFromFITS(maskfileP, 1, Float64)
    covII = nest2ring(readMapFromFITS(mapfile, 5, Float64)) * 1e12
    covQQ = nest2ring(readMapFromFITS(mapfile, 8, Float64)) * 1e12
    covUU = nest2ring(readMapFromFITS(mapfile, 10, Float64)) * 1e12
    N_white_T = sum(maskT.pixels.^2 .* (covII.pixels) .* Ωp) ./ (sum(maskT.pixels.^2))
    N_white_P = sum(maskP.pixels.^2 .* (covUU.pixels + covQQ.pixels) .* Ωp) ./ (2 * sum(maskP.pixels.^2))
#     @show freq split N_white_T N_white_P
    push!(df, (freq, split, N_white_T, N_white_P))
end

csvfile = joinpath(config["dir"]["scratch"], "whitenoise.dat")
CSV.write(csvfile, df)
