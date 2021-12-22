# ```@setup whitenoise
# # all examples are run on an example global.toml and downsized maps.
# ARGS = ["example.toml"] 
# ``` 
#
# # [White Noise Levels (whitenoise.jl)](@id whitenoise)
# This file estimates the amplitude of the noise power spectrum from the 
# pixel variance map, under the assumption that the noise is white.
# We want to compute
# ```math
# N_{\ell}^{\mathrm{white}} = \frac{1}{\sum_p m_p^2} \sum_p \sigma_{II}^2 \Omega_p m_p^2
# ```
# where ``m_p`` is mask at pixel ``p``, ``\Omega_p`` is the pixel area, and ``\sigma_{II}^2`` 
# is the pixel variance. 
#
# ```@raw html
# <pre class="shell">
# <code class="language-shell hljs">$ julia whitenoise.jl example.toml</code></pre>
# ```
# 
# We first need to do the usual setup steps. We read the command-line arguments and load
# the packages we need.

configfile = first(ARGS)

## setup data
using Plots
using PowerSpectra
using Healpix
using TOML
using CSV
using DataFrames

config = TOML.parsefile(configfile)
nside = config["general"]["nside"]
run_name = config["general"]["name"]
spectrapath = joinpath(config["scratch"], "rawspectra")
lmax = min(2508,nside2lmax(nside))
npix = nside2npix(nside)
Ωp = 4π / npix

# Next, we loop over each half-mission frequency map, and estimate the white noise power
# spectrum.
df = DataFrame(freq = String[], split = String[], noiseT = Float64[], noiseP = Float64[])

for freq ∈ ("100", "143", "217"),  split ∈ ("1", "2")
    mapid = "P$(freq)hm$(split)"
    maskfileT = joinpath(config["scratch"], "masks", "$(run_name)_$(mapid)_maskT.fits")
    maskfileP = joinpath(config["scratch"], "masks", "$(run_name)_$(mapid)_maskP.fits")
    mapfile = joinpath(config["scratch"], "maps", config["map"][mapid])
    maskT = readMapFromFITS(maskfileT, 1, Float64)
    maskP = readMapFromFITS(maskfileP, 1, Float64)
    covII = nest2ring(readMapFromFITS(mapfile, 5, Float64)) * 1e12
    covQQ = nest2ring(readMapFromFITS(mapfile, 8, Float64)) * 1e12
    covUU = nest2ring(readMapFromFITS(mapfile, 10, Float64)) * 1e12
    N_white_T = sum(maskT.pixels.^2 .* (covII.pixels) .* Ωp) ./ (sum(maskT.pixels.^2))
    N_white_P = sum(maskP.pixels.^2 .* (covUU.pixels + covQQ.pixels) .* Ωp) ./ (2 * sum(maskP.pixels.^2))
    push!(df, (freq, split, N_white_T, N_white_P))
end

# Finally, we save this all to a CSV.
csvfile = joinpath(config["scratch"], "whitenoise.dat")
CSV.write(csvfile, df)


# Let's inspect the values that we saved.
df
