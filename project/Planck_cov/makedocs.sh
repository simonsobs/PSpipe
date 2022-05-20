JULIA_NUM_THREADS=6

# download stuff
julia src/setup.jl example.toml

# we want the intermediate product: masks * missing pixels
julia src/rawspectra.jl example.toml P100hm1 P100hm2
julia src/rawspectra.jl example.toml P143hm1 P143hm2
julia src/rawspectra.jl example.toml P217hm1 P217hm2

# now we want to run some signal-only sims for the PS correction
julia src/signalsim.jl example.toml 143 143 TT 10
julia src/signalsim.jl example.toml 143 143 TE 10
julia src/signalsim.jl example.toml 143 143 EE 10

# compute the correction
julia src/corrections.jl example.toml 143 143 TT
julia src/corrections.jl example.toml 143 143 TE
julia src/corrections.jl example.toml 143 143 EE

# fit the noise model
julia src/fitnoisemodel.jl example.toml 143 TT
julia src/fitnoisemodel.jl example.toml 143 EE

# docs are assembled with this
julia docs/make.jl
