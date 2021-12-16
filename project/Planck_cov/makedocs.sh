# download stuff
julia pipeline/setup.jl example.toml

# we want the intermediate product: masks * missing pixels
julia pipeline/rawspectra.jl example.toml P100hm1 P100hm2
julia pipeline/rawspectra.jl example.toml P143hm1 P143hm2
julia pipeline/rawspectra.jl example.toml P217hm1 P217hm2

# docs are assembled with this
julia docs/make.jl
