# run from Planck_cov/ as ".scripts/gen_example_spectra.sh"
export JULIA_NUM_THREADS=8
export OMP_NUM_THREADS=8

git clone --branch planckcov https://github.com/simonsobs/PSpipe.git /tmp/PSpipe

julia src/setup.jl example.toml

julia src/rawspectra.jl example.toml P100hm1 P100hm1
julia src/rawspectra.jl example.toml P100hm1 P143hm1
julia src/rawspectra.jl example.toml P100hm1 P143hm2
julia src/rawspectra.jl example.toml P100hm1 P217hm1
julia src/rawspectra.jl example.toml P100hm1 P100hm2
julia src/rawspectra.jl example.toml P100hm1 P217hm2
julia src/rawspectra.jl example.toml P143hm1 P143hm1
julia src/rawspectra.jl example.toml P143hm1 P143hm2
julia src/rawspectra.jl example.toml P143hm1 P217hm1
julia src/rawspectra.jl example.toml P143hm1 P100hm2
julia src/rawspectra.jl example.toml P143hm1 P217hm2
julia src/rawspectra.jl example.toml P143hm2 P143hm2
julia src/rawspectra.jl example.toml P143hm2 P217hm1
julia src/rawspectra.jl example.toml P143hm2 P100hm2
julia src/rawspectra.jl example.toml P143hm2 P217hm2
julia src/rawspectra.jl example.toml P217hm1 P217hm1
julia src/rawspectra.jl example.toml P217hm1 P100hm2
julia src/rawspectra.jl example.toml P217hm1 P217hm2
julia src/rawspectra.jl example.toml P100hm2 P100hm2
julia src/rawspectra.jl example.toml P100hm2 P217hm2
julia src/rawspectra.jl example.toml P217hm2 P217hm2
