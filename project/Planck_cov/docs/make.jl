using Documenter, Literate
using PyPlot
pygui(false)
PyPlot.svg(true)

src = joinpath(@__DIR__, "src")
lit = joinpath(@__DIR__, "lit")

config = Dict(
    "credit" => false,  # credit is configured to render in Documenter instead
    "repo_root_url"=> "https://github.com/simonsobs/PSpipe/tree/planckcov/project/Planck_cov",
)

for (root, _, files) ∈ walkdir(lit), file ∈ files
    splitext(file)[2] == ".jl" || continue
    ipath = joinpath(root, file)
    opath = splitdir(replace(ipath, lit=>src))[1]
    Literate.markdown(ipath, opath; config=config)
end

makedocs(
    sitename = "PSPipe Planck",
    modules = Module[],
    
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
        assets=["assets/so.css"],
    ),
    pages = [
        "Introduction" => "index.md",
        "Setup" => "setup.md",
        "Raw Spectra" => "rawspectra.md",
        "Test" => "test.md"
        ]
    )