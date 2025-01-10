using Documenter, Literate
ENV["PLOTS_DEFAULT_BACKEND"] = "GR"
ENV["GKSwstype"]="nul"
using Plots
using Plots.PlotMeasures: mm

default(
    fontfamily = "Computer Modern", linewidth=1.5,
    titlefontsize=(16+8), guidefontsize=(11+5), 
    tickfontsize=(8+4), legendfontsize=(8+4),
    left_margin=5mm, right_margin=5mm)

src = joinpath(@__DIR__, "src")
lit = joinpath(@__DIR__, "src")

config = Dict(
    "credit" => false,  # credit is configured to render in Documenter instead
    "repo_root_url"=> "https://github.com/simonsobs/PSpipe/blob/master/project/Planck_cov/",
)

nonexecute_config = copy(config)
nonexecute_config["codefence"] = "```julia" => "```"
execution_exclusion = []

mkpath(joinpath(@__DIR__, "build"))
cp(joinpath(@__DIR__, "..", "src", "util.jl"), joinpath(@__DIR__, "build", "util.jl"); force=true)
cp(joinpath(@__DIR__, "..", "example.toml"), joinpath(@__DIR__, "build", "example.toml"); force=true)
cp(joinpath(@__DIR__, "..", "input"), joinpath(@__DIR__, "input"); force=true)

for (root, _, files) ∈ walkdir(joinpath(@__DIR__, "../src")), file ∈ files
    splitext(file)[2] == ".jl" || continue
    ipath = joinpath(root, file)
    opath = lit
    if file ∉ execution_exclusion
        Literate.markdown(ipath, opath; config=config)
    else
        @warn("not executing unless forced for ", file)
        Literate.markdown(ipath, opath; config=nonexecute_config)
    end
end

## manually clean and add back files
rm(joinpath(@__DIR__, "build"); force=true, recursive=true)
mkpath(joinpath(@__DIR__, "build"))
cp(joinpath(@__DIR__, "..", "src", "util.jl"), joinpath(@__DIR__, "build", "util.jl"); force=true)
cp(joinpath(@__DIR__, "..", "example.toml"), joinpath(@__DIR__, "build", "example.toml"); force=true)
cp(joinpath(@__DIR__, "..", "input"), joinpath(@__DIR__, "input"); force=true)


makedocs(
    sitename = "PSPipe Planck",
    modules = Module[],
    clean=false,
    
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
        assets=["assets/so.css"],
    ),
    pages = [
        "Introduction" => "index.md",
        "setup.jl" => "setup.md",
        "util.jl" => "util.md",
        "rawspectra.jl" => "rawspectra.md",
        "fitnoisemodel.jl" => "fitnoisemodel.md",
        "signalsim.jl" => "signalsim.md",
        "corrections.jl" => "corrections.md",
        "whitenoise.jl" => "whitenoise.md",
        "covmat.jl" => "covmat.md",
        "slurmgen.jl" => "slurmgen.md",

        # "Pipeline" => "pipeline.md"
        ]
    )