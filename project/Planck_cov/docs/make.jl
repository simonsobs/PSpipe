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
lit = joinpath(@__DIR__, "lit")

config = Dict(
    "credit" => false,  # credit is configured to render in Documenter instead
    "repo_root_url"=> "https://github.com/simonsobs/PSpipe/tree/planckcov/project/Planck_cov",
)

nonexecute_config = copy(config)
nonexecute_config["codefence"] = "```julia" => "```"
execution_exclusion = []

cp(joinpath(@__DIR__, "..", "src", "util.jl"), joinpath(@__DIR__, "build", "util.jl"); force=true)

for (root, _, files) ∈ walkdir(lit), file ∈ files
    splitext(file)[2] == ".jl" || continue
    ipath = joinpath(root, file)
    opath = splitdir(replace(ipath, lit=>src))[1]
    if file ∉ execution_exclusion
        Literate.markdown(ipath, opath; config=config)
    else
        @warn("not executing unless forced for ", file)
        Literate.markdown(ipath, opath; config=nonexecute_config)
    end
end


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
        "spectra_slurmgen.jl" => "spectra_slurmgen.md",

        "Pipeline" => "pipeline.md"
        ]
    )