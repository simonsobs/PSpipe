using Literate

config = Dict(
    "credit" => false,  # credit is configured to render in Documenter instead
    "repo_root_url"=> "https://github.com/simonsobs/PSpipe/tree/planckcov/project/Planck_cov/",
    # "repo_root_path" => ""
)
Literate.markdown("src/test.jl", joinpath(pwd(), "md/"); config=config)
