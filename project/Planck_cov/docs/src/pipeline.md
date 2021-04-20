```@meta
EditURL = "https://github.com/simonsobs/PSpipe/tree/planckcov/project/Planck_cov/docs/src/pipeline.md"
```

# Pipeline

The following file lists the shell commands that comprise the pipeline.

```@raw html 
<pre class="shell">
<code class="language-shell hljs">julia src/setup.jl global.toml
</code></pre>
```

```@raw html 
<pre class="shell">
<code class="language-shell hljs">sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P100hm1 P100hm1"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P100hm1 P143hm1"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P100hm1 P143hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P100hm1 P217hm1"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P100hm1 P100hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P100hm1 P217hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P143hm1 P143hm1"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P143hm1 P143hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P143hm1 P217hm1"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P143hm1 P100hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P143hm1 P217hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P143hm2 P143hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P143hm2 P217hm1"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P143hm2 P100hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P143hm2 P217hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P217hm1 P217hm1"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P217hm1 P100hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P217hm1 P217hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P100hm2 P100hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P100hm2 P217hm2"
sbatch scripts/4core1hr.cmd "julia src/rawspectra.jl global.toml P217hm2 P217hm2"
</code></pre>
```

