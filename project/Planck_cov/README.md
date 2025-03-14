# SO Planck Pipeline

This pipeline computes spectra and covariances for the Planck PR3 data release. Full documentation, generated directly from the code here, is available at https://simonsobs.github.io/planck-pr3-web.


This is a Simons Observatory pipeline for computing the spectra and covariances of the
[Planck 2018](https://arxiv.org/abs/1907.12875) high-``\ell`` likelihood. Default
settings have been chosen to reproduce the official ``\texttt{plic}`` analysis. The
code is also described in Li et al. (in prep). **Each page
is directly generated from a source file used in the pipeline**, in the style of
[literate programming](https://en.wikipedia.org/wiki/Literate_programming). Each source
file can be located on the [PSPipe GitHub](https://github.com/simonsobs/PSpipe) using the
*Edit on GitHub* link on the top of every page.

* All executed code within a pipeline component is displayed in light orange code blocks.
* Comments in the pipeline components are rendered as markdown.
* The rendered pages are run on low-resolution *Planck* 2018 half-mission maps and masks
  at 143 GHz, generated by averaging down to ``n_{\mathrm{side}} = 256``.
* At the top of each page, we provide examples for how to run the pipeline component
  on the *Planck* data.
* For the full analysis described in Li et al. (in prep), consult the pages on Slurm
  files and cluster use.

The map/alm handling routines for this project were contributed into
the [Healpix.jl](https://ziotom78.github.io/Healpix.jl/dev/) package, and the
mode-coupling and covariance matrix calculations were added to
[PowerSpectra.jl](https://xzackli.github.io/PowerSpectra.jl/dev/).
This pipeline mostly wrangles data and calls the routines from those packages.
