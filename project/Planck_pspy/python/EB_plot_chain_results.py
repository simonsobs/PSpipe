import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from getdist import plots

mpl.use("Agg")

roots = ["mcmc"]
params = ["beta", "alpha100", "alpha143", "alpha217", "alpha353"]


g = plots.get_subplot_plotter(
    chain_dir=os.path.join(os.getcwd(), "chains"),
    analysis_settings={"ignore_rows": 0.3},
)
kwargs = dict(colors=["k"], lws=[1])
g.triangle_plot(roots, params, **kwargs, diag1d_kwargs=kwargs)


# Show Minami & Komatsu results https://arxiv.org/abs/2011.11254
eb_results = {
    "beta": {"mean": 0.35, "std": 0.14},
    "alpha100": {"mean": -0.28, "std": 0.13},
    "alpha143": {"mean": +0.07, "std": 0.12},
    "alpha217": {"mean": -0.07, "std": 0.11},
    "alpha353": {"mean": -0.09, "std": 0.11},
}

from scipy.stats import norm

for i, param in enumerate(params):
    ax = g.subplots[i, i]
    xmin, xmax, ymin, ymax = ax.axis()
    x = np.linspace(xmin, xmax, 100)
    posterior = norm.pdf(x, eb_results[param]["mean"], eb_results[param]["std"])
    ax.plot(x, posterior / np.max(posterior), color="tab:red")

# Fake legend
g.subplots[0, 0].plot([], [], color="tab:red", label="Minami & Komatsu")
g.subplots[0, 0].plot([], [], color="k", label="PSpipe")
g.subplots[0, 0].legend(loc="upper left", bbox_to_anchor=(1, 1))

# Add table on figure
table_results = r"""
\begin{tabular} { l  c}
 Parameter &  68\% limits\\
\hline
{\boldmath$\alpha_{100}   $} & $-0.28\pm0.13 $\\
{\boldmath$\alpha_{143}   $} & $0.07 \pm0.12 $\\
{\boldmath$\alpha_{217}   $} & $-0.07\pm0.11 $\\
{\boldmath$\alpha_{353}   $} & $-0.09\pm0.11 $\\
{\boldmath$\beta          $} & $0.35 \pm0.14 $\\
\hline
\end{tabular}
"""

with mpl.rc_context(rc={"text.usetex": True}):
    table = g.sample_analyser.mcsamples[roots[0]].getTable(limit=1, paramList=params)
    kwargs = dict(size=15, ha="right")
    g.subplots[0, 0].text(5, +0.0, "Minami \& Komatsu" + table_results.replace("\n", ""), **kwargs)
    g.subplots[0, 0].text(5, -1.0, "PSpipe" + table.tableTex().replace("\n", ""), **kwargs)

plt.savefig("EB_plot_chain_results.png")
