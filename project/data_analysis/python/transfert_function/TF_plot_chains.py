import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from getdist import plots

mpl.use("Agg")

freq = [90, 150, 220]
spec_in_freq = {}
spec_in_freq[90] = ["dr6_pa5_f090", "dr6_pa6_f090"]
spec_in_freq[150] = ["dr6_pa4_f150", "dr6_pa5_f150", "dr6_pa6_f150"]
spec_in_freq[220] = ["dr6_pa4_f220"]

for f in freq:

    roots = []
    for spec in spec_in_freq[f]:
        roots += ["mcmc_%s" % spec]
    params = ["eps_min", "lmax", "beta"]


    g = plots.get_subplot_plotter(chain_dir=os.path.join(os.getcwd(), "chains"),analysis_settings={"ignore_rows": 0.3},)
    kwargs = dict(colors=["k"], lws=[1])
    g.triangle_plot(roots, params, **kwargs, diag1d_kwargs=kwargs)
    g.subplots[0, 0].plot([], [], color="k", label="PSpipe")
    g.subplots[0, 0].legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.savefig("chain_results_%s.png" % f)
