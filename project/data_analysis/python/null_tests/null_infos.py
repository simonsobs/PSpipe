
# Define the multipole range

skip_EB = True
fudge = False
skip_pa4_pol = True
plot_dir = "plots/array_nulls"

spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]

if skip_EB == True:
    tested_spectra = ["TT", "TE", "ET", "TB", "BT", "EE", "BB"]
    plot_dir = "plots/array_nulls_skip_EB"
else:
    tested_spectra = spectra
    plot_dir = "plots/array_nulls"



hist_label = ""

if skip_pa4_pol == True:
    hist_label = "skip_pa4pol"


multipole_range = {
    "dr6_pa4_f150": {
        "T": [1250, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    },
    "dr6_pa4_f220": {
        "T": [1000, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    },
    "dr6_pa5_f090": {
        "T": [1000, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    },
    "dr6_pa5_f150": {
        "T": [800, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    },
    "dr6_pa6_f090": {
        "T": [1000, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    },
    "dr6_pa6_f150": {
        "T": [600, 8500],
        "E": [500, 8500],
        "B": [500, 8500]
    }
}

# Options
l_pows = {
    "TT": 1,
    "TE": 0,
    "TB": 0,
    "ET": 0,
    "BT": 0,
    "EE": -1,
    "EB": -1,
    "BE": -1,
    "BB": -1
}
y_lims = {
    "TT": (-100000, 75000),
    "TE": (-30, 30),
    "TB": (-30, 30),
    "ET": (-30, 30),
    "BT": (-30, 30),
    "EE": (-0.01, 0.01),
    "EB": (-0.01, 0.01),
    "BE": (-0.01, 0.01),
    "BB": (-0.01, 0.01)
}
