
# Define the multipole range


multipole_range = {}
for el in ["el1", "el2", "el3"]:
    multipole_range[f"{el}_pa4_f150"] = {"T": [1250, 8500],  "E": [1250, 8500],  "B": [1250, 8500] }
    multipole_range[f"{el}_pa4_f220"] = {"T": [1000, 8500],  "E": [1000, 8500],  "B": [1000, 8500] }
    multipole_range[f"{el}_pa5_f090"] = {"T": [1000, 8500],  "E": [1000, 8500],  "B": [1000, 8500] }
    multipole_range[f"{el}_pa5_f150"] = {"T": [800, 8500],  "E": [800, 8500],  "B": [800, 8500] }
    multipole_range[f"{el}_pa6_f090"] = {"T": [1000, 8500],  "E": [1000, 8500],  "B": [1000, 8500] }
    multipole_range[f"{el}_pa6_f150"] = {"T": [600, 8500],  "E": [600, 8500],  "B": [600, 8500] }


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

