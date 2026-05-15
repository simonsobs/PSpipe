description = """
This script generates a yaml file for the sims given the parafile
"""

import yaml 
from pspipe_utils import log, pspipe_list
from pspy import so_dict, pspy_utils
import argparse

parser = argparse.ArgumentParser(description=description,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('paramfile', type=str,
                    help='Filename (full or relative path) of paramfile to use')
args = parser.parse_args()

class FlowList(list):
    pass

class Defaults(list):
    pass

class MyDumper(yaml.SafeDumper):
    blank_line_level = 2

    def write_line_break(self, data=None):
        super().write_line_break(data)

        # Add an extra blank line between top-level items
        if len(self.indents) == self.blank_line_level:
            super().write_line_break()

def flow_list_representer(dumper, data):
    return dumper.represent_sequence(
        "tag:yaml.org,2002:seq",
        data,
        flow_style=True
    )

def defaults_representer(dumper, data):
    return dumper.represent_sequence(
        "!defaults",
        data,
        flow_style=True
    )

MyDumper.add_representer(FlowList, flow_list_representer)
MyDumper.add_representer(Defaults, defaults_representer)

d = so_dict.so_dict()
d.read_from_file(args.paramfile)

yaml_dir = d["yaml_sim_dir"]
pspy_utils.create_directory(yaml_dir)

arrays = d["arrays_lat_iso"]

# Build list of the different map_set
map_set_list = pspipe_list.get_map_set_list(d)
spec_name_list = pspipe_list.get_spec_name_list(d, delimiter="_")

i_start = d["iStart"]
i_stop = d["iStop"] + 1

lmax = d["lmax"]

# now generate syst param yaml
cal_err_file = d["cal_err_file"]
poleff_err_file = d["poleff_err_file"]

with open(cal_err_file, 'r') as f:
    cal_err = yaml.safe_load(f)

with open(poleff_err_file, 'r') as f:
    poleff_err = yaml.safe_load(f)

syst_yaml = {
    "calG_all": {
        "prior": {
            "dist": "norm",
            "loc": 1.0,
            "scale": 0.003},
        "ref": {
            "dist": "norm",
            "loc": 1.0,
            "scale": 0.003},
        "proposal": 0.0015,
        "latex": rf"\mathrm{{cal}}_{{\rm ISO}}",
    }
}

for m in map_set_list:
    l, f = m.replace("lat_iso_", "").split("_")
    # for now, fixed bandshift
    syst_yaml[f"bandint_shift_{m}"] = {
        "value": 0.0,
        "latex": rf"\Delta_{{\rm band, {l}}}^{{\rm f{f}}}"
    }

for m in map_set_list:
    l, f = m.replace("lat_iso_", "").split("_")
    # cal and poleff errors from paramfile
    syst_yaml[f"cal_{m}"] = {
        "prior": {
        "dist": "norm",
        "loc": 1.0,
        "scale": cal_err[m]},
        "ref": {
        "dist": "norm",
        "loc": 1.0,
        "scale": 0.01},
        "proposal": 1e-3,
        "latex": rf"\mathrm{{c}}_{{\rm {l}}}^{{\rm f{f}}}"
    }

for m in map_set_list:
    l, f = m.replace("lat_iso_", "").split("_")
    syst_yaml[f"calE_{m}"] = {
        "prior": {
        "min": 0.9,
        "max": 1.1},
        "ref": {
        "dist": "norm",
        "loc": 1.0,
        "scale": poleff_err[m]},
        "proposal": 0.01,
        "latex": rf"\mathrm{{p}}_{{\rm {l}}}^{{\rm f{f}}}"
    }

MyDumper.blank_line_level = 1

with open(yaml_dir + "iso_syst_params.yaml", "w") as f:
    yaml.dump(syst_yaml, f, Dumper=MyDumper, sort_keys=False, default_flow_style=False)


gen_yaml_dict = {
"mflike.TTTEEE": {
"data_folder": "ISO/v1.0",

"input_file": "lat_iso_simu_sacc_00000.fits",

"cov_Bbl_file": "analytic_cov_and_Bbl.fits",

"requested_cls": ["tt", "te", "ee"],

"defaults": {
    "symmetrize": False,
    "polarizations": FlowList(["TT", "TE", "ET", "EE"]),
    "polarizations_auto": FlowList(["TT", "TE", "EE"]),
    "lmax": lmax,
    "scales": {
        "TT": [1000, lmax],
        "TE": [1000, lmax],
        "ET": [1000, lmax],
        "EE": [500, lmax]
            }
    },

"data": {
    "experiments": map_set_list,
    "spectra": [],
    }, 
    
"params": Defaults(["iso_syst_params"])
    }
}

for s in spec_name_list:
    s1, s2 = s.split("x")

    if s1 == s2:
        p = gen_yaml_dict["mflike.TTTEEE"]["defaults"]["polarizations_auto"]
    else:
        p = gen_yaml_dict["mflike.TTTEEE"]["defaults"]["polarizations"]

    gen_yaml_dict["mflike.TTTEEE"]["data"]["spectra"].append({
        "experiments": FlowList([s1, s2]),
        "polarization": p
    })

MyDumper.blank_line_level = 2 

for i in range(i_start, i_stop):
    yaml_dict = gen_yaml_dict.copy()
    yaml_dict["mflike.TTTEEE"]["input_file"] = f"lat_iso_simu_sacc_{i:05d}.fits"
    with open(yaml_dir + f"iso_sim_{i}.yaml", "w") as f:
        yaml.dump(yaml_dict, f, Dumper=MyDumper, sort_keys=False, default_flow_style=False)
