import os
import sys

# Setting directory name conventions
directories = (
    "windows",
    "mcms",
    "alms",
    "spectra",
    "best_fits",
    "noise_model",
    "sq_win_alms",
    "covariances",
)
# Build a lookup table with directory name and directory path. The path is the name but if we want
# to change the path then we prevent the API and the function name. For instance, we can set
# directories['windows'] = 'new_windows_path' but the the function will remain the same
# i.e. 'get_windows_dir()'
directories = {d: d for d in directories}

_product_dir = "."


def _get_directory(name):
    full_path = os.path.join(_product_dir, name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


module = sys.modules[__name__]

for name, path in directories.items():
    setattr(module, f"get_{name}_dir", lambda path=path: _get_directory(path))

# pspy spectra order
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
