import os
import sys

# Setting directory name conventions
directories = (
    "windows",
    "mcms",
    "alms",
    "spectra",
    "sim_spectra",
    "best_fits",
    "noise_model",
    "sq_win_alms",
    "covariances",
    "sacc",
    "plots",
)
# Build a lookup table with directory name and directory path. The path is the name but if we want
# to change the path then we prevent the API and the function name. For instance, we can set
# directories['windows'] = 'new_windows_path' but the the function will remain the same
# i.e. 'get_windows_dir()'
directories = {d: d for d in directories}

_product_dir = "."


def _get_directory(name, create=False):
    full_path = os.path.join(_product_dir, name)
    if create:
        os.makedirs(full_path, exist_ok=True)
    if not os.path.exists(full_path):
        raise ValueError(
            f"The directory '{full_path}' does not exist! Might need to create it first."
        )
    return full_path


module = sys.modules[__name__]

for name, path in directories.items():
    setattr(
        module,
        f"get_{name}_dir",
        lambda path=path, create=False: _get_directory(path, create=create),
    )


# pspy spectra order
def get_spectra_order(cov_T_E_only=False):
    if cov_T_E_only:
        return ["TT", "TE", "ET", "EE"]
    return ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
