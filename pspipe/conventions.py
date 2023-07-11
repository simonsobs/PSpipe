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


_product_dir = "."


def _get_directory(name):
    full_path = os.path.join(_product_dir, name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


module = sys.modules[__name__]

for d in directories:
    setattr(module, f"get_{d}_dir", lambda d=d: _get_directory(d))

# pspy spectra order
spectra = ["TT", "TE", "TB", "ET", "BT", "EE", "EB", "BE", "BB"]
