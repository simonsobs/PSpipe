#!/bin/bash
# install_pspipe.sh
# Purpose: Set up the PSpipe development environment.
# Date: 2026-03-06

set -e  # Exit on error

# --- Configuration ---
COMPILE_ARRAY_OPS=${1:-false} # Default to False unless "true" is passed as the first arg
PYTHON_VERSION="3.12"
BASE_DIR=$(realpath ..)
REPO_DIR="$BASE_DIR/repos"
INSTALL_DIR="$BASE_DIR/install"

# 1. Get auxiliary files
cd "$INSTALL_DIR"
curl -s https://api.github.com/repos/simonsobs/PSpipe/contents/project/SO/pISO/install?ref=zach_piso | \
grep "download_url" | \
grep -v "install.sh" | \
cut -d '"' -f 4 | \
xargs -n 1 wget

# 2. Create and activate Module
echo "Creating and loading tiger3/250723 Module"

# Initialize the module command (This mimics what /etc/bashrc does)
if [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
elif [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
fi

mv tiger_module_250723 ~/Modules/modulefiles/tiger3/250723
sed -i "18s|_ENLIB_PATH|$REPO_DIR/_enlib|" ~/Modules/modulefiles/tiger3/250723
module purge
module load tiger3/250723

# VERIFICATION: Check if the module actually loaded
if ! module list 2>&1 | grep -q "tiger3/250723"; then
    echo "Error: Failed to load tiger3/250723 module."
    exit 1
fi

# 3. Clone local requirements
echo "Checking out branches and prepping repositories..."

bash clone_local_requirements.sh

# 4. Organize enlib and switch branches
cd "$REPO_DIR"
mkdir -p _enlib
if [ -d "enlib" ]; then
    mv enlib _enlib/
fi

# Branch management
declare -A branches=(
    ["ducc"]="coupling_matrices_beta"
    ["pspy"]="zach_piso"
    ["pspipe_utils"]="zach_piso"
    ["PSpipe"]="zach_piso"
)

for repo in "${!branches[@]}"; do
    cd "$REPO_DIR/$repo"
    git checkout "${branches[$repo]}"
done

# 5. Virtual Environment Setup
cd "$BASE_DIR"
uv venv --python "$PYTHON_VERSION"
ln -sf .venv/bin/activate activate

# 6. Installation Phase
echo "Starting installation..."

# NumPy 1.x is required for ducc/pixell compatibility here
uv pip install "numpy<2"

# Install ducc from source (optimized local wheel)
cd "$INSTALL_DIR"
uv pip install ../repos/ducc --no-binary ducc0 --no-cache-dir

# Bulk install from requirements.in
uv pip install -r requirements.in 

# 7. Specialized Package Builds
uv pip install ../repos/optweight --no-build-isolation

# Dynamic Compilation: enlib array_ops
if [ "$COMPILE_ARRAY_OPS" = "true" ]; then
    echo "Compiling enlib array_ops..."
    source "$BASE_DIR/activate" # to access f2py in numpy
    cd "$REPO_DIR/_enlib/enlib"
    cp "$INSTALL_DIR/enlib_array_ops_python3.12_Makefile" array_ops/Makefile # to override defaults in enlib/compile_opts
    make array_ops
    deactivate
    cd "$INSTALL_DIR"
else
    echo "Skipping enlib array_ops compilation (default)."
fi

# Build pspy
uv pip install ../repos/pspy --no-build-isolation

# 8. Editable Installs
echo "Installing local packages in editable mode..."
uv pip install -e ../repos/sofind
uv pip install -e ../repos/mnms
uv pip install -e ../repos/pspipe_utils

# 9. Final Cleanup
cd "$BASE_DIR"
mkdir -p slurm_output

echo "Setup complete. Source the environment with: source activate"
