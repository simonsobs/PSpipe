#!/bin/bash
# install_pspipe.sh
# Purpose: Set up the PSpipe development environment.
# Date: 2026-03-06

set -e  # Exit on error

# --- 1. Argument Handling ---
# Check if at least one argument (the enlib flag) is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 [true/false: compile array_ops] [optional: email address]"
    exit 1
fi

COMPILE_ARRAY_OPS=$1        # Required: must be true or false
USER_EMAIL=$2               # Optional: email address
PYTHON_VERSION="3.12"
BASE_DIR=$(realpath ..)
REPO_DIR="$BASE_DIR/repos"
INSTALL_DIR="$BASE_DIR/install"

echo "Running setup in: $BASE_DIR"

# 1. Get auxiliary files
cd "$INSTALL_DIR"
curl -s https://api.github.com/repos/simonsobs/PSpipe/contents/project/SO/pISO/install?ref=zach_piso | \
grep "download_url" | \
grep -v "install.sh" | \
cut -d '"' -f 4 | \
xargs -n 1 wget

# 2. Create and activate Module
echo "Creating and loading tiger3/250723 Module"

# from talking with Gemini, it's important to not "module purge" here or else
# the script will hang
cp tiger_module_250723 ~/Modules/modulefiles/tiger3/250723
sed -i "s|_ENLIB_PATH|$REPO_DIR/_enlib|g" ~/Modules/modulefiles/tiger3/250723
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
uv cache clean
uv venv --python "$PYTHON_VERSION"
ln -sf .venv/bin/activate activate

# 6. Installation Phase
echo "Starting installation..."

# NumPy 1.x is required for ducc/pixell compatibility here
uv pip install "numpy<2" --no-binary numpy --no-cache-dir -Csetup-args=-Dblas=mkl-dynamic-ilp64-iomp -Csetup-args=-Dlapack=mkl-dynamic-ilp64-iomp -Csetup-args=-Duse-ilp64=true

# Install ducc from source (optimized local wheel)
cd "$INSTALL_DIR"
uv pip install ../repos/ducc --no-binary ducc0 --no-cache-dir

# Bulk install from requirements.in
uv pip install -r requirements.in 

# 7. Specialized Package Builds
uv pip install ../repos/optweight --no-cache-dir --no-build-isolation

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
uv pip install ../repos/pspy --no-cache-dir --no-build-isolation

# 8. Editable Installs
echo "Installing local packages in editable mode..."
uv pip install -e ../repos/sofind --no-cache-dir
uv pip install -e ../repos/mnms --no-cache-dir
uv pip install -e ../repos/pspipe_utils --no-cache-dir

# 9. Modify Slurm Template
echo "Configuring Slurm template at $REPO_DIR/PSpipe/project/SO/pISO/slurm/tiger.slurm"
SLURM_FILE="$REPO_DIR/PSpipe/project/SO/pISO/slurm/tiger.slurm"

# A. Expand $BASE_DIR on the output line
# We look for the literal string '$BASE_DIR' and replace it with the actual path
sed -i "s|BASE_DIR|$BASE_DIR|g" "$SLURM_FILE"

# B. Handle Email logic
if [ -n "$USER_EMAIL" ]; then
    echo "Adding email notifications for: $USER_EMAIL"
    
    # We create a block of text to insert
    EMAIL_BLOCK="#SBATCH --mail-type=begin        # send email when job begins\n#SBATCH --mail-type=end          # send email when job ends\n#SBATCH --mail-user=$USER_EMAIL"
    
    # Insert the block after the account line
    # 'a' command in sed appends text after the match
    sed -i "/#SBATCH --account=simonsobs/a $EMAIL_BLOCK" "$SLURM_FILE"
fi

[ -n "$USER_EMAIL" ] && echo "Slurm notifications set to: $USER_EMAIL"

# 10. Final Cleanup
cd "$BASE_DIR"
mkdir -p slurm_output

echo "Setup complete. Source the environment with: source activate"
