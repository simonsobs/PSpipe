module load python
module load intel
module load texlive

# Both working in bash/zsh
cwd=$(dirname ${BASH_SOURCE[0]:-${(%):-%x}})
base_dir=$(realpath ${BASE_DIR:-${cwd}})

pyenv_dir=${base_dir}/pyenv/dr6
if  [ ! -d ${pyenv_dir} ]; then
  echo "Creating virtual env in '${pyenv_dir}'..."
  python -m venv ${pyenv_dir}
  source ${pyenv_dir}/bin/activate
  # Remove cached wheels to avoid conflict with older wheels possibly made with different setup
  python -m pip cache purge
  python -m pip install -U pip wheel setuptools ipython
  python -m pip install pspy==1.8.0
  # Following https://docs.nersc.gov/development/languages/python/parallel-python/#mpi4py-in-your-custom-conda-environment
  module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
  MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
fi

source ${pyenv_dir}/bin/activate

function install_enlib() {
  (
    amaurea_dir=${software_dir}/amaurea
    mkdir -p ${amaurea_dir}; cd ${amaurea_dir}
    git clone git@github.com:amaurea/tenki.git
    git clone git@github.com:amaurea/enlib.git
    if [ ${USE_ENLIB:-0} -eq 1 ]; then
      # We may need to downgrade setuptools to install these things
      cd enlib
      ENLIB_COMP=cca_intel make array_ops
    fi
  )
}

function install_cosmorec() {
  (
    cosmorec_release="CosmoRec.v2.0.3b"
    cd ${software_dir}
    wget https://www.cita.utoronto.ca/~jchluba/Recombination/_Downloads_/${cosmorec_release}.tar.gz -P /tmp &&
    tar -xvf /tmp/${cosmorec_release}.tar.gz
    (
      cd ${cosmorec_release}
      sed -i 's/^CXXFLAGS =.*/CXXFLAGS = -Wall -pedantic -O2 -fPIC/g' Makefile.in
      make all
    )
    # Tell camb that cosmorec is available and recompile
    (
      cd camb/fortran
      # Use @ to avoid clash with / from path
      sed -i 's@^\(RECOMBINATION_FILES ?=.*\)@\1 cosmorec\nCOSMOREC_PATH = '${software_dir}/${cosmorec_release}'@' Makefile_main
      cd ..; python setup.py make
    )
  )
}

function install_local_module() {
  git_path=${1,,}
  git_branch=$2
  install_dir=$(echo ${git_path} | sed 's/.*\/\(.*\)\.git/\1/')
  (
    cd ${software_dir}
    [[ ! -z ${install_dir} ]] && git clone --recursive ${git_path}
    cd ${install_dir}
    [[ ! -z ${git_branch} ]] && git checkout ${git_branch} -b ${git_branch}
    python -m pip install -e .
  )
}

software_dir=${base_dir}/software
if [ ! -d ${software_dir} ]; then
  echo "Installing local software in '${software_dir}'..."
  mkdir -p ${software_dir}

  install_enlib
  # Need proper installation process (see https://github.com/simonsobs/pspy?tab=readme-ov-file#installing)
  # install_local_module git@github.com:simonsobs/pspy.git v1.8.0

  install_local_module git@github.com:cmbant/CAMB.git 1.5.9
  install_local_module git@github.com:simonsobs/pspipe_utils.git
  install_local_module git@github.com:simonsobs/PSpipe.git
  install_local_module git@github.com:AdriJD/optweight.git
  install_local_module git@github.com:simonsobs/sofind.git
  install_local_module git@github.com:simonsobs/mnms.git
  install_local_module git@github.com:ACTCollaboration/act_dr6_mflike.git
  install_local_module git@github.com:ACTCollaboration/dr6-cmbonly.git

  # After camb installation (see above), we recompile it with CosmoRec support
  install_cosmorec

  # Reinstall ducc and compile it to take advantage of NERSC hardware
  python -m pip uninstall -y ducc0; python -m pip install --no-binary ducc0 ducc0

  # To keep track of what have been installed
  python -m pip freeze > ${cwd}/requirements.txt
fi

slurm_account=mp107b
export SBATCH_ACCOUNT=${slurm_account}
export SALLOC_ACCOUNT=${slurm_account}

export SOFIND_SYSTEM=${SOFIND_SYSTEM:-perlmutter}
# Set python path to find Sigurd's python scripts
export TENKI_PATH=${software_dir}/amaurea/tenki
export PYTHONPATH=$PYTHONPATH:${software_dir}/amaurea

export BASE_DIR=${base_dir}
echo "Installation & loading of virtual env. done"
