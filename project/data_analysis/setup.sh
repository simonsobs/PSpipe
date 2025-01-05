slurm_account=mp107b
export SBATCH_ACCOUNT=${slurm_account}
export SALLOC_ACCOUNT=${slurm_account}

# newgrp act
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
  # Remove cached wheels
  python -m pip cache purge
  python -m pip install -U pip wheel setuptools ipython
  python -m pip install pspy==1.8.0
  module swap PrgEnv-${PE_ENV,,} PrgEnv-gnu
  MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py
fi

source ${pyenv_dir}/bin/activate

function install_enlib() {
  (
      cd ${software_dir}
      git clone git@github.com:amaurea/tenki.git
      git clone git@github.com:amaurea/enlib.git
      if [ ${USE_ENLIB:-0} -eq 1 ]; then
        # We may need to downgrade setuptools to install these things
        cd enlib
        ENLIB_COMP=cca_intel make array_ops
      fi
  )
}

function install_local_module() {
  git_path=${1,,}
  git_branch=$2
  install_dir=$(echo ${git_path} | sed 's/.*\/\(.*\)\.git/\1/')
  (
    cd ${software_dir}
    [[ ! -z ${install_dir} ]] && git clone ${git_path}
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
  install_local_module git@github.com:simonsobs/pspipe_utils.git
  install_local_module git@github.com:simonsobs/PSpipe.git
  install_local_module git@github.com:AdriJD/optweight.git
  install_local_module git@github.com:simonsobs/sofind.git
  install_local_module git@github.com:simonsobs/mnms.git

  # Reinstall ducc and compile it to take advantage of NERSC hardware
  python -m pip uninstall -y ducc0; python -m pip install --no-binary ducc0 ducc0

  # To keep track of what have been installed
  pip freeze > ${cwd}/requirements.txt
fi

export TENKI_PATH=${software_dir}/tenki
export SOFIND_SYSTEM=perlmutter
export PYTHONPATH=$PYTHONPATH:${software_dir}


echo "Installation & loading of virtual env. done"
