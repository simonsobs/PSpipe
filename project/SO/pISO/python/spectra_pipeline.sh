#!/bin/bash

PARAMFILE="paramfiles/global_dr6v4xlegacyxso_ISO_1024_coadd.dict"
PARAMFILE="paramfiles/global_dr6v4xlegacyxso_ISO_1024.dict"

# the number of ntasks is not optimal as different scripts may have different number of optimal mpi tasks.
run_pre() {
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python python/get_masks_from_ivar.py $PARAMFILE
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python python/get_windows.py $PARAMFILE
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python python/get_mcm_and_bbl.py $PARAMFILE
}

run_post() {
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python python/get_alms.py $PARAMFILE
    OMP_NUM_THREADS=48 srun -n 5 -c 48 --cpu-bind=cores python python/get_spectra_from_alms.py $PARAMFILE
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python python/plot_everything.py $PARAMFILE spectra python/plots_1019.yaml
    # OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python python/plot_everything_no_cross.py $PARAMFILE spectra python/plots_1019.yaml
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python python/fit_noises.py $PARAMFILE spectra
}

case "$1" in
    pre)
        echo "Running get_windows.py and get_mcm_and_bbl.py"
        run_pre
        ;;
    post)
        echo "Running get_alm.py and get_spectra_from_alms.py using existing mcms and windows folders."
        run_post
        ;;
    *)
        echo "Running whole spectra pipeline using $PARAMFILE"
        run_pre
        run_post
        ;;
esac