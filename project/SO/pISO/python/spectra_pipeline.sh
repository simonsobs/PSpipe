#!/bin/bash
# Run this with an interactive allocation
# bash spectra_pipeline.sh {paramfile_path} {pre/post/test/ } {args}

PARAMFILE=$1

# read how many mpi processes are needed for different scripts
read Narr Carr allCarr Nspec Cspec allCspec Narr_SO Carr_SO allCarr_SO< <(
python - <<END
from pspy import so_dict
from pspipe_utils import pspipe_list
d = so_dict.so_dict()
d.read_from_file("$PARAMFILE")
n_arr = pspipe_list.get_arrays_list(d)[0]
n_arr_SO = len(d['arrays_SO'])
n_spec = pspipe_list.get_spectra_list(d)[0]
while n_spec >= 16:
    n_spec = (n_spec + 1) // 2
n_spec = (n_spec + 1) // 2
c_arr = 256 // n_arr
c_arr_SO = 256 // n_arr_SO
c_spec = 256 // n_spec
print(n_arr, c_arr, n_arr*c_arr, n_spec, c_spec, n_spec * c_spec, n_arr_SO, c_arr_SO, n_arr_SO*c_arr_SO)
END
)

echo Arrays mpi : n=$Narr c=$Carr $allCarr/256
echo SO Arrays mpi : n=$Narr_SO c=$Carr_SO $allCarr_SO/256
echo Spectras mpi : n=$Nspec c=$Cspec $allCspec/256

PYTHON_PATH="/pscratch/sd/m/merrydup/PSpipe_merry_piso/project/SO/pISO/python"

src_subtract() {
    MAPS_PATH=$1
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python $PYTHON_PATH/subtract_sources.py "$PARAMFILE" "$MAPS_PATH"
}

run_pre() {
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python $PYTHON_PATH/masks/get_xtra_mask.py $PARAMFILE
    OMP_NUM_THREADS=$Carr srun -n $Narr -c $Carr --cpu-bind=cores python $PYTHON_PATH/get_windows.py $PARAMFILE
    OMP_NUM_THREADS=$Cspec srun -n $Nspec -c $Cspec --cpu-bind=cores python $PYTHON_PATH/get_mcm_and_bbl.py $PARAMFILE
}

run_post() {
    OMP_NUM_THREADS=$Carr srun -n $Narr -c $Carr --cpu-bind=cores python $PYTHON_PATH/get_alms.py $PARAMFILE
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python $PYTHON_PATH/get_spectra_from_alms.py $PARAMFILE
    # OMP_NUM_THREADS=$Carr srun -n $Narr -c $Carr --cpu-bind=cores python $PYTHON_PATH/get_spectra_from_maps.py $PARAMFILE
}

run_plot_fit() {
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python $PYTHON_PATH/plot/plot_everything.py $PARAMFILE
    # OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python python/plot_everything_no_cross.py $PARAMFILE spectra python/plots_1019.yaml
    OMP_NUM_THREADS=256 srun -n 1 -c 256 --cpu-bind=cores python $PYTHON_PATH/plot/fit_noises.py $PARAMFILE
}

case "$2" in
    src_subtract)
        echo "Subtract sources from maps and copy everything at $3"
        src_subtract $3
        ;;
    pre)
        echo "Running get_windows.py and get_mcm_and_bbl.py"
        run_pre
        ;;
    post)
        echo "Running get_spectra_from_maps.py using existing mcms and windows folders."
        run_post
        ;;
    plot_fit)
        echo "Plot spectra and fit noise."
        run_plot_fit
        ;;
    test)
        ;;
    *)
        echo "Running whole spectra pipeline using $PARAMFILE"
        run_pre
        run_post
        ;;
esac