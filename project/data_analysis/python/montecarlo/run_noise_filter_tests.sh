sbatch noise_filter_test.slurm paramfiles/global_dr6_v3_4pass_pa6a_A_NO.dict tiled A_NO yes 1
sbatch noise_filter_test.slurm paramfiles/global_dr6_v3_4pass_pa6a_A_NO.dict wavelet A_NO yes 1
sbatch noise_filter_test.slurm paramfiles/global_dr6_v3_4pass_pa6a_A_YES.dict tiled A_YES no 1
sbatch noise_filter_test.slurm paramfiles/global_dr6_v3_4pass_pa6a_A_YES.dict wavelet A_YES no 1

sbatch noise_filter_test.slurm paramfiles/global_dr6_v3_4pass_pa6a_W_NO.dict tiled W_NO yes 1
sbatch noise_filter_test.slurm paramfiles/global_dr6_v3_4pass_pa6a_W_NO.dict wavelet W_NO yes 1
sbatch noise_filter_test.slurm paramfiles/global_dr6_v3_4pass_pa6a_W_YES.dict tiled W_YES no 1
sbatch noise_filter_test.slurm paramfiles/global_dr6_v3_4pass_pa6a_W_YES.dict wavelet W_YES no 1
