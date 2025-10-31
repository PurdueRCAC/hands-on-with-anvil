#!/bin/bash
#SBATCH -A cis230270
#SBATCH -J cupy_xfer
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p wholenode
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load PrgEnv-gnu
module load rocm/5.3.0
module load craype-accel-amd-gfx90a

source ~/miniconda-anvil-handson/bin/activate base
source activate ~/.conda/envs/cupy-anvil

srun -n1 --gpus-per-task=3 --gpu-bind=closest python3 data_transfer.py
