#!/bin/bash
#SBATCH -A cis230270
#SBATCH -J h5py
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p wholenode
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load anaconda
conda activate py3.10-galaxy

srun -n42 python3 -u hdf5_parallel.py
