#!/bin/bash
#SBATCH -A cis230270
#SBATCH -p shared
#SBATCH -J h5py
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -N 1
#SBATCH -n 42

date

module reset
module load anaconda/2020.05-py38
module list

conda activate py3.10-galaxy

srun python3 -u hdf5_parallel.py

date
