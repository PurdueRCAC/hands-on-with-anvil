#!/bin/bash
#SBATCH -A cis230270
#SBATCH -p batch
#SBATCH -J mpi4py
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -N 1
#SBATCH -n 42

date

module reset
module load anaconda/2021.05-py38

conda activate py3.10-galaxy

srun python3 -u hello_mpi.py

date
