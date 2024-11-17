#!/bin/bash
#SBATCH -A cis230270
#SBATCH -J galaxy
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p wholenode
#SBATCH -N 1

unset SLURM_EXPORT_ENV

date

module load anaconda
conda activate py3.10-galaxy

srun -n8 python3 -u galaxy.py
