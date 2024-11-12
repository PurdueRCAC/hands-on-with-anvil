#!/bin/bash
#SBATCH -A cis230270
#SBATCH -p shared
#SBATCH -J galaxy
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -N 1
#SBATCH -n 8

date

module reset
module load anaconda/2024.02-py311
module list

conda activate py3.12-galaxy

srun python3 -u galaxy.py

date
