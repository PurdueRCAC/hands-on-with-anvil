#!/bin/bash
#SBATCH -A cis230270
#SBATCH -J pytorch_cnn
#SBATCH -o %x-%j.out
#SBATCH -t 1:00:00
#SBATCH -p wholenode
#SBATCH -N 1

date

module reset
module load anaconda/2024.02-py311
module list

conda activate py3.12-torch
python3 -u cnn.py

date
