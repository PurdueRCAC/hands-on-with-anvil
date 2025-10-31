#!/bin/bash
#SBATCH -A cis230270-gpu
#SBATCH -J cupy_xfer
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=3

cd $SLURM_SUBMIT_DIR
date

module purge
module load modtree/gpu
module load anaconda

source activate /anvil/projects/x-cis230270/data/envs/cupy-anvil

srun -n1 python3 -u data_transfer.py
