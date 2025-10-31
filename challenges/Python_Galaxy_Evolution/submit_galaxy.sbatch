#!/bin/bash
#SBATCH -A cis230270
#SBATCH -J galaxy
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p wholenode
#SBATCH -N 1
#SBATCH --ntasks=128

cd $SLURM_SUBMIT_DIR
date

module purge
module load modtree/cpu
module load hdf5/1.10.7
module load anaconda

source activate /anvil/projects/x-cis230270/data/envs/h5pympi-anvil

LD_PRELOAD=/usr/lib64/slurm/libslurmfull.so srun -n4 -c16 python3 -u galaxy.py
