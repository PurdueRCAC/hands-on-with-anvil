#!/bin/bash
#SBATCH -A cis230270
#SBATCH -J h5py
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p shared
#SBATCH -N 1
#SBATCH --ntasks=42

cd $SLURM_SUBMIT_DIR
date

module purge
module load modtree/cpu
module load hdf5/1.10.7
module load anaconda

source activate /anvil/projects/x-cis230270/data/envs/h5pympi-anvil

LD_PRELOAD=/usr/lib64/slurm/libslurmfull.so srun -n42 python3 -u hdf5_parallel.py
