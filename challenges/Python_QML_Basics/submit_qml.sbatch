#!/bin/bash
#SBATCH -A cis230270-gpu
#SBATCH -J qml_basics
#SBATCH -o %x-%j.out
#SBATCH -N 1
#SBATCH -t 0:20:00
#SBATCH -p gpu
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4

cd $SLURM_SUBMIT_DIR
date

# Load modules
module purge
module load modtree/gpu
module load anaconda

source activate /anvil/projects/x-cis230270/data/envs/qml-anvil

# Needed to bypass MIOpen, Disk I/O Errors
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache" 
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH} 
rm -rf ${MIOPEN_USER_DB_PATH} 
mkdir -p ${MIOPEN_USER_DB_PATH}

# Get address of head node
export MASTER_ADDR=$(hostname -i)

export OMP_NUM_THREADS=1

LD_PRELOAD=/usr/lib64/slurm/libslurmfull.so srun -n1 python3 -W ignore -u qml.py
