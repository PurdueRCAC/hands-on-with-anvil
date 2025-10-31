#!/bin/bash
#SBATCH -A cis230270-gpu
#SBATCH -J pytorch_cnn
#SBATCH -o %x-%j.out
#SBATCH -t 0:30:00
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1

cd $SLURM_SUBMIT_DIR
date

module purge
module load modtree/gpu
module load anaconda

source activate /anvil/projects/x-cis230270/data/envs/torch-anvil

export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

python3 -u cnn.py

