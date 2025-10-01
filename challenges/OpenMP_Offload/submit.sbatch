#!/bin/bash
#SBATCH -A cis230270-gpu
#SBATCH -J mat_mul
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=1 
#SBATCH -p gpu-debug
#SBATCH -N 1


module purge
module load modtree/gpu
module load nvhpc
module load openblas

srun -n1 -c1 --gpus-per-task=1 ./matrix_multiply
#srun -n1 -c1 ./matrix_multiply
