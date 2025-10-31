#!/bin/bash
#SBATCH -A cis230270-gpu
#SBATCH -J dgemm
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH --gpus-per-node=1
#SBATCH -p gpu
#SBATCH -N 1

module load modtree/gpu
module load openblas

srun -n1 -c1 --gpus-per-task=1 --gpu-bind=closest ./cpu_gpu_dgemm
