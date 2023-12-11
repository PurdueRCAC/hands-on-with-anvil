#!/bin/bash 
#SBATCH -A cis230270
#SBATCH -J add_vec_cpu
#SBATCH -o %x-%j.out
#SBATCH -t 00:10:00
#SBATCH -p shared
#SBATCH -N 1

srun -n1 --ntasks-per-node=1 ./run 
