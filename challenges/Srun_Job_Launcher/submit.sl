#!/bin/bash
#SBATCH -A cis230270
#SBATCH -J srun_<enter your name here>
#SBATCH -o %x-%j.out
#SBATCH -t 10:00
#SBATCH -p wholenode
#SBATCH -N 1

# number of OpenMP threads
export OMP_NUM_THREADS=1

# srun command to modify 
srun -N 1 -n 1 -c 1 ./hello_mpi_omp | sort
