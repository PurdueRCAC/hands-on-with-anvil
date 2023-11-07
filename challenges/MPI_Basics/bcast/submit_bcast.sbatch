#!/bin/bash
#SBATCH -A cis230270
#SBATCH -J bcast
#SBATCH -o %x-%j.out
#SBATCH -t 0:10:00
#SBATCH -p wholenode
#SBATCH -N 1


srun -n4 -c1 ./run
