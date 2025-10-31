#!/bin/bash
#SBATCH -A cis230270
#SBATCH -J qlsa
#SBATCH -o "%x_%j".out
#SBATCH -N 1
#SBATCH -p shared
#SBATCH -t 00:10:00
#SBATCH --ntasks=1

cd $SLURM_SUBMIT_DIR
date

module purge
module load modtree/cpu
module load anaconda

source activate /anvil/projects/x-cis230270/data/envs/qlsa-solver-anvil

# HHL circuit generator
srun -N1 -n1 -c1 python circuit_HHL.py -case sample-tridiag -casefile input_vars.yaml --savedata

# Run on simulator
srun -N1 -n1 -c1 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000 -backtyp ideal --savedata

# Run on emulator
#srun -N1 -n1 -c1 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000 -backtyp real-iqm -backmet fake_garnet --savedata

# Run on real device (not available for SC25)
#source keys.sh 
#srun -N1 -n1 -c1 python solver.py -case sample-tridiag -casefile input_vars.yaml -s 1000 -backtyp real-iqm -backmet garnet --savedata

# Plot results
srun -N1 -n1 -c1 python plot_fidelity_vs_shots.py

