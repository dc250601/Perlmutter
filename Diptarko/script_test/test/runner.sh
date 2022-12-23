#!/bin/bash
#BATCH -A ntrain
#SBATCH -q debug
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH --time=00:06:00
timeout 1m srun python3 pause.py
