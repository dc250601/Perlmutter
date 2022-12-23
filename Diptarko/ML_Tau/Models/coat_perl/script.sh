#!/bin/bash

#SBATCH -A ntrain
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=diptarko.choudhury@gmail.com


srun bash $HOME/Diptarko/ML_Tau/Models/coat_perl/data.sh
source $HOME/Diptarko/work/bin/activate 
srun python3 benchmark.py
