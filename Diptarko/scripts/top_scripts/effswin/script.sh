#!/bin/bash
#SBATCH -A ntrain
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cdiptarko@gmail.com
#SBATCH --open-mode=append

srun bash $HOME/Diptarko/ML_Top/Models/effswin/data.sh
source $HOME/Diptarko/work/bin/activate
srun python3 $HOME/Diptarko/ML_Top/Models/effswin/trainer.py 50 0.0001 0.05 "Top_224_effswin" 256 50 0.001 0.05
