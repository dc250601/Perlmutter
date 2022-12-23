#!/bin/bash
#SBATCH -A ntrain
#SBATCH -q preempt
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=24:00:00
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=cdiptarko@gmail.com
#SBATCH --comment=196:00:00  #desired time limit
#SBATCH --requeue
#SBATCH --open-mode=append

srun bash $HOME/Diptarko/ML_Top/Models/coat_perl/data_small.sh
source $HOME/Diptarko/work/bin/activate
srun python3 $HOME/Diptarko/ML_Top/Models/coat_perl/trainer.py 100 0.0005 0.05 "Top_Small_Batch_256_coat_LR_0.0005" 256
