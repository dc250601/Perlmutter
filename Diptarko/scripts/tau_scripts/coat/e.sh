#!BATCH -A ntrain
#SBATCH -q regular
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -N 1
#SBATCH --time=12:00:00
#SBATCH -c 128
#SBATCH --gpus-per-task=1
#SBATCH --mail-type=begin,end
#SBATCH --mail-user=cdiptarko@gmail.com
#SBATCH --open-mode=append

srun bash $HOME/Diptarko/ML_Tau/Models/coat_perl/data.sh
source $HOME/Diptarko/work/bin/activate

start=$(($(date +%H)+$(($(date +%j)*24))))
timeout 11h srun python3 $HOME/Diptarko/ML_Tau/Models/coat_perl/trainer.py 100 0.0005 0.005 "A5" 128

end=$(($(date +%H)+$(($(date +%j)*24))))

tot=$((end-start))
if [ $tot > 5 ]; then
	sbatch e.sh 
fi

