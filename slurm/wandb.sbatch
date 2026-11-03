#!/bin/bash
#SBATCH --job-name=UQ
#SBATCH --comment="UQ in T2I"
#SBATCH --partition=NvidiaAll
#SBATCH --mail-user=user@uni-mail
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/u/user/workspace/CS4243_MiniProject/
#SBATCH --ntasks=1
date;hostname;id;pwd

echo 'activating virtual environment'
source .venv/bin/activate
echo "Python executable: $(which python)"

echo 'force login'
wandb login --relogin $WANDB_API_KEY

python 
