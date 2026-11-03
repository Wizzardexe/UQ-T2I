cd ~/workspace/UQ-T2I
#echo 'Enter your WANDB Token' && read -s WANDB_API_KEY
#sbatch --export=WANDB_API_KEY=$WANDB_API_KEY slurm/wandb.sbatch
sbatch --clusters=hlai slurm/run.sbatch