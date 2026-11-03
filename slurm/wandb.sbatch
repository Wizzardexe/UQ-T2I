#!/bin/bash
#SBATCH --job-name=UQ
#SBATCH --comment="UQ in T2I"
#SBATCH --clusters=hlai
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=Adrian.Weghorst@campus.lmu.de
#SBATCH --mail-type=ALL
#SBATCH --chdir=/dss/dsshome1/03/ra24loy2/workspace/UQ-T2I/
#SBATCH --ntasks=1

date; hostname; id; pwd

source /dss/dsshome1/03/ra24loy2/miniconda3/bin/activate
conda activate py312
which python3.12

nvidia-smi

echo "Running with uv"
uv run src/uq_t2i/main.py 