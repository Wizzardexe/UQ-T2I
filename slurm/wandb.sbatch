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
set -euo pipefail

date; hostname; id; pwd

export HF_HOME="$HOME/hf_cache"
unset HF_HUB_CACHE                     
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

date; hostname; id; pwd
nvidia-smi

.venv/bin/python -m uq_t2i.main