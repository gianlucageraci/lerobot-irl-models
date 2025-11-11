#!/bin/bash
#SBATCH -p accelerated               # Partition
#SBATCH --gres=gpu:4                 # 4 GPUs anfordern
#SBATCH --mem=64G                    # RAM
#SBATCH --time=6:00:00               # Maximale Laufzeit
#SBATCH -J train_flower              # Jobname
#SBATCH -o logs/%x_%j.out            # STDOUT-Log
#SBATCH -e logs/%x_%j.err            # STDERR-Log

# Umgebung vorbereiten
source ~/.bashrc
conda activate lerobot-irl-models


torchrun --nproc_per_node=4 src/train_flower.py

# Alternative Option, falls torchrun nicht passt:
# python -m torch.distributed.launch --nproc_per_node=4 src/train_flower.py
