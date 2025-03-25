#!/bin/bash
#SBATCH --partition gpu
#SBATCH --cpus-per-task 12
#SBATCH --mem 20G
#SBATCH --time 0-1:30:00
#SBATCH --nodes 1
#SBATCH --gpus-per-node=v100:1

module purge
module load Python/3.10.8-GCCcore-12.2.0

source ~/.bashrc
cd /scratch/s4790820/biotech-news-sentiment/
source .venv/bin/activate

echo "Nvidia specs"
nvidia-smi

python notebooks/fine_tuning_3labels.py