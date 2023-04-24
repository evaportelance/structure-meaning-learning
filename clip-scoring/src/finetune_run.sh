#!/bin/bash
#SBATCH --job-name=clip_finetune
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32Gb
#SBATCH --time=48:00:00
#SBATCH --partition=unkillable

module load anaconda/3
conda activate py39-to113
python ./clip_finetune.py --experiment_name "struct-finetune" --num_epochs 2 --seed 421 --with_struct_loss
