#!/bin/bash
#SBATCH --job-name=clip_finetune
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32Gb
#SBATCH --partition=unkillable

module load anaconda/3
conda activate py39-to113
python ./clip_finetune.py --experiment_name "all_condition421" --num_epochs 1 --seed 421
python ./clip_finetune.py --experiment_name "all_condition982" --num_epochs 1 --seed 982
python ./clip_finetune.py --experiment_name "all_condition030" --num_epochs 1 --seed 30
python ./clip_finetune.py --experiment_name "all_condition374" --num_epochs 1 --seed 374

