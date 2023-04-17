#!/bin/bash
#SBATCH --job-name=clip_struct
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32Gb
#SBATCH --time=48:00:00
#SBATCH --partition=unkillable

module load anaconda/3
conda activate py39-to113
python ./clip_struct.py --dataset 'abstractscenes' --odir '../results/abstractscenes/parse_diff_results' --eval_data '../../../as_test_data.pkl' --parse_diff
