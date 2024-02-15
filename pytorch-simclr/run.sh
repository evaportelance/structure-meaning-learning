#!/bin/bash
#SBATCH --job-name=simclr
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --mail-user=eva.portelance@mila.quebec
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=48Gb
#SBATCH --partition=main

module load anaconda/3
conda activate py39-to113
python ./simclr.py --dataset 'abstractscenes' --batch-size 100 --num-epochs 500 --cosine-anneal --test-freq 5