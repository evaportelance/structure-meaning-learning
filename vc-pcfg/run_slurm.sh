#!/bin/bash
#SBATCH --job-name=vcpcfg3
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --mail-user=eva.portelance@mila.quebec
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32Gb
#SBATCH --partition=long

module load anaconda/3
conda activate py39-to113

# python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-12_joint_mtalpha_1e-10_lmalpha_1_50ep' --vse_mt_alpha 0.01 --vse_lm_alpha 1.0

# python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-12_joint_mtalpha_1e-100_lmalpha_1_50ep' --vse_mt_alpha 0.001 --vse_lm_alpha 1.0

# python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-12_joint_mtalpha_1_lmalpha_1_50ep' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0

python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-12_joint_mtalpha_1_lmalpha_0_50ep' --vse_mt_alpha 1.0 --vse_lm_alpha 0.0