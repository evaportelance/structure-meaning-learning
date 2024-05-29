#!/bin/bash
#SBATCH --job-name=vcpcfg5
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



## In distribution training and test

### Joint model

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 527 

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 627

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 214

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s1018' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 1018

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 91


### Sem first model

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 527 --sem_first

python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 627 --sem_first

python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 214 --sem_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s1018' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 1018 --sem_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 91 --sem_first


### Syn first model

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 527 --syn_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 627 --syn_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 214 --syn_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s1018' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 1018 --syn_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 91 --syn_first


### joint gold

#python ./as_train.py --num_epochs 30 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 527

#python ./as_train.py --num_epochs 30 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 627

#python ./as_train.py --num_epochs 30 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 214

#python ./as_train.py --num_epochs 30 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s1018' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 1018

#python ./as_train.py --num_epochs 30 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 91