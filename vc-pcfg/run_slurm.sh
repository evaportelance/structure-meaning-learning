#!/bin/bash
#SBATCH --job-name=vcpcfg9
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

# python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-12_joint_mtalpha_1e-10_lmalpha_1_50ep' --vse_mt_alpha 0.01 --vse_lm_alpha 1.0 --one_shot

# python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-12_joint_mtalpha_1e-100_lmalpha_1_50ep' --vse_mt_alpha 0.001 --vse_lm_alpha 1.0 --one_shot

# python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-12_joint_mtalpha_1_lmalpha_1_50ep' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot

#python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-12_joint_mtalpha_1_lmalpha_0_50ep' --vse_mt_alpha 1.0 --vse_lm_alpha 0.0 --one_shot

#python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_synonly_oneshot' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot

#python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_synonly_all' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0

#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_synonly_oneshot_gold' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot

#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_synonlyallspans_all_gold' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0

### commented out short spans only in model_vis.py for these ones

## Joint model 1
#python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_oneshot' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot

#python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_all' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0

## Gold baseline 1
#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_oneshot_gold' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot

## Gold all baseline 1
#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_all_gold' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0

### Joint model 2-5

#python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_oneshot_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 627

#python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_oneshot_s1018' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 1018

#python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_oneshot_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 214

#python ./as_train.py --num_epochs 50 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_oneshot_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 91

### Gold baselines 2-5

### One shot
#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_oneshot_gold_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 627

#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_oneshot_gold_s1018' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 1018

#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_oneshot_gold_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 214

#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_oneshot_gold_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 91

### All 

#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_all_gold_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 627

#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_all_gold_s1018' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 1018

#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_all_gold_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 214

#python ./as_train.py --num_epochs 50 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-14_joint_mtalpha_1_lmalpha_1_50ep_allspans_all_gold_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 91

### Semantics first model
### need to change training so that matching loss only uses either bag of words or final embedding only train 25 epochs with alpha_lm = 0 then train 25 epochs with alpha_mt = 0

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-19_sem_first_oneshot_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 527 --sem_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-19_sem_first_oneshot_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 627 --sem_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-19_sem_first_oneshot_s1018' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 1018 --sem_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-19_sem_first_oneshot_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 214 --sem_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-19_sem_first_oneshot_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 91 --sem_first



### Syntax first model
### need to change training so that matching loss only uses either bag of words or final embedding only train 25 epochs with alpha_mt = 0 then train 25 epochs with alpha_mt = 0

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-19_syn_first_oneshot_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 527 --syn_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-19_syn_first_oneshot_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 627 --syn_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-19_syn_first_oneshot_s1018' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 1018 --syn_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-19_syn_first_oneshot_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 214 --syn_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-19_syn_first_oneshot_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 91 --syn_first


## Hyper param search
### --t_states 60 --nt_states 30

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-25_syn_first_oneshot_t40_nt20_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 527 --syn_first --t_states 40 --nt_states 20

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-25_syn_first_oneshot_t40_nt20_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 627 --syn_first --t_states 40 --nt_states 20

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-25_syn_first_oneshot_t40_nt20_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 214 --syn_first --t_states 40 --nt_states 20

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-25_joint_oneshot_t40_nt20_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 627 --t_states 40 --nt_states 20

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-25_joint_oneshot_t40_nt20_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 527 --t_states 40 --nt_states 20

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-25_joint_oneshot_t40_nt20_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 214 --t_states 40 --nt_states 20

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-25_sem_first_oneshot_t40_nt20_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 627 --t_states 40 --nt_states 20 --sem_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-25_sem_first_oneshot_t40_nt20_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 527 --t_states 40 --nt_states 20 --sem_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-25_sem_first_oneshot_t40_nt20_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --one_shot --seed 214 --t_states 40 --nt_states 20 --sem_first


## In distribution training and test

### Joint model

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 527

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 627

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 214

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s1018' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 1018

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 91


### Sem first model

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s527' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 527 --sem_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s627' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 627 --sem_first

#python ./as_train.py --num_epochs 30 --encoder_file "all_as-resn-50.npy" --img_dim 2048 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s214' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 214 --sem_first

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

python ./as_train.py --num_epochs 30 --encoder_file "all_flat_features_gold.npy" --img_dim 756 --log_step 1000 --visual_mode --logger_name '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s91' --vse_mt_alpha 1.0 --vse_lm_alpha 1.0 --seed 91