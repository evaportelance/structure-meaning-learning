#!/bin/bash
#SBATCH --job-name=vcpcfg-eval
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --mail-user=eva.portelance@mila.quebec
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32Gb
#SBATCH --partition=main

module load anaconda/3
conda activate py39-to113

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s1018/checkpoints/29.pth.tar' --out_file 'joint_1018_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s91/checkpoints/29.pth.tar' --out_file 'joint_91_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s214/checkpoints/29.pth.tar' --out_file 'joint_214_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s527/checkpoints/29.pth.tar' --out_file 'joint_527_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s627/checkpoints/29.pth.tar' --out_file 'joint_627_parses.json'


#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s1018/checkpoints/29.pth.tar' --out_file 'syn-first_1018_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s91/checkpoints/29.pth.tar' --out_file 'syn-first_91_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s214/checkpoints/29.pth.tar' --out_file 'syn-first_214_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s527/checkpoints/29.pth.tar' --out_file 'syn-first_527_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s627/checkpoints/29.pth.tar' --out_file 'syn-first_627_parses.json'


#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s1018/checkpoints/29.pth.tar' --out_file 'sem-first_1018_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s91/checkpoints/29.pth.tar' --out_file 'sem-first_91_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s214/checkpoints/29.pth.tar' --out_file 'sem-first_214_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s527/checkpoints/29.pth.tar' --out_file 'sem-first_527_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s627/checkpoints/29.pth.tar' --out_file 'sem-first_627_parses.json'


#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s1018/checkpoints/29.pth.tar' --out_file 'joint_gold_1018_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s91/checkpoints/29.pth.tar' --out_file 'joint_gold_91_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s214/checkpoints/29.pth.tar' --out_file 'joint_gold_214_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s527/checkpoints/29.pth.tar' --out_file 'joint_gold_527_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s627/checkpoints/29.pth.tar' --out_file 'joint_gold_627_parses.json'


#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s1018/checkpoints/14.pth.tar' --out_file 'joint_1018_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s91/checkpoints/14.pth.tar' --out_file 'joint_91_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s214/checkpoints/14.pth.tar' --out_file 'joint_214_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s527/checkpoints/14.pth.tar' --out_file 'joint_527_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s627/checkpoints/14.pth.tar' --out_file 'joint_627_parses.json'


#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s1018/checkpoints/14.pth.tar' --out_file 'syn-first_1018_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s91/checkpoints/14.pth.tar' --out_file 'syn-first_91_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s214/checkpoints/14.pth.tar' --out_file 'syn-first_214_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s527/checkpoints/14.pth.tar' --out_file 'syn-first_527_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s627/checkpoints/14.pth.tar' --out_file 'syn-first_627_parses.json'


#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s1018/checkpoints/14.pth.tar' --out_file 'sem-first_1018_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s91/checkpoints/14.pth.tar' --out_file 'sem-first_91_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s214/checkpoints/14.pth.tar' --out_file 'sem-first_214_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s527/checkpoints/14.pth.tar' --out_file 'sem-first_527_parses.json'

#python ./as_extra_evals.py --model_init '../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s627/checkpoints/14.pth.tar' --out_file 'sem-first_627_parses.json'


for epoch in {0..29..1}
do
    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s1018/checkpoints/$epoch.pth.tar" --out_file 'joint_1018_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s91/checkpoints/$epoch.pth.tar" --out_file 'joint_91_parses.json'

   python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s214/checkpoints/$epoch.pth.tar" --out_file 'joint_214_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s527/checkpoints/$epoch.pth.tar" --out_file 'joint_527_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_joint_indist_s627/checkpoints/$epoch.pth.tar" --out_file 'joint_627_parses.json'


   python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s1018/checkpoints/$epoch.pth.tar" --out_file 'syn-first_1018_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s91/checkpoints/$epoch.pth.tar" --out_file 'syn-first_91_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s214/checkpoints/$epoch.pth.tar" --out_file 'syn-first_214_parses.json'

python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s527/checkpoints/$epoch.pth.tar" --out_file 'syn-first_527_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_syn_first_indist_s627/checkpoints/$epoch.pth.tar" --out_file 'syn-first_627_parses.json'


    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s1018/checkpoints/$epoch.pth.tar" --out_file 'sem-first_1018_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s91/checkpoints/$epoch.pth.tar" --out_file 'sem-first_91_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s214/checkpoints/$epoch.pth.tar" --out_file 'sem-first_214_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s527/checkpoints/$epoch.pth.tar" --out_file 'sem-first_527_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_sem_first_indist_s627/checkpoints/$epoch.pth.tar" --out_file 'sem-first_627_parses.json'


    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s1018/checkpoints/$epoch.pth.tar" --out_file 'joint_gold_1018_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s91/checkpoints/$epoch.pth.tar" --out_file 'joint_gold_91_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s214/checkpoints/$epoch.pth.tar" --out_file 'joint_gold_214_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s527/checkpoints/$epoch.pth.tar" --out_file 'joint_gold_527_parses.json'

    python ./as_extra_evals.py --model_init "../../../scratch/vcpcfg/runs/2024-03-27_joint_gold_indist_s627/checkpoints/$epoch.pth.tar" --out_file 'joint_gold_627_parses.json'

done